
import os
import traceback
import time
import torch
import weaviate
from flask import Flask, request, jsonify, render_template
from llmsherpa.readers import LayoutPDFReader
from transformers import AutoTokenizer, AutoModel
from weaviate.exceptions import UnexpectedStatusCodeException

# --- Configuration ---
# Set up Flask app
app = Flask(__name__, template_folder="templates")

# Get environment variables for service URLs
LLMSHERPA_API_URL = os.environ.get("LLMSHERPA_API_URL", "http://localhost:5001")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
HUGGING_FACE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


# --- Query Endpoint ---

@app.route('/query', methods=['POST'])
def query_chunks():
    """
    Accepts a JSON body with a 'query' field, embeds it, and returns the top 4 most similar chunks from Weaviate.
    """
    if not client or not client.is_ready():
        return jsonify({"error": "Weaviate service is not available or not ready"}), 503

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body."}), 400
    query_text = data['query']

    # Generate embedding for the query
    query_vec = get_embedding(query_text)

    # Query Weaviate for top 4 most similar chunks
    try:
        result = client.query.get("Chunk", ["tag", "text", "source_file"]).with_near_vector({"vector": query_vec}).with_limit(4).do()
        top_chunks = result.get('data', {}).get('Get', {}).get('Chunk', [])
        return jsonify({
            "query": query_text,
            "results": top_chunks
        })
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
# --- Global Variables & Initialization (Run once on startup) ---

# 1. Initialize llmsherpa reader
print("Initializing llmsherpa PDF reader...")
pdf_reader = LayoutPDFReader(LLMSHERPA_API_URL)
print("llmsherpa PDF reader initialized.")

# 2. Download and cache the Hugging Face model
# This is now done once when the application starts, not per request.
# The 'transformers' library automatically caches the model to your local disk.
print(f"Loading and caching model: '{HUGGING_FACE_MODEL_NAME}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(HUGGING_FACE_MODEL_NAME)
    model = AutoModel.from_pretrained(HUGGING_FACE_MODEL_NAME)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print(f"Fatal: Could not load Hugging Face model. {e}")
    # Exit if model can't be loaded, as it's critical for the app's function.
    exit()

# 3. Initialize Weaviate client
print(f"Connecting to Weaviate at {WEAVIATE_URL}...")
client = None
try:
    client = weaviate.Client(WEAVIATE_URL, timeout_config=(5, 15))  # Connect with a timeout
    print("Weaviate client initialized.")
except Exception as e:
    print(f"Warning: Could not connect to Weaviate. The service may be unavailable. {e}")
    # The app can still run, but /extract_data will fail.

# --- Helper Functions ---

def get_embedding(text: str) -> list[float]:
    """Generates a normalized embedding for a given text using the pre-loaded model."""
    # Encode the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling to get a sentence-level embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Normalize the embeddings to a unit vector
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
    return normalized_embeddings[0].tolist()


def initialize_weaviate_schema(weaviate_client: weaviate.Client):
    """
    Ensures the 'Chunk' class exists in the Weaviate schema.
    Includes a robust retry mechanism to handle the 'leader not found' error during startup.
    """
    if not weaviate_client:
        print("Weaviate client is not available. Skipping schema initialization.")
        return

    class_obj = {
        "class": "Chunk",
        "description": "Stores text chunks extracted from documents.",
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "tag", "dataType": ["text"]},
            {"name": "source_file", "dataType": ["text"]},
        ],
    }

    max_retries = 6
    retry_delay_seconds = 10 # Increased delay to give Weaviate more time
    
    for attempt in range(max_retries):
        try:
            # Attempt to get the class to see if it exists
            weaviate_client.schema.get(class_name="Chunk")
            print(f"Success: 'Chunk' class already exists in Weaviate.")
            return  # Exit successfully

        except UnexpectedStatusCodeException as e:
            # Case 1: Leader not found (transient error). We need to wait and retry.
            if 'leader not found' in str(e).lower() or e.status_code == 500:
                print(f"Attempt {attempt + 1}/{max_retries}: Weaviate not ready (leader not found). Retrying in {retry_delay_seconds}s...")
                time.sleep(retry_delay_seconds)
                continue # Go to the next iteration of the loop

            # Case 2: Class does not exist (404 Not Found). We need to create it.
            elif e.status_code == 404:
                print(f"Attempt {attempt + 1}/{max_retries}: 'Chunk' class not found. Creating now...")
                try:
                    weaviate_client.schema.create_class(class_obj)
                    print("Success: 'Chunk' class created.")
                    return # Exit successfully
                except Exception as create_e:
                    print(f"Fatal: Failed to create 'Chunk' class after finding it was missing. Error: {create_e}")
                    return # Exit with failure

            # Case 3: Another unexpected HTTP error.
            else:
                print(f"Fatal: An unexpected HTTP error occurred: {e}")
                print(traceback.format_exc())
                return # Exit with failure

        except Exception as e:
            # Catch other exceptions like a connection refusal
            print(f"Attempt {attempt + 1}/{max_retries}: An unexpected connection error occurred: {e}. Retrying in {retry_delay_seconds}s...")
            time.sleep(retry_delay_seconds)

    print("Fatal: Could not initialize Weaviate schema after multiple retries. Please check the Weaviate container logs.")           
            
# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')


@app.route('/extract_data', methods=['POST'])
def extract_data():
    """
    Receives a PDF, extracts chunks, generates embeddings, and stores them in Weaviate.
    """
    if not client or not client.is_ready():
        return jsonify({"error": "Weaviate service is not available or not ready"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if not (file and file.filename.endswith('.pdf')):
        return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400
        
    try:
        start_time = time.time()
        
        pdf_bytes = file.read()
        
        # Use the pre-initialized PDF reader
        print(f"Processing '{file.filename}' with llmsherpa...")
        doc = pdf_reader.read_pdf(file.filename, contents=pdf_bytes)
        
        extracted_data_response = []
        all_chunks = list(doc.chunks())
        print(f"Extracted {len(all_chunks)} chunks from the PDF. Starting embedding and storage.")

        # Batch process chunks for efficient import into Weaviate
        with client.batch as batch:
            batch.batch_size = 100  # Process 100 chunks at a time
            for i, chunk in enumerate(all_chunks):
                
                # **3. Print extracted text**
                chunk_text = chunk.to_text()
                print(f"\n--- Chunk {i + 1}/{len(all_chunks)} ---")
                print(f"TAG: {chunk.tag}")
                print(f"TEXT: {chunk_text}")

                # Generate embedding using the pre-loaded global model
                chunk_embedding = get_embedding(chunk_text)
                
                # **4. Print the vector** (snippet for readability)
                print(f"VECTOR (first 5 dimensions): {chunk_embedding[:5]}...")

                # Prepare the data object for Weaviate
                data_object = {
                    'text': chunk_text,
                    'tag': chunk.tag,
                    'source_file': file.filename
                }
                
                # Add object to the Weaviate batch
                batch.add_data_object(
                    data_object=data_object,
                    class_name="Chunk",
                    vector=chunk_embedding
                )

                # Append data to the JSON response list
                extracted_data_response.append({
                    'tag': chunk.tag,
                    'text': chunk_text,
                    'embedding': chunk_embedding
                })
        
        print("\n--- Processing Complete ---")
        print(f"Successfully processed and stored {len(extracted_data_response)} chunks in Weaviate.")
        total_time = time.time() - start_time
        print(f"Total request time: {total_time:.2f} seconds.")

        return jsonify({"data": extracted_data_response})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# --- Main Execution Block ---

if __name__ == '__main__':
    # Ensure Weaviate schema is initialized before starting the app server
    initialize_weaviate_schema(client)
    
    # Run the Flask app
    # Use debug=False in a production environment
    app.run(host='0.0.0.0', port=1212, debug=True)