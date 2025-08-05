import asyncio
from src.components.data_ingestion import process_documents_simple

async def main():
    """Main function to run the document ingestion pipeline."""
    print("Starting document ingestion pipeline...")
    
    try:
        result = await process_documents_simple("./data/raw")
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print("="*50)
        print(f"✅ Successfully processed: {result.successful_documents} documents")
        print(f"📊 Total chunks created: {result.total_chunks}")
        print(f"❌ Failed documents: {result.failed_documents}")
        print(f"⏱️  Total processing time: {result.processing_time:.2f} seconds")
        
        if result.successful_documents > 0:
            print(f"📈 Average chunks per document: {result.total_chunks / result.successful_documents:.1f}")
            print(f"🚀 Processing rate: {result.successful_documents / result.processing_time:.2f} docs/sec")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("Check the logs for more detailed error information.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())