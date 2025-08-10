import re
import json
from collections import defaultdict

LOGFILE = "logs/4th_Submission.log"

def parse_log(logfile):
    with open(logfile, "r") as f:
        lines = f.readlines()

    requests = []
    current = None
    question_map = {}
    answer_map = {}
    stats_map = defaultdict(lambda: defaultdict(list))
    errors_map = defaultdict(list)
    warnings_map = defaultdict(list)

    for line in lines:
        # API Request
        if "API Request to /api/v1/hackrx/run" in line:
            m = re.search(r'API Request to /api/v1/hackrx/run: (\{.*?\})', line)
            if m:
                try:
                    req = json.loads(m.group(1))
                    print(f"[DEBUG] Parsed API Request: {req}")
                except Exception as e:
                    print(f"[ERROR] Could not parse API Request JSON: {e}\nLine: {line}")
                    continue
                current = {
                    "request_id": req["request_id"],
                    "documents": [req["document_url"]],
                    "questions": req["questions"],
                    "answers": [],
                    "stats": {},
                    "improvement_notes": [],
                }
                requests.append(current)
                question_map[req["request_id"]] = req["questions"]
            else:
                print(f"[WARN] Could not find JSON in API Request line: {line}")
        # API Response
        elif "API Response from /api/v1/hackrx/run" in line:
            m = re.search(r'API Response from /api/v1/hackrx/run.*: (\{.*?\})', line)
            if m:
                try:
                    resp = json.loads(m.group(1))
                except Exception as e:
                    print(f"[ERROR] Could not parse API Response JSON: {e}\nLine: {line}")
                    continue
                rid = resp["request_id"]
                for req in requests:
                    if req["request_id"] == rid:
                        req["answers"] = resp["answers"]
                        req["stats"]["total_processing_time_seconds"] = resp["processing_time_seconds"]
            else:
                print(f"[WARN] Could not find JSON in API Response line: {line}")
# Document info
        elif "download_complete" in line:
            m = re.search(r'Document downloaded successfully - Details: (.*)', line)
            if m:
                det = json.loads(m.group(1))
                rid = None
                # Find the current request by matching document_id
                for req in requests:
                    if det["document_id"] in line:
                        rid = req["request_id"]
                        req["stats"]["document_size_bytes"] = det["size_bytes"]
                        req["stats"]["document_type"] = det["content_type"]
        # Chunking info
        elif "rechunking_complete" in line:
            m = re.search(r'Completed text rechunking process - Details: (.*)', line)
            if m:
                det = json.loads(m.group(1))
                for req in requests:
                    if det["rechunk_id"] in line:
                        req["stats"]["chunks_created"] = det["new_chunks"]
        # Token info
        elif "context_token_estimation" in line:
            m = re.search(r'Estimated token count for context - Details: (.*)', line)
            if m:
                det = json.loads(m.group(1))
                for req in requests:
                    if det["question_id"] in line:
                        req["stats"]["tokens_in_context"] = det["estimated_tokens"]
                        req["stats"]["tokens_with_prompt"] = det["estimated_tokens_with_prompt"]
        # LLM response time
        elif "gemini_response_received" in line:
            m = re.search(r'Received response from Gemini API - Details: (.*)', line)
            if m:
                det = json.loads(m.group(1))
                for req in requests:
                    req["stats"].setdefault("llm_response_times_seconds", []).append(det["duration_seconds"])
        # Answer length
        elif "question_answered" in line:
            m = re.search(r'Question processing completed - Details: (.*)', line)
            if m:
                det = json.loads(m.group(1))
                for req in requests:
                    req["stats"].setdefault("answer_lengths", []).append(det["answer_length"])
        # Errors/warnings
        elif "unknown_file_extension" in line:
            for req in requests:
                if "unknown_file_extension" in line:
                    req["stats"].setdefault("warnings", []).append("Unknown file extension, treated as PDF")
        elif "error" in line.lower():
            for req in requests:
                req["stats"].setdefault("errors", []).append(line.strip())

    # Add improvement notes
    for req in requests:
        notes = []
        stats = req["stats"]
        if stats.get("document_type", "").startswith("image"):
            notes.append("Image files are being treated as PDFs; consider adding OCR-specific handling.")
        if stats.get("chunks_created", 0) == 1:
            notes.append("Only one chunk created from document—may limit retrieval accuracy.")
        if stats.get("tokens_in_context", 0) < 100:
            notes.append("Context size is small; answers may be limited by available text.")
        if stats.get("warnings"):
            notes.extend(stats["warnings"])
        if not stats.get("errors"):
            notes.append("No errors, but reranking is skipped—could improve answer quality if enabled.")
        req["improvement_notes"] = notes

    print(f"[INFO] Parsed {len(requests)} requests from log.")
    print(f"[DEBUG] Final requests: {json.dumps(requests, indent=2)}")
    return requests
def save_report(report_data, output_file):
    """
    Save the report data to a file in JSON format.
    Parameters:
        report_data (Any): The data to save (usually a list or dict).
        output_file (str): Path to the output file.
    """
    with open(output_file, 'w') as f:
        json.dump(report_data, f, indent=2)

if __name__ == "__main__":
    report = parse_log(LOGFILE)
    output_path = "logs/4th_Submission.json"
    save_report(report, output_path)
    print(f"Report saved to {output_path}")
