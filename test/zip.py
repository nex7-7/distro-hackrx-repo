"""
Test for Main RAG API Endpoint

This test sends a POST request to the /api/v1/hackrx/run/ endpoint with a sample payload and prints the response.
"""
import requests
import json


def test_rag_api():
    url = "http://localhost:8000/api/v1/hackrx/run/"
    token = "fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "documents": "https://superman-rx.s3.eu-north-1.amazonaws.com/policy.pdf.zip",
        "questions": [
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:", response.json())
    except Exception:
        print("Response Text:", response.text)


if __name__ == "__main__":
    test_rag_api()
