
# LLM Document Processing System

Build a system that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails.


## Objective

The system should take an input query like:

"46-year-old male, knee surgery in Pune, 3-month-old insurance policy"


It must then:

- Parse and structure the query to identify key details such as age, procedure, location, and policy duration.

- Search and retrieve relevant clauses or rules from the provided documents using semantic understanding rather than simple keyword matching.

- Evaluate the retrieved information to determine the correct decision, such as approval status or payout amount, based on the logic defined in the clauses.

- Return a structured JSON response containing: Decision (e.g., approved or rejected), Amount (if applicable), and Justification, including mapping of each decision to the specific clause(s) it was based on.


## Requirements

- Input documents may include PDFs, Word files, or emails.

- The query processing and retrieval must work even if the query is vague, incomplete, or written in plain English.

- The system must be able to explain its decision by referencing the exact clauses used from the source documents.

- The output should be consistent, interpretable, and usable for downstream applications such as claim processing or audit tracking.


## Applications

This system can be applied in domains such as insurance, legal compliance, human resources, and contract management.


## Sample Query

"46M, knee surgery, Pune, 3-month policy"


## Sample Response

"Yes, knee surgery is covered under the policy."

## API Mechanism

### Base URL (Local Development):
http://localhost:8080/api/v1

Authentication:
Authorization: Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08
✅ Team token loaded successfully

POST
/hackrx/run
Run Submissions

### Sample Upload Request:
POST /hackrx/run
Content-Type: application/json
Accept: application/json
Authorization: Bearer fd8defb3118175da9553e106c05f40bc476971f0b46a400db1e625eaffa1fc08

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

### Sample Response:

{
"answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    ]
}

### Instructions for API Mechaism
- Use FAST API
- Speed is Key. We need this to be as fast as possible without any wasted time