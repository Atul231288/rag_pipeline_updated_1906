from langchain_core.prompts import PromptTemplate
CLASSIFICATION_TEMPLATE = """
You are an expert query classifier.
Classify the query according to:
1. DOMAIN
2. DATASET
3. QUERY TYPE
4. RETRIEVAL REQUIRED

----------------------------------
RETRIEVAL REQUIRED
----------------------------------

NO:
- Greetings
- Math calculations
- Common knowledge

YES:
- Domain knowledge
- Specific factual information
- Legal documents
- Financial reports
- Biomedical evidence

----------------------------------
DOMAINS & DATASETS
----------------------------------

LEGAL
- cuad

CUSTOMER_SUPPORT
- delusionqa
- emanual
- techqa

FINANCE
- finqa
- tatqa

BIOLOGICAL
- pubmedqa
- covidqarag

GENERAL_KNOWLEDGE
- hotpotqa
- msmarco
- hagrid
- experqa

Return JSON only.

{format_instructions}

Query:
{query}

Provide consise reasoning.
"""