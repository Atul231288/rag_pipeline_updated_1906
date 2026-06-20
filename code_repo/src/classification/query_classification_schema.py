from pydantic import BaseModel
class QueryClassification(BaseModel):
    domain: str
    dataset: str
    retrieval_required: bool
    reasoning: str