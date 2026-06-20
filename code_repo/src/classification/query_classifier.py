from langchain_core.output_parsers import JsonOutputParser
from classification.query_classification_schema import QueryClassification
from prompts.query_classification_prompt import CLASSIFICATION_TEMPLATE
from langchain_core.prompts import PromptTemplate

parser = JsonOutputParser(
    pydantic_object=QueryClassification
)

prompt = PromptTemplate(
    template=CLASSIFICATION_TEMPLATE,
    input_variables=["query"],
    partial_variables={
        "format_instructions":
        parser.get_format_instructions()
    }
)

def build_classifier(llm):
    chain = (
        prompt
        | llm
        | parser
    )
    return chain