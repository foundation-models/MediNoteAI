import asyncio
import json
from typing import Tuple, Dict, Type
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import os
from autogen_agents_deep_research import InvestmentThesis

load_dotenv()

# Initialize the Azure OpenAI Chat model
model = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    temperature=0
)

async def call_extraction_model_all(user_input: str, fields: list) -> dict:
    """
    Uses LangChain with Azure OpenAI to extract fields from user input.
    
    Args:
        user_input: The text input from the user.
        fields: A list of field names to extract.
    
    Returns:
        A dictionary mapping each field to its extracted value.
    """
    # Create the chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that extracts information from text."),
        HumanMessage(content=f"""Extract the following fields from the text if available and output a valid JSON object with these keys:
{fields}

Rules:
- Only output the JSON object and nothing else.
- If a field is not present in the text, set its value to an empty string.

Text: {user_input}""")
    ])
    
    # Create the output parser
    output_parser = JsonOutputParser()
    
    # Create the chain
    chain = prompt | model | output_parser
    
    try:
        # Execute the chain
        result = await chain.ainvoke({})
        return result
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {}

async def extract_params(user_input: str, model_type: Type[BaseModel]) -> Tuple[BaseModel, Dict[str, str]]:
    """
    Generic parameter extraction function that works with any Pydantic model.
    Uses LangChain to extract all field values in a single call.
    
    Args:
        user_input: The text input from the user.
        model_type: The Pydantic model class to extract parameters for.
    
    Returns:
        A tuple containing:
          - An instance of the model_type with the extracted values.
          - A dictionary of extraction messages for each field.
    """
    extracted_values = {}
    extraction_messages = {}

    model_fields = model_type.__annotations__
    field_names = list(model_fields.keys())
    friendly_model_name = model_type.__name__.replace("_", " ").lower()

    # Call the LangChain model to extract all fields
    model_response = await call_extraction_model_all(user_input, field_names)
    
    for field_name in field_names:
        friendly_field_name = field_name.replace("_", " ")
        prefix_message = f"In extracting the {friendly_model_name}, looking for {friendly_field_name}:"
        
        extracted_value = model_response.get(field_name, "").strip()
        if extracted_value:
            extracted_values[field_name] = extracted_value
            extraction_messages[field_name] = f"{prefix_message} Found: {extracted_value}"
        else:
            extraction_messages[field_name] = f"{prefix_message} Not found."
    
    try:
        model_instance = model_type(**extracted_values)
        return model_instance, extraction_messages
    except ValidationError as e:
        error_messages = {}
        for error in e.errors():
            field = ".".join(error["loc"])
            msg = error["msg"]
            error_messages[field] = f"{field}: {msg}"
        return None, error_messages
    except ValueError as e:
        return None, {"error": str(e)}

# Example usage with a Pydantic model:
if __name__ == "__main__":
    user_text = "Looking for companies in Boston that work in tech."
    
    async def main():
        model_instance, messages = await extract_params(user_text, InvestmentThesis)
        print("Extraction Messages:")
        for field, message in messages.items():
            print(f"{field}: {message}")
        
        if model_instance:
            print("\nExtracted Model Data:")
            print(model_instance.json())
    
    asyncio.run(main())
        

