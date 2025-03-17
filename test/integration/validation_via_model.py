import asyncio
import json
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, ValidationError
from typing import Tuple, Dict, Type
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import UserMessage
import os
from autogen_agents_deep_research import InvestmentThesis

load_dotenv()
# model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
model_client = AzureOpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_type="azure",
    #   credential=AzureKeyCredential("api_key"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
    api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    seed=42,
    temperature=0,
)

def extract_response_content(response):
    """Extract text content from Atogen response object."""
    # Handle different response formats in Autogen 0.4.8
    if hasattr(response, 'content'):
        return response.content  # Direct content attribute
    elif hasattr(response, 'message') and hasattr(response.message, 'content'):
        return response.message.content  # Message object with content
    elif hasattr(response, 'text'):
        return response.text  # Some clients might return text attribute
    elif hasattr(response, 'choices') and len(response.choices) > 0:
        # OpenAI-style response format
        if hasattr(response.choices[0], 'message'):
            return response.choices[0].message.content
        elif hasattr(response.choices[0], 'text'):
            return response.choices[0].text
    
    # If we can't extract in any standard way, convert to string
    return str(response)

async def call_extraction_model_all(user_input: str, fields: list) -> dict:
    """
    Calls the OpenAI GPT-40-mini model to extract all specified fields from the user_input in one call.
    
    Args:
        user_input: The text input from the user.
        fields: A list of field names to extract.
    
    Returns:
        A dictionary mapping each field to its extracted value. If a field is not found,
        its value will be an empty string.
    """
    # Create a JSON schema-like prompt: keys must match the field names, missing fields get empty string.
    prompt = (
        f"Extract the following fields from the text if available and output a valid JSON object with these keys:\n"
        f"{fields}\n\n"
        f"Rules:\n"
        f"- Only output the JSON object and nothing else.\n"
        f"- If a field is not present in the text, set its value to an empty string.\n\n"
        f"Text: {user_input}"
    )
            # Get LLM response
    
    try:
        # response = openai.ChatCompletion.create(
        #     model="gpt-40-mini",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
        #         {"role": "user", "content": prompt},
        #     ],
        #     temperature=0,
        # )
        # answer = response.choices[0].message.content.strip()
        # Parse the JSON response
        response = await model_client.create(
            [UserMessage(content=prompt, source="user")]
        )
        answer = extract_response_content(response)
        result = json.loads(answer)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        result = {}
    
    return result

async def extract_params(user_input: str, model_type: Type[BaseModel]) -> Tuple[BaseModel, Dict[str, str]]:
    """
    Generic parameter extraction function that works with any Pydantic model.
    It makes a single call to an OpenAI GPT-40-mini model to extract all field values.
    
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

    # Call the OpenAI model once to extract all fields
    model_response = await call_extraction_model_all(user_input, field_names)
    
    for field_name in field_names:
        friendly_field_name = field_name.replace("_", " ")
        prefix_message = f"In extracting the {friendly_model_name}, looking for {friendly_field_name}:"
        
        # Get the value from the model response; if not present, default to empty string.
# First, await the coroutine to get the actual response
        # resolved_response = await model_response
        resolved_response = model_response
        # Then call .get on the resolved response
        extracted_value = resolved_response.get(field_name, "").strip()        
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
        

