from pydantic import BaseModel
from typing import Optional
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_agentchat.agents import AssistantAgent

import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from typing import Type, Any, Dict, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

class WeatherRequest(BaseModel):
    city: str
    time: str


from pydantic import BaseModel
from typing import Type, Any, Dict, Optional
import re
import os

def get_user_prompt(model_type: Type[BaseModel]) -> str:
    """
    Generate a user-friendly prompt based on the Pydantic model.
    
    Args:
        model_type: The Pydantic model class to extract parameters for
        
    Returns:
        A prompt string describing what information is needed
    """
    # Get the model name for the prompt
    model_name = model_type.__name__
    friendly_model_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', model_name).lower()
    
    # Get field information from the model
    model_fields = model_type.__annotations__
    
    # Create a list of required fields
    field_names = [re.sub(r'_', ' ', field) for field in model_fields.keys()]
    
    if len(field_names) == 1:
        fields_text = field_names[0]
    elif len(field_names) == 2:
        fields_text = f"{field_names[0]} and {field_names[1]}"
    else:
        fields_text = ", ".join(field_names[:-1]) + f", and {field_names[-1]}"
    
    # Generate prompt
    prompt = f"For {friendly_model_name}, please provide {fields_text} (or type 'exit' to quit): "
    
    return prompt

def extract_params(user_input: str, model_type: Type[BaseModel]) -> Tuple[BaseModel, Dict[str, str]]:
    """
    Generic parameter extraction function that works with any Pydantic model.
    
    Args:
        user_input: The text input from the user
        model_type: The Pydantic model class to extract parameters for
        
    Returns:
        A tuple containing:
        - An instance of the model_type with extracted values
        - A dictionary of extraction messages for each field
    """
    # Create a dictionary to store extracted values
    extracted_values = {}
    
    # Create a dictionary to store extraction messages
    extraction_messages = {}
    
    # Get the model name for prefixing messages
    model_name = model_type.__name__
    friendly_model_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', model_name).lower()
    
    # Get field information from the model
    model_fields = model_type.__annotations__
    
    # Define common patterns for different field types
    extraction_patterns = {
        'city': [
            r'(?:in|for|at|city:?)\s+([A-Za-z\s]+?)(?:\s|$|,|\.|!|\?)',
            r'(?:city|location|place)(?:\s+is|:)\s+([A-Za-z\s]+?)(?:\s|$|,|\.|!|\?)'
        ],
        'time': [
            r'(?:at|time:?|during|for)\s+([A-Za-z0-9\s:]+?)(?:\s|$|,|\.|!|\?)',
            r'(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))',
            r'(?:today|tomorrow|yesterday|morning|afternoon|evening|night)',
            r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
        ],
        'date': [
            r'(?:on|date:?)\s+([A-Za-z0-9\s,]+?)(?:\s|$|,|\.|!|\?)',
            r'(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?'
        ],
        'temperature': [
            r'(\d+(?:\.\d+)?)\s*(?:degrees|°|celsius|fahrenheit|c|f)',
            r'temperature(?:\s+is|:)\s+(\d+(?:\.\d+)?)'
        ],
        'name': [
            r'(?:name(?:\s+is|:))\s+([A-Za-z\s]+?)(?:\s|$|,|\.|!|\?)',
            r'(?:i\s+am|my\s+name\s+is)\s+([A-Za-z\s]+?)(?:\s|$|,|\.|!|\?)'
        ],
        'email': [
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        ],
        'phone': [
            r'(\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9})'
        ]
    }
    
    # For each field in the model, try to extract a value
    for field_name, field_type in model_fields.items():
        field_type_str = str(field_type).lower()
        friendly_field_name = field_name.replace('_', ' ')
        
        # Create prefix message for this field
        prefix_message = f"In extracting the {friendly_model_name}, looking for {friendly_field_name}:"
        
        # Determine which patterns to use based on field name and type
        patterns = []
        
        # Get specific patterns based on field name
        if field_name.lower() in extraction_patterns:
            patterns.extend(extraction_patterns[field_name.lower()])
        
        # Add type-based patterns
        if 'str' in field_type_str:
            # For string fields not covered by specific patterns
            if not patterns:
                # Generic pattern for string fields - look for field name followed by value
                patterns.append(rf'(?:{field_name}(?:\s+is|:))\s+([A-Za-z0-9\s]+?)(?:\s|$|,|\.|!|\?)')
        elif 'int' in field_type_str or 'float' in field_type_str:
            # For numeric fields
            if not patterns:
                patterns.append(rf'(?:{field_name}(?:\s+is|:))\s+(\d+(?:\.\d+)?)(?:\s|$|,|\.|!|\?)')
        elif 'bool' in field_type_str:
            # For boolean fields
            if not patterns:
                patterns.append(rf'(?:{field_name}(?:\s+is|:))\s+(true|false|yes|no)(?:\s|$|,|\.|!|\?)')
        
        # Try each pattern
        for pattern in patterns:
            matches = re.search(pattern, user_input, re.IGNORECASE)
            if matches:
                extracted_value = matches.group(1).strip()
                # Convert to appropriate type if needed
                if 'int' in field_type_str:
                    try:
                        extracted_value = int(extracted_value)
                    except ValueError:
                        continue
                elif 'float' in field_type_str:
                    try:
                        extracted_value = float(extracted_value)
                    except ValueError:
                        continue
                elif 'bool' in field_type_str:
                    extracted_value = extracted_value.lower() in ('true', 'yes')
                
                extracted_values[field_name] = extracted_value
                extraction_messages[field_name] = f"{prefix_message} Found '{extracted_value}'"
                break
        
        # If no value found, add message about missing field
        if field_name not in extracted_values:
            extraction_messages[field_name] = f"{prefix_message} No value found"
    
    # Create model instance with extracted values
    try:
        return model_type(**extracted_values), extraction_messages
    except Exception as e:
        # If validation fails, return model with default values
        error_msg = f"Validation error: {e}"
        for field in model_fields:
            if field not in extraction_messages:
                extraction_messages[field] = f"{prefix_message} {error_msg}"
        return model_type(), extraction_messages
        
async def get_weather(city: str, time: str) -> str:
    """Get weather information for a specific city and time."""
    # Actual implementation would call a weather API
    return f"Weather for {city} at {time}: Sunny, 72°F"

async def clarify_input(assistant, message_history):
    """Helper function to clarify missing parameters one at a time."""
    request = WeatherRequest(city="", time="")
    
    # Check for city
    if not request.city:
        clarification = await assistant.on_messages(
            message_history + [TextMessage(content="Please provide the city name.", source="assistant")],
            CancellationToken()
        )
        # Extract city from user response
        # You'd need logic here to parse the user's response
        request.city = clarification.chat_message.content
        
    # Check for time
    if not request.time:
        clarification = await assistant.on_messages(
            message_history + [TextMessage(content="Please provide the time.", source="assistant")],
            CancellationToken()
        )
        # Extract time from user response
        request.time = clarification.chat_message.content
        
    return request

async def main() -> None:
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
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. You can call tools to help user.",
        model_client=model_client,
        tools=[get_weather],
        reflect_on_tool_use=True,
    )

    # Define the model we're working with
    request_model = WeatherRequest
    
    message_history = []
    
    # Generate initial context-aware prompt
    user_prompt = get_user_prompt(request_model)

    
    message_history = []
    
    while True:
        user_input = input(f"{user_prompt}")
        if user_input.lower() == "exit":
            break

            
        message = TextMessage(content=user_input, source="user")
        message_history.append(message)
        
        # Try to extract parameters from initial message
        try:
            # Extract parameters and get extraction messages
            weather_params, extraction_messages = extract_params(user_input, request_model)
            
            # Print extraction results for transparency
            for message in extraction_messages.values():
                print(f"System: {message}")
                
            # Check if we have all required parameters
            if not (weather_params.city and weather_params.time):
                # If missing parameters, go into clarification mode
                if not weather_params.city:
                    print("System: I need to know which city you're interested in.")
                    city_input = input("Please provide the city: ")
                    weather_params.city = city_input
                
                if not weather_params.time:
                    print("System: I need to know what time you want the weather for.")
                    time_input = input("Please provide the time: ")
                    weather_params.time = time_input
            
            # Once we have all parameters, call the weather function
            weather_result = await get_weather(weather_params.city, weather_params.time)
            response = await assistant.on_messages(
                message_history + [TextMessage(content=weather_result, source="system")],
                CancellationToken()
            )
        except Exception as e:
            response = await assistant.on_messages(
                message_history + [TextMessage(content=f"Error: {str(e)}", source="system")],
                CancellationToken()
            )
                        
        message_history.append(response.chat_message)
        print("Assistant:", response.chat_message.content)

asyncio.run(main())