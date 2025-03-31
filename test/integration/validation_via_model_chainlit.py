import asyncio
import chainlit as cl
from typing import Type, List
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import os

# Define the deals model
class DealsInfo(BaseModel):
    deals_value: str
    location: str
    industry: str | None = None

load_dotenv()

# Initialize the OpenAI Chat model with Ollama endpoint
model = ChatOpenAI(
    model="qwen2.5:7b",
    openai_api_base="https://ollama.dc.dev1.intapp.com/v1",
    openai_api_key="ollama",
    temperature=0
)

async def call_extraction_model_all(conversation_history: List[str], fields: list) -> dict:
    """
    Uses LangChain with Ollama to extract fields from user input.
    Now includes conversation history for context.
    """
    # Combine conversation history into a single context
    combined_text = " ".join(conversation_history)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that extracts deal information from text. Consider the entire conversation history when extracting information."),
        HumanMessage(content=f"""Extract the following fields from the conversation if available and output a valid JSON object with these keys:
{fields}

Rules:
- Only output the JSON object and nothing else.
- If a field is not present in the text, set its value to an empty string.
- For deals_value, extract any monetary values or deal amounts mentioned.
- Consider all previous messages for context when extracting information.
- If new information conflicts with old information, use the most recent information.

Conversation History:
{combined_text}""")
    ])
    
    output_parser = JsonOutputParser()
    chain = prompt | model | output_parser
    
    try:
        result = await chain.ainvoke({})
        return result
    except Exception as e:
        await cl.Message(f"Error calling Ollama API: {e}").send()
        return {}

async def extract_params(conversation_history: List[str], model_type: Type[BaseModel]):
    """
    Generic parameter extraction function that works with any Pydantic model.
    Now includes conversation history.
    """
    extracted_values = {}
    extraction_messages = []

    model_fields = model_type.__annotations__
    field_names = list(model_fields.keys())
    friendly_model_name = model_type.__name__.replace("_", " ").lower()

    # Call the LangChain model to extract all fields using conversation history
    model_response = await call_extraction_model_all(conversation_history, field_names)
    
    for field_name in field_names:
        friendly_field_name = field_name.replace("_", " ")
        prefix_message = f"Looking for {friendly_field_name}:"
        
        extracted_value = model_response.get(field_name, "").strip()
        if extracted_value:
            extracted_values[field_name] = extracted_value
            extraction_messages.append(f"{prefix_message} Found: {extracted_value}")
        else:
            extraction_messages.append(f"{prefix_message} Not found.")
    
    try:
        model_instance = model_type(**extracted_values)
        return model_instance, extraction_messages
    except Exception as e:
        extraction_messages.append(f"Error: {str(e)}")
        return None, extraction_messages

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Initialize conversation history in the session
    cl.user_session.set("conversation_history", [])
    
    welcome_message = """Welcome! I can help you extract deal information from text. 

The following information is required for each deal:
- **Deals Value**: The monetary value or amount of the deal
- **Location**: The geographic location where the deal takes place
- **Industry**: (Optional) The industry sector related to the deal

You can provide this information gradually across multiple messages, and I'll maintain the context of our conversation. For example, you might say:
'There's a $50M deal in the tech sector in Boston' or provide the details separately like 'It's a deal in Boston' followed by 'The value is 50 million dollars.'"""

    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    """Process each message from the user"""
    # Get the current message
    user_text = message.content
    
    # Get and update conversation history
    conversation_history = cl.user_session.get("conversation_history", [])
    conversation_history.append(user_text)
    cl.user_session.set("conversation_history", conversation_history)
    
    # Show thinking indicator
    async with cl.Step("Extracting deal information..."):
        model_instance, messages = await extract_params(conversation_history, DealsInfo)
        
        # Send extraction process messages
        for msg in messages:
            await cl.Message(content=msg).send()
        
        if model_instance:
            # Send formatted results
            elements = []
            for field, value in model_instance.model_dump().items():
                if value:  # Only show non-empty fields
                    elements.append(f"**{field.replace('_', ' ').title()}**: {value}")
            
            await cl.Message(
                content="Here's the extracted deal information (based on our entire conversation):\n\n" + "\n".join(elements)
            ).send()
        else:
            await cl.Message(
                content="I couldn't extract valid deal information. Please provide more details about the deal."
            ).send() 