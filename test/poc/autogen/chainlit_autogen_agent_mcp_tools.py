from typing import List, cast, Dict, Any
import logging
import chainlit as cl
import yaml
import os
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to format file listings for Chainlit
def format_file_listing(content: str) -> str:
    """Format file listing results for better display in Chainlit."""
    try:
        # Try to parse as JSON if possible
        data = json.loads(content)
        if isinstance(data, list):
            # Format a list of files/directories
            formatted = "### Files and Directories\n\n"
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    icon = "📁" if item.get("isDirectory", False) else "📄"
                    size = f" ({item.get('size', 'Unknown size')})" if not item.get("isDirectory", False) else ""
                    formatted += f"{icon} {item['name']}{size}\n"
                else:
                    formatted += f"- {item}\n"
            return formatted
        elif isinstance(data, dict) and "entries" in data:
            # Format directory listing with entries
            formatted = f"### Contents of {data.get('path', 'Directory')}\n\n"
            for entry in data.get("entries", []):
                icon = "📁" if entry.get("isDirectory", False) else "📄"
                size = f" ({entry.get('size', 'Unknown size')})" if not entry.get("isDirectory", False) else ""
                formatted += f"{icon} {entry.get('name', 'Unknown')}{size}\n"
            return formatted
        else:
            # Just pretty print the JSON
            return f"```json\n{json.dumps(data, indent=2)}\n```"
    except (json.JSONDecodeError, ValueError):
        # Not JSON, return as is with minimal formatting
        if "Error:" in content:
            return f"❌ **Error occurred:**\n\n{content}"
        return content


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Greetings",
            message="Hello! What can you help me with today?",
        ),
        cl.Starter(
            label="MCP Tools",
            message="List all MCP tools?",
        ),
    ]


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    # Load model configuration and create the model client.
    try:
        with open("model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f)
        model_client = ChatCompletionClient.load_component(model_config)
        logger.info("Successfully loaded model configuration")
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        await cl.Message(content=f"Error initializing the AI model: {str(e)}").send()
        return

    # Create server params for the MCP filesystem service
    server_params = StdioServerParams(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/home/hosseina/Pictures"]
    )

    # Initialize MCP tools
    mcp_tools = []
    try:
        logger.info("Attempting to load MCP filesystem tools...")
        # Get all available tools from the MCP server
        mcp_tools = await mcp_server_tools(server_params)
        tool_names = [tool.name for tool in mcp_tools]
        logger.info(f"Successfully loaded {len(mcp_tools)} MCP tools: {', '.join(tool_names)}")
        cl.user_session.set("mcp_tools_available", True)  # type: ignore
    except Exception as e:
        error_message = f"Error loading MCP filesystem tools: {e}"
        logger.error(error_message)
        cl.user_session.set("mcp_tools_available", False)  # type: ignore
        cl.user_session.set("mcp_error", error_message)  # type: ignore

    # Build a system message with tool descriptions
    system_message = "You are a helpful assistant that can browse and interact with files in the Pictures directory.\n\n"
    system_message += "Available tools and their capabilities:\n"
    
    # Add each tool and its description as a bullet point
    if mcp_tools:
        for tool in mcp_tools:
            desc = tool.description if hasattr(tool, 'description') and tool.description else f"Use {tool.name} to work with files"
            system_message += f"• {tool.name}: {desc}\n"
    
    # Create the assistant agent with MCP tools
    assistant = AssistantAgent(
        name="assistant",
        tools=mcp_tools,
        model_client=model_client,
        system_message=system_message,
        model_client_stream=True,  # Enable model client streaming.
        reflect_on_tool_use=True,  # Reflect on tool use.
    )

    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("agent", assistant)  # type: ignore
    
    # Inform the user about tool availability
    if mcp_tools:
        await cl.Message(content=f"Filesystem tools are available: {', '.join([tool.name for tool in mcp_tools])}").send()
    else:
        await cl.Message(content="Filesystem tools are not available. Standard chat functionality will work, but file browsing capabilities will be limited.").send()


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    # Get the assistant agent from the user session.
    agent = cast(AssistantAgent, cl.user_session.get("agent"))  # type: ignore
    
    # Check if we're talking about MCP features but tools aren't available
    mcp_tools_available = cl.user_session.get("mcp_tools_available", False)  # type: ignore
    if not mcp_tools_available and any(keyword in message.content.lower() for keyword in 
                                      ["file", "directory", "folder", "browse", "picture", "photo", "list"]):
        # Get detailed error message if available
        mcp_error = cl.user_session.get("mcp_error", "unknown reason")  # type: ignore
        
        # Inform the user that filesystem tools are not available
        response = cl.Message(content=f"I'm sorry, but filesystem tools are not currently available due to: {mcp_error}\n\n" +
                             "Please check that you have the proper MCP filesystem package installed " +
                             "using 'npm install -g @modelcontextprotocol/server-filesystem'.")
        await response.send()
        return
    
    # Initialize variables to track the state of our interaction
    streaming_response = cl.Message(content="")  # For streaming normal AI responses
    tool_response_content = None  # To track tool response content
    tool_message_sent = False  # Flag to track if we've sent a tool response
    
    # Process the messages from the agent
    try:
        async for msg in agent.on_messages_stream(
            messages=[TextMessage(content=message.content, source="user")],
            cancellation_token=CancellationToken(),
        ):
            # Handle streaming chunks from the model
            if isinstance(msg, ModelClientStreamingChunkEvent):
                # Only stream if we haven't sent a tool message yet
                if not tool_message_sent:
                    await streaming_response.stream_token(msg.content)
            
            # Handle tool messages (responses from tool calls)
            elif hasattr(msg, 'content') and hasattr(msg, 'role') and msg.role == 'tool':
                # Log the tool response
                logger.info(f"Tool message received: {msg.content}")
                tool_response_content = msg.content
                
                # Check if this is a file listing related request
                is_listing_operation = any(keyword in message.content.lower() for keyword in 
                                      ["list", "show", "display"]) and any(keyword in message.content.lower() for keyword in 
                                      ["file", "directory", "folder", "content", "pictures"])
                
                # If we're listing files, format the response appropriately
                if is_listing_operation:
                    # Format the raw tool response into a nicer display
                    formatted_content = format_file_listing(msg.content)
                    
                    # Send a new, nicely formatted message with the file listing
                    file_list_message = cl.Message(content=formatted_content)
                    await file_list_message.send()
                    tool_message_sent = True
                    
                    # Clear the streaming response since we're sending a formatted message instead
                    streaming_response = cl.Message(content="")
            
            # Handle the final response from the agent
            elif isinstance(msg, Response):
                # If we've already sent a tool response, don't send another message
                if tool_message_sent:
                    continue
                
                # If we have a tool response but haven't sent a formatted message yet
                if tool_response_content and not tool_message_sent:
                    # Check if it's an error
                    if "Error:" in str(tool_response_content):
                        error_message = cl.Message(content=f"I'm sorry, I encountered an error while accessing the files:\n\n```\n{tool_response_content}\n```\n\nPlease make sure the directory exists and I have permission to access it.")
                        await error_message.send()
                    else:
                        # If it's not a listing operation but still a tool response,
                        # send the streaming response (AI's interpretation of the tool result)
                        if streaming_response.content:
                            await streaming_response.send()
                        else:
                            # If no streaming content, create a message with the raw tool response
                            raw_response = cl.Message(content=f"```\n{tool_response_content}\n```")
                            await raw_response.send()
                else:
                    # No tool was used, just send the normal AI response
                    await streaming_response.send()
                
    except Exception as e:
        # Handle any exceptions during processing
        logger.error(f"Error in chat processing: {e}")
        error_msg = cl.Message(content=f"I'm sorry, but I encountered an error while processing your request: {str(e)}")
        await error_msg.send()
    