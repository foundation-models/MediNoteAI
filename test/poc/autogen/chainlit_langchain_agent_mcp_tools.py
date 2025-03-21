from typing import List, Dict, Any, Optional, Awaitable, AsyncIterator, Callable
import logging
import chainlit as cl
import yaml
import os
import json
import asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler


# Custom AsyncIteratorCallbackHandler implementation
class AsyncIteratorCallbackHandler(BaseCallbackHandler):
    """Callback handler that yields LLM tokens as they become available."""

    def __init__(self) -> None:
        super().__init__()
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        self.tokens = []

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run when a new token is available."""
        await self.queue.put(token)
        self.tokens.append(token)

    async def on_llm_end(self, *args, **kwargs) -> None:
        """Run when LLM ends."""
        self.done.set()

    async def on_llm_error(self, *args, **kwargs) -> None:
        """Run when LLM errors."""
        self.done.set()

    async def aiter(self) -> AsyncIterator[str]:
        """Async iterator for tokens."""
        try:
            while not self.done.is_set() or not self.queue.empty():
                token = await self.queue.get()
                yield token
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            pass

    def __del__(self) -> None:
        """Clean up resources."""
        self.done.set()


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
            model = ChatOpenAI(
                model=model_config.get("model", "gpt-3.5-turbo"),
                api_key=model_config.get("api_key"),
                temperature=model_config.get("temperature", 0),
                streaming=True
            )
        logger.info("Successfully loaded model configuration")
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        await cl.Message(content=f"Error initializing the AI model: {str(e)}").send()
        return

    # Create server params for the MCP filesystem service
    server_params = StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/home/hosseina/Pictures"]
    )

    # Initialize MCP tools
    mcp_tools = []
    try:
        logger.info("Attempting to load MCP filesystem tools...")
        
        # Get tools using the stdio client and MCP session
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # Get tools
                mcp_tools = await load_mcp_tools(session)
                tool_names = [tool.name for tool in mcp_tools]
                logger.info(f"Successfully loaded {len(mcp_tools)} MCP tools: {', '.join(tool_names)}")
                
                # Create and run the agent
                agent = create_react_agent(model, mcp_tools)
                
                # Store the agent in the session
                cl.user_session.set("agent", agent)  # type: ignore
                cl.user_session.set("mcp_tools_available", True)  # type: ignore
                
                # Send welcome message with available tools
                welcome_message = (
                    f"Filesystem tools are available: {', '.join(tool_names)}\n\n"
                    f"You can now ask me to browse, list, or manipulate files in the directory: /home/hosseina/Pictures"
                )
                await cl.Message(content=welcome_message).send()
                
    except Exception as e:
        error_message = f"Error loading MCP filesystem tools: {e}"
        logger.error(error_message)
        cl.user_session.set("mcp_tools_available", False)  # type: ignore
        cl.user_session.set("mcp_error", error_message)  # type: ignore
        await cl.Message(content="Filesystem tools are not available. Standard chat functionality will work, but file browsing capabilities will be limited.").send()


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    # Get the agent from the user session
    agent = cl.user_session.get("agent")  # type: ignore
    
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
    
    # Create a streaming message
    streaming_response = cl.Message(content="")
    await streaming_response.send()
    
    # Create callback handler for streaming
    callback_handler = AsyncIteratorCallbackHandler()
    
    # Configure the LangChain agent to use streaming
    config = RunnableConfig(
        callbacks=[callback_handler]
    )
    
    # Start the agent execution in a background task
    try:
        logger.info(f"Starting agent execution for message: {message.content[:50]}...")
        task = asyncio.create_task(
            agent.ainvoke(
                {"messages": message.content},
                config=config
            )
        )
    except Exception as e:
        logger.error(f"Error creating agent task: {str(e)}")
        await streaming_response.update(content=f"I'm sorry, there was an error initializing the agent: {str(e)}")
        return
    
    # Initialize variables to track the state of our interaction
    tool_response_content = None
    tool_message_sent = False
    
    # Process the streaming output
    try:
        # Stream the thinking process
        async for chunk in callback_handler.aiter():
            if chunk and not tool_message_sent:
                await streaming_response.stream_token(chunk)
        
        # Get the final result
        result = await task
        
        # Process the final messages
        for response in result["messages"]:
            # Handle tool messages (responses from tool calls)
            if isinstance(response, ToolMessage):
                # Log the tool response
                logger.info(f"Tool message received: {response.content}")
                tool_response_content = response.content
                
                # Always try to parse the content for better display
                try:
                    # Check if this is a file listing related request
                    is_listing_operation = any(keyword in message.content.lower() for keyword in 
                                ["list", "show", "display"]) and any(keyword in message.content.lower() for keyword in 
                                ["file", "directory", "folder", "content", "pictures"])
                    
                    # If we're listing files, format the response appropriately
                    if is_listing_operation:
                        # Format the raw tool response into a nicer display
                        formatted_content = format_file_listing(response.content)
                        
                        # Send a new, nicely formatted message with the file listing
                        file_list_message = cl.Message(content=formatted_content)
                        await file_list_message.send()
                        tool_message_sent = True
                    else:
                        # For other tool responses, still format them nicely
                        formatted_content = format_file_listing(response.content)
                        tool_message = cl.Message(content=formatted_content)
                        await tool_message.send()
                        tool_message_sent = True
                except Exception as format_error:
                    logger.warning(f"Error formatting tool response: {format_error}")
                    # If formatting fails, still show the raw response
                    raw_response = cl.Message(content=f"```\n{tool_response_content}\n```")
                    await raw_response.send()
                    tool_message_sent = True
            
            # Handle the final AI response
            elif isinstance(response, AIMessage) and not tool_message_sent:
                # Update the streaming response with the final content
                streaming_response.content = response.content
                await streaming_response.update()
                
        # If we have a tool response but haven't sent a formatted message yet
        if tool_response_content and not tool_message_sent:
            # Check if it's an error
            if "Error:" in str(tool_response_content):
                error_message = cl.Message(content=f"I'm sorry, I encountered an error while accessing the files:\n\n```\n{tool_response_content}\n```\n\nPlease make sure the directory exists and I have permission to access it.")
                await error_message.send()
            else:
                # If no formatted message was sent, send the raw tool response
                raw_response = cl.Message(content=f"```\n{tool_response_content}\n```")
                await raw_response.send()
                
    except Exception as e:
        # Handle any exceptions during processing
        logger.error(f"Error in chat processing: {e}")
        
        # Check for recursion limit errors
        error_message = str(e)
        if "recursion_limit" in error_message.lower():
            error_msg = cl.Message(content=(
                "I encountered a recursion limit error while trying to process your request. "
                "This usually happens when I'm unable to complete a task with the given tools. "
                "Try breaking your request into smaller, more specific steps or providing more details.\n\n"
                f"Technical details: {error_message}"
            ))
        else:
            error_msg = cl.Message(content=f"I'm sorry, but I encountered an error while processing your request: {error_message}")
        
        await error_msg.send()
    