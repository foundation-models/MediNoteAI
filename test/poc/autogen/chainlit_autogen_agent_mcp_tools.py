from typing import List, cast
import logging
import chainlit as cl
import yaml
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Greetings",
            message="Hello! What can you help me with today?",
        ),
        cl.Starter(
            label="Browse Files",
            message="Can you show me files in my Pictures directory?",
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

    # Create the assistant agent with MCP tools
    assistant = AssistantAgent(
        name="assistant",
        tools=mcp_tools,
        model_client=model_client,
        system_message="You are a helpful assistant that can browse and interact with files in the Pictures directory",
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
    
    # Construct the response message.
    response = cl.Message(content="")
    async for msg in agent.on_messages_stream(
        messages=[TextMessage(content=message.content, source="user")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(msg, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            await response.stream_token(msg.content)
        elif isinstance(msg, Response):
            # Done streaming the model client response. Send the message.
            await response.send()
    