import chainlit as cl
from dotenv import load_dotenv

from agents import Runner, enable_verbose_stdout_logging
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent
)
from agents_mcp import Agent, RunnerContext
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables and enable logging
load_dotenv()
enable_verbose_stdout_logging()

@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat system with OpenAI agents and filesystem tools."""
    agent = Agent(
        name="Filesystem Assistant",
        instructions="""You are a file system operations assistant. Your role is to:
        1. Execute file system operations using available tools (list_directory, read_file)
        2. Present results in a clean, structured format
        3. Only show the actual results, without explaining the process
        4. Use markdown tables for structured data
        
        When searching for files, use filesystem-list_directory to list files in a directory.
        When reading files, use filesystem-read_file to get the contents of specific files.
        
        Keep responses focused on the results only, without explanations or step-by-step process.
        Call 'APPROVE' only after showing the complete results.
        """,
        mcp_servers=["filesystem"]
    )
    
    cl.user_session.set("agent", agent)
    print("Chat system initialized with OpenAI MCP agent.")

@cl.on_message
async def chat(message: cl.Message) -> None:
    """
    Handle the conversation by processing messages and streaming responses.

    Args:
        message (cl.Message): The message received from the user.
    """    
    msg = cl.Message(content="", author="Filesystem Assistant")
    await msg.send()

    agent = cl.user_session.get("agent")
    result = Runner.run_streamed(
        agent,
        input=message.content,
        context=RunnerContext(),
    )
    
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                await msg.stream_token(event.data.delta)
            
        elif isinstance(event, RunItemStreamEvent):
            # Handle tool events if needed in the future
            pass
        
        elif isinstance(event, AgentUpdatedStreamEvent):
            # Handle agent updates if needed in the future
            pass

    if result.final_output and not result.final_output.isspace():
        await msg.update(content=result.final_output)