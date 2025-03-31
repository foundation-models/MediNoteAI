import os
import yaml
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

# Global variable for agent configuration
agent_config = None

def load_agent_config(agent_id: str) -> dict:
    """
    Load agent configuration from YAML file.
    
    Args:
        agent_id (str): The ID of the agent in the configuration.
        
    Returns:
        dict: Agent configuration dictionary.
    """
    global agent_config
    config_path = os.path.join(os.path.dirname(__file__), 'agent_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not config or 'agents' not in config:
        raise ValueError("Invalid configuration file: missing 'agents' section")
    
    agent_config = config['agents'].get(agent_id)
    if not agent_config:
        raise ValueError(f"Agent '{agent_id}' not found in configuration")
    
    return agent_config

@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat system with OpenAI agents and some tools."""
    global agent_config
    try:
        agent_config = load_agent_config('default_agent')
        agent = Agent(
            name=agent_config['name'],
            instructions=agent_config['instructions'],
            mcp_servers=agent_config['mcp_servers']
        )
        
        cl.user_session.set("agent", agent)
        print("Chat system initialized with OpenAI MCP agent.")
    except Exception as e:
        print(f"Error initializing chat system: {str(e)}")
        raise

@cl.on_message
async def chat(message: cl.Message) -> None:
    """
    Handle the conversation by processing messages and streaming responses.

    Args:
        message (cl.Message): The message received from the user.
    """    
    global agent_config
    msg = cl.Message(content="", author=agent_config['name'])
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