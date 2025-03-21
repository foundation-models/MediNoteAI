"""
Example demonstrating how to use an agent with MCP servers.

This example shows how to:
1. Load MCP servers from the config file automatically
2. Create an agent that specifies which MCP servers to use
3. Run the agent to dynamically load and use tools from the specified MCP servers

To use this example:
1. Create an mcp_agent.config.yaml file in this directory or a parent directory
2. Configure your MCP servers in that file
3. Run this example
"""

import asyncio
from typing import TYPE_CHECKING

from openai.types.responses import ResponseTextDeltaEvent

if TYPE_CHECKING:
    pass

from agents import Runner

from agents_mcp import Agent, RunnerContext


async def main():


    # Create a context object containing MCP settings
    context = RunnerContext()

    # Create an agent with specific MCP servers you want to use
    # These must be defined in your mcp_agent.config.yaml file
    agent: Agent = Agent(
        name="MCP Assistant",
        instructions="""You are a helpful assistant with access to both local tools
            and tools from MCP servers. Use these tools to help the user.""",
        mcp_servers=[
            "filesystem",
        ],  # Specify which MCP servers to use (must be defined in your MCP config)
        mcp_server_registry=None,  # Specify a custom MCP server registry per-agent if needed
    )

    # Run the agent - tools from the specified MCP servers will be automatically loaded
    result = Runner.run_streamed(
        agent,
        input="find all files",
        context=context,
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
    