import asyncio
from agents import Agent, Runner, enable_verbose_stdout_logging

# Enable verbose logging to see what's happening
enable_verbose_stdout_logging()

# Create an agent that references MCP servers
agent = Agent(
    name="MCP Assistant",
    instructions="""You are a helpful assistant with access to filesystem tools.
When searching for files, use filesystem-list_directory to list files in a directory.
When reading files, use filesystem-read_file to get the contents of specific files.""",
    mcp_servers=["filesystem"]  # Use the filesystem server from config file
)

# The context object will be automatically populated
class AgentContext:
    pass

async def main():
    result = await Runner.run(
        agent, 
        input="List files in the current directory to find pyproject.toml", 
        context=AgentContext()
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())

