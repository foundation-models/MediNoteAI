import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
import yaml
import logging

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main() -> None:
    # Create server params for the MCP filesystem service - updated to use actual workspace path
    workspace_path = "/home/hosseina/workspace/MediNoteAI/test/poc/autogen"
    server_params = StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", workspace_path]
    )

    # Load model configuration
    with open("model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    # Initialize ChatOpenAI model
    model = ChatOpenAI(
        model=model_config.get("model", "gpt-3.5-turbo"),
        api_key=model_config.get("api_key"),
        temperature=model_config.get("temperature", 0),
    )
    logger.info("Successfully loaded model configuration")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            
            # Execute the agent with a task
            agent_response = await agent.ainvoke({
                "messages": "List all files in the current directory."
            })
            
            # Loop over the responses and print them
            for response in agent_response["messages"]:
                prefix = ""
                
                if isinstance(response, HumanMessage):
                    prefix = "**User**"
                elif isinstance(response, ToolMessage):
                    prefix = "**Tool**"
                elif isinstance(response, AIMessage):
                    prefix = "**AI**"
                    
                # Check if response.content is a list or just a string
                if isinstance(response.content, list):
                    for content in response.content:
                        print(f'{prefix}: {content.get("text", "")}')
                    continue
                print(f"{prefix}: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
