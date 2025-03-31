import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_ollama import ChatOllama
import os

async def main() -> None:
    # Create server params for the remote MCP service
    server_params = StdioServerParameters(
        command="npx", args=["-y", "mcp-media-processor@latest"]
    )

    # Initialize the model using Ollama
    model = ChatOllama(
        base_url="https://ollama.dc.dev1.intapp.com/",
        model="qwen2.5:latest",
        temperature=0,
    )

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
                "messages": "compress handwirtten2.png in /home/agent/workspace folder and save it in /home/agent/workspace/compressed_image.png"
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
