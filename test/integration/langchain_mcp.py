import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage


# Initialize a ChatOpenAI model
model = ChatOllama(
    base_url="https://ollama.dc.dev1.intapp.com/",
    model="qwen2.5:latest",
    temperature=0,
)

server_params = StdioServerParameters(
    command="npx",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["-y","mcp-media-processor@latest"],
)

# Wrap the async code inside an async function
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            # agent_response = await agent.ainvoke({"messages": "compress handwirtten2.png in /home/agent/workspace folder and save it in /home/agent/workspace/compressed_image.png"})
            agent_response = await agent.ainvoke({"messages": "compress-image inputPath='/home/agent/workspac/handwirtten2.png' outputPath='/home/agent/workspacecompressed_image.png'"})
            
            # print(agent_response)
            # Loop over the responses and print them
            for response in agent_response["messages"]:
                user = ""
                
                if isinstance(response, HumanMessage):
                    user = "**User**"
                elif isinstance(response, ToolMessage):
                    user = "**Tool**"
                elif isinstance(response, AIMessage):
                    user = "**AI**"
                    
                # Check if response.content is a list or just a string
                if isinstance(response.content, list):
                    for content in response.content:
                        print(f'{user}: {content.get("text", "")}')
                    continue
                print(f"{user}: {response.content}")

# Run the async function in the event loop
asyncio.run(main())