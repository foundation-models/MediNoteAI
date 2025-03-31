import asyncio
import chainlit as cl
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# Initialize the model
model = ChatOllama(
    base_url="https://ollama.dc.dev1.intapp.com/",
    model="qwen2.5:latest",
    temperature=0,
)

# Server parameters
server_params = StdioServerParameters(
    command="uvx",
    args=[
        "mcp-server-duckdb",
        "--db-path",
        "/home/agent/workspace/test.duckdb"
    ]
)


@cl.on_chat_start
async def start():

    # Initialize the client
    pass



@cl.on_message
async def main(message: cl.Message):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            # Check for termination command
            if message.content.lower() == "terminate":
                await cl.Message(content="Terminating the session. Goodbye!").send()
                return
            
            # Create a message object to show that we're processing
            msg = cl.Message(content="Processing...")
            await msg.send()

            try:
                # Invoke the agent
                agent_response = await agent.ainvoke({"messages": message.content})
                
                # Process and display the agent_responses
                content = []
                for response in agent_response["messages"]:
                    prefix = ""
                    if isinstance(response, HumanMessage):
                        prefix = "User: "
                    elif isinstance(response, ToolMessage):
                        prefix = "Tool: "
                    elif isinstance(response, AIMessage):
                        prefix = "AI: "
                        
                    if isinstance(response.content, list):
                        for item in response.content:
                            content.append(f"{prefix}{item.get('text', '')}")
                    else:
                        content.append(f"{prefix}{response.content}")
                
                # Send a new message with the content instead of updating
                await cl.Message(content="\n".join(content)).send()
                
            except Exception as e:
                await cl.Message(content=f"An error occurred: {str(e)}").send()

if __name__ == "__main__":
    cl.run() 