import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os

async def main() -> None:
    # Create server params for the remote MCP service
    server_params = StdioServerParams(
        command="npx", args=["-y", "mcp-media-processor@latest"]
    )

    # Create an agent that can use the translation tool
    model_client = AzureOpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_type="azure",
        #   credential=AzureKeyCredential("api_key"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        seed=42,
        temperature=0,
    )


    # Get all available tools from the server
    tools = await mcp_server_tools(server_params)

    # Create an agent that can use all the tools
    agent = AssistantAgent(
        name="compress_image",
        model_client=model_client,
        tools=tools,  # type: ignore
    )
    try:
        # Let the agent translate some text
        result = await agent.run(task="compress handwirtten2.png in /home/agent/workspace folder and save it in /home/agent/workspace/compressed_image.png", cancellation_token=CancellationToken())
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
