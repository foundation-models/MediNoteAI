import os
import asyncio
from dotenv import load_dotenv
from agents_mcp import Agent, RunnerContext
from agents import Runner

async def main():
    # Load environment variables from .env
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["CLIENT_ID", "CLIENT_SECRET", "GDRIVE_CREDS_DIR"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please make sure these are set in your .env file")
        return
    
    # Configure the MCP server for Google Drive
    mcp_servers = {
        "gdrive": {
            "command": "npx",
            "args": ["@isaacphi/mcp-gdrive"],
            "env": {
                "CLIENT_ID": os.getenv("CLIENT_ID"),
                "CLIENT_SECRET": os.getenv("CLIENT_SECRET"),
                "GDRIVE_CREDS_DIR": os.getenv("GDRIVE_CREDS_DIR")
            }
        }
    }

    # Create an agent with access to the Google Drive MCP server
    agent = Agent(
        name="GDriveTest",
        instructions="You are a helpful assistant that can access Google Drive.",
        mcp_servers=mcp_servers
    )

    # Create a simple search query that will trigger authentication
    query = "List all files in my Google Drive"
    
    print("Making first request to Google Drive...")
    print("You should see a browser window open for authentication.")
    
    # Run the query
    try:
        result = await Runner.run(
            agent,
            input=query,
            context=RunnerContext()
        )
        print("\nResult:", result)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 