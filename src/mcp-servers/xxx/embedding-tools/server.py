"""MCP Embedding Server implementation."""

from pathlib import Path
import aiohttp
import pandas as pd
import yaml
import os
from typing import List, Any, Optional
import asyncio
import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Initialize server
server = Server("embedding-tools")

# Load configuration
def _load_config():
    """Load configuration from config.yaml file with optional environment variable override."""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        # Load base configuration from yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Get values from config with environment variable override
        return {
            "api_url": os.getenv("EMBEDDING_API_URL", config["embedding-tools"]["api_url"]),
            "model": os.getenv("EMBEDDING_MODEL", config["embedding-tools"]["model"])
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load config.yaml: {str(e)}")

# Load config at module level
config = _load_config()

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available embedding tools."""
    return [
        types.Tool(
            name="process-parquet",
            description="Process a Parquet file and add embeddings for the specified text column",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the input Parquet file"
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Name of the column containing text to embed"
                    }
                },
                "required": ["file_path", "text_column"]
            }
        )
    ]

async def _get_embeddings_batch(texts: List[str]) -> List:
    """Get embeddings for a batch of texts.
    
    Args:
        texts: List of texts to get embeddings for.
        
    Returns:
        List of embeddings.
    """
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            payload = {
                "model": config["model"],
                "input": text
            }
            try:
                async with session.post(config["api_url"], json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        embeddings.append(result.get("data", [{}])[0].get("embedding", []))
                    else:
                        # If there's an error, append an empty list as embedding
                        embeddings.append([])
            except Exception as e:
                print(f"Error getting embedding for text: {str(e)}")
                embeddings.append([])
    return embeddings

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle embedding tools execution requests."""
    if not arguments:
        raise ValueError("Missing arguments")

    if name == "process-parquet":
        file_path = arguments.get("file_path")
        text_column = arguments.get("text_column")
        
        if not file_path or not text_column:
            raise ValueError("Missing required arguments")

        try:
            # Read the Parquet file
            print(f"Processing file: {file_path}")
            df = pd.read_parquet(file_path)
            
            if text_column not in df.columns:
                return [types.TextContent(
                    type="text",
                    text=f"Error: Column {text_column} not found in the Parquet file"
                )]
            
            # Get embeddings for each text in the column
            embeddings = await _get_embeddings_batch(df[text_column].tolist())
            
            # Add embeddings as a new column
            embedding_column = f"{text_column}_embedding"
            df[embedding_column] = embeddings
            
            # Save to a new Parquet file
            output_path = f"{Path(file_path).stem}_with_embeddings.parquet"
            df.to_parquet(output_path)
            
            return [types.TextContent(
                type="text",
                text=f"Successfully processed file. Output saved to: {output_path}"
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error processing file: {str(e)}"
            )]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the server using stdin/stdout streams."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="embedding-tools",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 