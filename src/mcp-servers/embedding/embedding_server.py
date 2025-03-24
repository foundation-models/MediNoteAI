from pathlib import Path
import aiohttp
import pandas as pd
import yaml
import os
from mcp.server.fastmcp import FastMCP

class EmbeddingServer:
    def __init__(self):
        self.mcp = FastMCP("Embedding Server")
        self._load_config()
        self.setup_handlers()

    def _load_config(self):
        """Load configuration from config.yaml file with optional environment variable override"""
        config_path = Path(__file__).parent / "config.yaml"
        try:
            # Load base configuration from yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get values from config with environment variable override
            self.api_url = os.getenv('EMBEDDING_API_URL', config['embedding']['api_url'])
            self.model = os.getenv('EMBEDDING_MODEL', config['embedding']['model'])
            
        except Exception as e:
            raise RuntimeError(f"Failed to load config.yaml: {str(e)}")

    def setup_handlers(self):
        @self.mcp.tool()
        async def process_parquet(file_path: str, text_column: str) -> str:
            """Process a Parquet file and add embeddings for the specified text column"""
            try:
                # Read the Parquet file
                print(f"Processing file: {file_path}")
                df = pd.read_parquet(file_path)
                
                if text_column not in df.columns:
                    return f"Error: Column {text_column} not found in the Parquet file"
                
                # Get embeddings for each text in the column
                embeddings = await self._get_embeddings_batch(df[text_column].tolist())
                
                # Add embeddings as a new column
                embedding_column = f"{text_column}_embedding"
                df[embedding_column] = embeddings
                
                # Save to a new Parquet file
                output_path = f"{Path(file_path).stem}_with_embeddings.parquet"
                df.to_parquet(output_path)
                
                return f"Successfully processed file. Output saved to: {output_path}"
            except Exception as e:
                return f"Error processing file: {str(e)}"

    async def _get_embeddings_batch(self, texts: list[str]) -> list:
        """Get embeddings for a batch of texts"""
        embeddings = []
        async with aiohttp.ClientSession() as session:
            for text in texts:
                payload = {
                    "model": self.model,
                    "input": text
                }
                try:
                    async with session.post(self.api_url, json=payload) as response:
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

    def run(self):
        """Run the MCP server"""
        self.mcp.run()

if __name__ == "__main__":
    server = EmbeddingServer()
    server.run() 