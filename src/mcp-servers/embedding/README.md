# MCP Embedding Server

A Model Context Protocol (MCP) server for generating text embeddings using Ollama.

## Features

- Process Parquet files and add embeddings for specified text columns
- Configurable API endpoint and model via config file or environment variables
- Asynchronous batch processing of texts

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Configuration

The server can be configured through `config.yaml` or environment variables:

```yaml
embedding:
  api_url: "https://ollama.dc.dev1.acme.com/v1/embeddings"
  model: "gte-qwen2"
```

Environment variables (override config.yaml):
- `EMBEDDING_API_URL`: URL of the embedding API
- `EMBEDDING_MODEL`: Model name to use for embeddings

## Usage

1. Start the server:
```bash
python -m embedding
```

2. Use the MCP client to interact with the server:
```python
from mcp.client import Client

async with Client() as client:
    result = await client.process_parquet(
        file_path="data.parquet",
        text_column="text"
    )
    print(result)
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Format code:
```bash
black embedding/
ruff check --fix embedding/
```

## Project Structure

```
embedding/
├── embedding_server.py     # Main server implementation
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
