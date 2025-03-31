import streamlit as st
import asyncio
from agents import Agent, Runner, enable_verbose_stdout_logging

# Enable verbose logging
enable_verbose_stdout_logging()

# Create an agent that references MCP servers
agent = Agent(
    name="MCP Assistant",
    instructions="""You are a helpful assistant with access to filesystem tools.
When searching for files, use filesystem-list_directory to list files in a directory.
When reading files, use filesystem-read_file to get the contents of specific files.""",
    mcp_servers=["filesystem"]
)

class AgentContext:
    pass

async def get_agent_response(user_input: str) -> str:
    """Get response from the agent for the given user input."""
    result = await Runner.run(
        agent,
        input=user_input,
        context=AgentContext()
    )
    return result.final_output

def main():
    st.title("MCP Agent Assistant")
    st.write("Ask me anything about the filesystem or any other tasks!")

    # Create a text input for user queries
    user_input = st.text_area("Enter your query:", height=100)
    
    # Create a button to submit the query
    if st.button("Submit"):
        if user_input:
            with st.spinner("Processing your request..."):
                # Run the agent in the async event loop
                response = asyncio.run(get_agent_response(user_input))
                st.write("### Response:")
                st.write(response)
        else:
            st.warning("Please enter a query first.")

    # Add some helpful examples
    with st.expander("Example queries"):
        st.write("""
        - List files in the current directory
        - Read the contents of a specific file
        - Search for files with a specific extension
        - Show me the contents of pyproject.toml
        """)

if __name__ == "__main__":
    main() 