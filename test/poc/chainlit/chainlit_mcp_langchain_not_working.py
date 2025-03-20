import json
import os
import re
import traceback
from typing import Dict, Any, List, Callable, Optional
from functools import partial

from mcp import ClientSession as McpClientSession
import chainlit as cl
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langchain_core.tools import Tool, ToolException
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import asyncio


load_dotenv(".env")  # Load environment variables from .env file

SYSTEM_PROMPT = """You are a helpful assistant with access to various tools. When using tools:
1. Always check the tool's description and required parameters
2. Only pass parameters that are specified in the tool's schema
3. If a tool doesn't require parameters, pass an empty dict
4. Format all responses in a clear, user-friendly way"""

def sync_wrapper(async_func: Callable, *args, **kwargs) -> Any:
    """Wrapper to run async functions in sync context."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_func(*args, **kwargs))

class LangchainAgentClient:
    def __init__(self) -> None:
        self.deployment_name = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.llm = ChatOpenAI(
            model=self.deployment_name,
            temperature=0,
            streaming=True
        )
        self.messages: List[HumanMessage | AIMessage | SystemMessage] = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
        self.tools = []
        self.agent_executor = None
        self._mcp_session = None

    def _create_tool_function(self, tool_name: str, schema: Optional[Dict] = None) -> Callable:
        """Create a function that handles tool execution with proper argument validation."""
        async def tool_func(**kwargs):
            try:
                # Validate arguments against schema if provided
                if schema:
                    # Basic validation - you might want to add more sophisticated validation
                    required_params = schema.get("required", [])
                    for param in required_params:
                        if param not in kwargs:
                            raise ToolException(f"Missing required parameter: {param}")

                result = await self._call_tool(tool_name, kwargs)
                return result
            except Exception as e:
                raise ToolException(str(e))
        
        return lambda **kwargs: sync_wrapper(tool_func, **kwargs)

    def setup_agent(self, tools: List[Dict[str, Any]], mcp_session: McpClientSession) -> None:
        """Setup Langchain agent with the provided tools."""
        self._mcp_session = mcp_session
        
        # Convert MCP tools to Langchain tools
        self.tools = [
            Tool(
                name=tool["name"],
                description=tool["description"],
                func=self._create_tool_function(tool["name"], tool.get("input_schema")),
                args_schema=tool.get("input_schema")
            )
            for tool in tools
        ]

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        agent = create_openai_functions_agent(
            self.llm,
            self.tools,
            prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )

    async def _call_tool(self, tool_name: str, tool_args: Dict[str, Any] = None) -> str:
        """Call MCP tool through Chainlit's context."""
        try:
            print(f"Calling tool: {tool_name} with args: {tool_args}")
            if not self._mcp_session:
                raise ValueError("MCP session not initialized")
            
            # Ensure tool_args is a dict
            if tool_args is None:
                tool_args = {}
            elif not isinstance(tool_args, dict):
                tool_args = {}
            
            result = await self._mcp_session.call_tool(tool_name, tool_args)
            return str(result)
        except Exception as e:
            print(f"Error in _call_tool: {e}")
            traceback.print_exc()
            raise ToolException(str(e))

    async def generate_response(self, human_input: str) -> str:
        """Generate response using Langchain agent."""
        if not self.agent_executor:
            yield "Agent not initialized. Please wait for MCP connection."
            return

        try:
            # Add user message to history
            self.messages.append(HumanMessage(content=human_input))

            # Run agent with streaming
            final_response = []
            async for chunk in self.agent_executor.astream(
                {
                    "input": human_input,
                    "chat_history": self.messages[:-1]  # Exclude the last message as it's the current input
                }
            ):
                if isinstance(chunk, str):
                    final_response.append(chunk)
                    yield chunk
                elif isinstance(chunk, dict) and "output" in chunk:
                    final_response.append(chunk["output"])
                    yield chunk["output"]

            # Add agent's response to history
            self.messages.append(AIMessage(content="".join(final_response)))

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            yield error_msg

@cl.on_chat_start
async def start_chat():
    client = LangchainAgentClient()
    cl.user_session.set("client", client)

@cl.on_mcp_connect
async def on_mcp(connection, session: McpClientSession):
    try:
        result = await session.list_tools()
        tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        } for t in result.tools]
        
        # Get the client and set up the agent with tools
        client = cl.user_session.get("client")
        client.setup_agent(tools, session)
        
        print(f"Successfully connected to MCP and loaded {len(tools)} tools")
    except Exception as e:
        print(f"Error in on_mcp_connect: {e}")
        traceback.print_exc()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        # Get the client from session
        client = cl.user_session.get("client")
        if not client:
            await cl.Message(content="Session not initialized properly.").send()
            return

        # Create streaming message
        msg = cl.Message(content="")
        async for chunk in client.generate_response(message.content):
            await msg.stream_token(chunk)
        
        await msg.send()
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        await cl.Message(content=error_msg).send()

