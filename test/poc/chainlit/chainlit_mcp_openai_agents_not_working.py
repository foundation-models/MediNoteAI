import json
import os
import re
import traceback
from typing import Dict, Any, List
import functools

from mcp import ClientSession as McpClientSession
import chainlit as cl
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, enable_verbose_stdout_logging


load_dotenv(".env")  # Load environment variables from .env file

SYSTEM_PROMPT = "you are a helpful assistant."

# Enable verbose logging from the agents SDK if needed
# enable_verbose_stdout_logging()

class ChatClient:
    def __init__(self) -> None:
        self.deployment_name = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")  # Default to GPT-4 Turbo
        self.client = AsyncOpenAI()  # Will automatically use OPENAI_API_KEY from environment
        self.messages = []
        self.system_prompt = SYSTEM_PROMPT
        
    async def create_agent_with_tools(self, tools_list):
        """
        Create an OpenAI Agent with the provided tools
        """
        # Convert MCP tools to Agent SDK function tools
        agent_tools = []
        
        for tool in tools_list:
            # Create a wrapper for each tool that properly captures the tool name
            # Use a factory function to create a proper closure
            def create_tool_wrapper(tool_name, tool_desc, tool_schema):
                async def wrapper(**kwargs):
                    try:
                        # Call the MCP tool using the call_tool function
                        result = await call_tool(tool_name, kwargs)
                        return result
                    except Exception as e:
                        return f"Error calling {tool_name}: {str(e)}"
                
                # Apply the function_tool decorator with the correct parameters
                # Using name_override to set the tool name
                decorated_wrapper = function_tool(
                    wrapper,
                    name_override=tool_name,
                    description_override=tool_desc,
                    strict_mode=False  # Set to False if you want more flexible parameter handling
                )
                
                # Set the input schema for the tool
                decorated_wrapper.schema = tool_schema
                
                return decorated_wrapper
            
            # Create a unique tool function for each tool
            tool_func = create_tool_wrapper(
                tool["name"], 
                tool["description"], 
                tool["input_schema"]
            )
            agent_tools.append(tool_func)
        
        # Create the agent with the tools
        agent = Agent(
            name="MediNote Assistant",
            instructions=self.system_prompt + "\n\nAnswer the user's questions directly. When appropriate, use the available tools.",
            model=self.deployment_name,
            tools=agent_tools
        )
        
        return agent

    async def generate_response(self, human_input, tools=None, temperature=0):
        print(f"human_input: {human_input}")
        self.messages.append({"role": "user", "content": human_input})
        
        if tools:
            # Use the Agents SDK to handle the message
            try:
                # Create an agent with the tools
                agent = await self.create_agent_with_tools(tools)
                
                # Create a context for the agent
                class AgentContext:
                    pass
                
                # Use Runner.run_streamed as a static method
                result = Runner.run_streamed(
                    agent,
                    input=human_input,
                    context=AgentContext()
                )
                
                # Stream events from the result
                async for event in result.stream_events():
                    # If it's a raw response event with text
                    if (event.type == "raw_response_event" and 
                        hasattr(event, "data") and 
                        hasattr(event.data, "delta")):
                        yield event.data.delta
                    # Handle other streaming events as needed
                
                # Note: The agent will handle tool calls internally
                
            except Exception as e:
                error_message = f"Error using agent: {str(e)}"
                print(error_message)
                traceback.print_exc()
                yield error_message
        else:
            # Fallback to direct LLM if no tools are available
            response_stream = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=self.messages,
                stream=True,
                temperature=temperature
            )
            
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        # Add the complete response to messages history
        # Note: In a complete implementation, we would collect the full response
        # and add it to the message history

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("messages", [])
    cl.user_session.set("system_prompt", SYSTEM_PROMPT)
    cl.user_session.set("chat_messages", [])

@cl.on_mcp_connect
async def on_mcp(connection, session: McpClientSession):
    try:
        result = await session.list_tools()
        tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
            } for t in result.tools]
        
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)
        print(f"Successfully connected to MCP and loaded {len(tools)} tools")
    except Exception as e:
        print(f"Error in on_mcp_connect: {e}")
        traceback.print_exc()

@cl.step(type="tool")
async def call_tool(function_name: str, function_args: Dict[str, Any]):
    try:
        print(f"Calling tool: {function_name} with args: {function_args}")
        # Get the MCP session
        mcp_session, _ = cl.context.session.mcp_sessions.get("default")
        # Call the tool
        result = await mcp_session.call_tool(function_name, function_args)
        return result
    except Exception as e:
        print(f"Error in call_tool: {e}")
        traceback.print_exc()
        return json.dumps({"error": str(e)})

@cl.on_message
async def on_message(message: cl.Message):   
    # Get available tools
    mcp_tools = cl.user_session.get("mcp_tools", {})
    tools = []
    for connection_tools in mcp_tools.values():
        tools.extend(connection_tools)
    
    # Create a fresh client instance for each message
    client = ChatClient()
    
    msg = cl.Message(content="")
    async for text in client.generate_response(human_input=message.content, tools=tools):
        await msg.stream_token(text)
    
    await msg.send()

