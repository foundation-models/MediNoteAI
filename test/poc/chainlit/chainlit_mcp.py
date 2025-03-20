import json
import os
import re
import traceback
from typing import Dict, Any

from mcp import ClientSession as McpClientSession
import chainlit as cl
from openai import AsyncOpenAI
from dotenv import load_dotenv


load_dotenv(".env")  # Load environment variables from .env file

SYSTEM_PROMPT = "you are a helpful assistant."

class ChatClient:
    def __init__(self) -> None:
        self.deployment_name = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")  # Default to GPT-4 Turbo
        self.client = AsyncOpenAI()  # Will automatically use OPENAI_API_KEY from environment
        self.messages = []
        self.system_prompt = SYSTEM_PROMPT

    async def process_response_stream(self, response_stream, tools=None, temperature=0):
        """
        Process response streams from OpenAI.
        """
        collected_messages = []
        
        try:
            async for part in response_stream:
                if part.choices == []:
                    continue
                delta = part.choices[0].delta
                finish_reason = part.choices[0].finish_reason
                
                # Process assistant content
                if delta.content:
                    collected_messages.append(delta.content)
                    yield delta.content
                
                # Handle tool calls if tools are provided
                if tools and delta.tool_calls:
                    if len(delta.tool_calls) > 0:
                        tool_call = delta.tool_calls[0]
                        if tool_call.function.name:
                            function_name = tool_call.function.name
                            function_args = tool_call.function.arguments
                            try:
                                # Call the tool
                                result = await call_tool(function_name, json.loads(function_args))
                                # Add the tool response to messages
                                self.messages.append({
                                    "role": "tool",
                                    "name": function_name,
                                    "content": str(result)
                                })
                            except Exception as e:
                                print(f"Error calling tool {function_name}: {e}")
                                traceback.print_exc()
                
                # Check if we've reached the end of assistant's response
                if finish_reason == "stop":
                    # Add final assistant message if there's content
                    if collected_messages:
                        final_content = ''.join([msg for msg in collected_messages if msg is not None])
                        if final_content.strip():
                            self.messages.append({"role": "assistant", "content": final_content})
                    return
        except GeneratorExit:
            return
        except Exception as e:
            print(f"Error in process_response_stream: {e}")
            traceback.print_exc()
    
    async def generate_response(self, human_input, tools=None, temperature=0):
        print(f"human_input: {human_input}")
        self.messages.append({"role": "user", "content": human_input})
        response_stream = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=self.messages,
            tools=tools,
            stream=True,
            temperature=temperature
        )
        try:
            async for token in self.process_response_stream(response_stream, tools, temperature):
                yield token
        except GeneratorExit:
            return

@cl.on_chat_start
async def start_chat():
    client = ChatClient()
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
        tools.extend([{"type": "function", "function": tool} for tool in connection_tools])
    
    # Create a fresh client instance for each message
    client = ChatClient()
    # Restore conversation history
    client.messages = cl.user_session.get("messages", [])
    
    msg = cl.Message(content="")
    async for text in client.generate_response(human_input=message.content, tools=tools):
        await msg.stream_token(text)
    
    # Update the stored messages after processing
    cl.user_session.set("messages", client.messages)

