import json
from mcp import ClientSession
import os
import re
from aiohttp import ClientSession
import chainlit as cl
from openai import OpenAI, AsyncOpenAI
import traceback
from dotenv import load_dotenv
import ast
load_dotenv(".env")  # Change to standard .env file

SYSTEM_PROMPT = "you are a helpful assistant."

class ChatClient:
    def __init__(self) -> None:
        self.deployment_name = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")  # Default to GPT-4 Turbo
        self.client = AsyncOpenAI()  # Will automatically use OPENAI_API_KEY from environment
        self.messages = []
        self.system_prompt = SYSTEM_PROMPT
    async def process_response_stream(self, response_stream, tools, temperature=0):
        """
        Recursively process response streams to handle multiple sequential function calls.
        This function can call itself when a function call is completed to handle subsequent function calls.
        """
        function_arguments = ""
        function_name = ""
        tool_call_id = ""
        is_collecting_function_args = False
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
                
                # Handle tool calls
                if delta.tool_calls:
                    if len(delta.tool_calls) > 0:
                        tool_call = delta.tool_calls[0]
                        
                        # Get function name
                        if tool_call.function.name:
                            function_name = tool_call.function.name
                            tool_call_id = tool_call.id
                        
                        # Process function arguments delta
                        if tool_call.function.arguments:
                            function_arguments += tool_call.function.arguments
                            is_collecting_function_args = True
                
                # Check if we've reached the end of a tool call
                if finish_reason == "tool_calls" and is_collecting_function_args:
                    # Process the current tool call
                    print(f"function_name: {function_name} function_arguments: {function_arguments}")
                    function_args = json.loads(function_arguments)
                    mcp_tools = cl.user_session.get("mcp_tools", {})
                    mcp_name = None
                    for connection_name, session_tools in mcp_tools.items():
                        if any(tool.get("name") == function_name for tool in session_tools):
                            mcp_name = connection_name
                            break
                    
                    reply_to_customer = function_args.get('reply_to_customer')
                    print(f"reply_to_customer: {reply_to_customer}")
                    # Output any replies to the customer
                    if reply_to_customer:
                        tokens = re.findall(r'\s+|\w+|[^\w\s]', reply_to_customer)
                        for token in tokens:
                            yield token
                    
                    # Add the assistant message with tool call
                    self.messages.append({
                        "role": "assistant", 
                        "content": reply_to_customer,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "function": {
                                    "name": function_name,
                                    "arguments": function_arguments
                                },
                                "type": "function"
                            }
                        ]
                    })
                    func_response = await call_tool(mcp_name, function_name, function_args)
                    # Add the tool response
                    self.messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": func_response,
                    })
                    
                    # Create a new stream to continue processing
                    new_response = await self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=self.messages,
                        tools=tools,
                        parallel_tool_calls=False,
                        stream=True,
                        temperature=temperature
                    )
                    
                    # Use a separate try block for recursive processing
                    try:
                        async for token in self.process_response_stream(new_response, tools, temperature):
                            yield token
                    except GeneratorExit:
                        return
                    return
                
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
    
    # Main entry point that uses the recursive function
    async def generate_response(self, human_input, tools, temperature=0):
        print(f"human_input: {human_input}")
        self.messages.append({"role": "user", "content": human_input})
        response_stream = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=self.messages,
            tools=tools,
            parallel_tool_calls=False,
            stream=True,
            temperature=temperature
        )
        try:
        # Process the initial stream with our recursive function
            async for token in self.process_response_stream(response_stream, tools, temperature):
                yield token
        except GeneratorExit:
            return

def flatten(xss):
    return [x for xs in xss for x in xs]


@cl.on_chat_start
async def start_chat():
    client = ChatClient()
    cl.user_session.set("messages", [])
    cl.user_session.set("system_prompt", SYSTEM_PROMPT)
    cl.user_session.set("chat_messages", [])
    
@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
        } for t in result.tools]
    
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)



@cl.step(type="tool") 
async def call_tool(mcp_name, function_name, function_args):
    try:
        print(f"Function Name: {function_name} Function Args: {function_args}")
        mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
        func_response = await mcp_session.call_tool(function_name, function_args)
    except Exception as e:
        traceback.print_exc()
        func_response = json.dumps({"error": str(e)})
    return str(func_response.content)
    
    # current_step = cl.context.current_step
    # current_step.name = tool_name
    
    # # Identify which mcp is used
    # mcp_tools = cl.user_session.get("mcp_tools", {})
    # mcp_name = None
    
    # for connection_name, tools in mcp_tools.items():
    #     if any(tool.get("name") == tool_name for tool in tools):
    #         mcp_name = connection_name
    #         break
    
    # if not mcp_name:
    #     current_step.output = json.dumps({"error": f"Tool {tool_name} not found in any MCP connection"})
    #     return current_step.output
    
    # mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    
    # if not mcp_session:
    #     current_step.output = json.dumps({"error": f"MCP {mcp_name} not found in any MCP connection"})
    #     return current_step.output
    
    # try:
    #     current_step.output = await mcp_session.call_tool(tool_name, tool_input)
    # except Exception as e:
    #     current_step.output = json.dumps({"error": str(e)})
    
    # return current_step.output

# async def call_claude(chat_messages):
#     msg = cl.Message(content="")
#     mcp_tools = cl.user_session.get("mcp_tools", {})
#     # Flatten the tools from all MCP connections
#     tools = flatten([tools for _, tools in mcp_tools.items()])
    
#     # Load OpenAI config
#     with open("MediNoteAI/test/poc/autogen/model_config.yaml", "r") as f:
#         config = yaml.safe_load(f)
    
#     # Initialize OpenAI client
#     client = AsyncOpenAI(
#         api_key=config["config"]["api_key"],
#         base_url="https://api.openai.com/v1"  # Using OpenAI's API directly
#     )
    
#     # Convert tools to OpenAI function format
#     openai_tools = [{
#         "type": "function",
#         "function": {
#             "name": tool["name"],
#             "description": tool["description"],
#             "parameters": tool["input_schema"]
#         }
#     } for tool in tools]
    
#     # Call OpenAI API
#     response = await client.chat.completions.create(
#         model=config["config"]["model"],
#         messages=chat_messages,
#         tools=openai_tools,
#         tool_choice="auto"
#     )
    
#     return response

@cl.on_message
async def on_message(message: cl.Message):   
    mcp_tools = cl.user_session.get("mcp_tools", {})
    tools = flatten([tools for _, tools in mcp_tools.items()])
    tools = [{"type": "function", "function": tool} for tool in tools]
    
    # Create a fresh client instance for each message
    client = ChatClient()
    # Restore conversation history
    client.messages = cl.user_session.get("messages", [])
    
    msg = cl.Message(content="")
    async for text in client.generate_response(human_input=message.content, tools=tools):
        await msg.stream_token(text)
    
    # Update the stored messages after processing
    cl.user_session.set("messages", client.messages)
    
    # chat_messages = cl.user_session.get("chat_messages")
    # chat_messages.append({"role": "user", "content": msg.content})
    # response = await call_claude(chat_messages)
    
    # while response.choices[0].message.tool_calls:
    #     tool_calls = response.choices[0].message.tool_calls
    #     for tool_call in tool_calls:
    #         tool_use = type('ToolUse', (), {
    #             'name': tool_call.function.name,
    #             'input': json.loads(tool_call.function.arguments),
    #             'id': tool_call.id
    #         })
    #         tool_result = await call_tool(tool_use)

    #         chat_messages.extend([
    #             {"role": "assistant", "content": None, "tool_calls": [
    #                 {
    #                     "id": tool_call.id,
    #                     "type": "function",
    #                     "function": {
    #                         "name": tool_call.function.name,
    #                         "arguments": tool_call.function.arguments
    #                     }
    #                 }
    #             ]},
    #             {
    #                 "role": "tool",
    #                 "content": str(tool_result),
    #                 "tool_call_id": tool_call.id
    #             }
    #         ])
        
    #     response = await call_claude(chat_messages)

    # final_response = response.choices[0].message.content

    # chat_messages.append({"role": "assistant", "content": final_response})
    # await cl.Message(content=final_response).send()