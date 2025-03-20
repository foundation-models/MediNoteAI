import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from dotenv import load_dotenv
import chainlit as cl
import yaml
from autogen_core.models import ChatCompletionClient
from typing import List, cast
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage

load_dotenv("/home/agent/workspace/guideline2actions/src/aryoGen/.env")
async def user_input_func(
    prompt: str, cancellation_token: CancellationToken | None = None
) -> str:
    """
    Ask the user for input through the Chainlit interface. If no input is received within a time limit, it returns a default message.

    Args:
        prompt (str): The prompt message to be displayed to the user.
        cancellation_token (CancellationToken | None): Optional cancellation token to handle task cancellation.

    Returns:
        str: The user's input or a timeout message if no input is received within the time limit.

    """
    try:
        response = await cl.AskUserMessage(content=prompt).send()
    except TimeoutError:
        return "User did not provide any input within the time limit."
    if response:
        return response["output"]  # type: ignore
    else:
        return "User did not provide any input."


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    """
    Initialize the chat system, loads the configuration, and sets up agents (including WebSurfer, Assistant, and User agents).
    It also defines the roles, selector prompt, and termination conditions for the group chat.

    This function loads the model configuration, initializes the agents, and stores the team in the user session.

    """

    server_params = StdioServerParams(
        command="npx",
        args=["openai-websearch-mcp-server"]
    )
    tools = await mcp_server_tools(server_params)
    # Load model configuration
    with open("model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_config)

    fetcher = AssistantAgent(
        name="fetcher",
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=True
    )
    assistant = AssistantAgent(
        name="ai_assistant",
        system_message="""
        You are responsible for guiding all research operations efficiently without asking unnecessary questions from the user.
        Follow this structured process:

            ### **Step 1:
            - Extract **entity name** from the user request.
            ### **Step 2:
            - Find **entity details** from relevant websites.
            - Collect key information such as:
                - Specifications
                - Features
                - Pricing (if applicable)
                - Reviews/Ratings
                - Availability
            ### **Step 3:
            - Research **market comparisons** and alternatives.
            - Focus on similar entities with comparable features.
            ### **Step 4: Generate the Final Output**
            - Once all searches are complete, **format the findings into a structured table**:
                - **Entity Name and Description**
                - **Key Specifications**
                - **Price Range** (if applicable)
                - **Available Sources/Retailers**
                - **Website References** (with URLs)
                ## **Alternatives
                - **Alternative Options**
                - **Comparative Analysis**
                - **References** (with URLs)
            
            Talk to fetcher until you make sure all tasks are done and you fill out all table cells.
            Only if you complete the Table call 'APPROVE'.          
        """,
        model_client=model_client,
        model_client_stream=True,
    )
    inner_termination = TextMentionTermination("APPROVE")
    inner_team = RoundRobinGroupChat([fetcher, assistant], termination_condition=inner_termination)

    society_of_mind_agent = SocietyOfMindAgent("society_of_mind", team=inner_team, model_client=model_client)

    user = UserProxyAgent(
        name="user",
        description="Represents the human user in the conversation.",
        input_func=user_input_func,
    )
    termination = TextMentionTermination("TERMINATE", sources=["user"])
    print("Agents initialized successfully.")  # Debugging log

    team = RoundRobinGroupChat([user,society_of_mind_agent], termination_condition=termination)

    cl.user_session.set("prompt_history", "")
    cl.user_session.set("team", team)

    print("Chat system initialized.")
    # Run the agent and stream the response to the console
    # await Console(team.run_stream(task="Ask user what he want and help them find all info", cancellation_token=CancellationToken()))




@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    """
    Handle the conversation by processing messages from the user and streaming responses from the agents.
    It manages the task assignments, message streaming, and task termination.

    Args:
        message (cl.Message): The message received from the user.

    """
    # Get the chat system
    team = cast(RoundRobinGroupChat, cl.user_session.get("team"))

    streaming_response: cl.Message | None = None
    # Stream the messages from the team.
    async for msg in team.run_stream(
        task=[TextMessage(content=message.content, source="user")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(msg, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            if streaming_response is None:
                # Start a new streaming response.
                streaming_response = cl.Message(content="", author=msg.source)
            await streaming_response.stream_token(msg.content)
        elif streaming_response is not None:
            # Done streaming the model client response.
            # We can skip the current message as it is just the complete message
            # of the streaming response.
            await streaming_response.send()
            # Reset the streaming response so we won't enter this block again
            # until the next streaming response is complete.
            streaming_response = None
        elif isinstance(msg, TaskResult):
            # Send the task termination message.
            final_message = "Task terminated. "
            if msg.stop_reason:
                final_message += msg.stop_reason
            await cl.Message(content=final_message).send()
        else:
            # Skip all other message types.
            pass