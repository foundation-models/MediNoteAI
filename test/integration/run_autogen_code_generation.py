# create an AssistantAgent named "assistant"
# from IPython.display import Image, display
import os

import autogen
from autogen.coding import LocalCommandLineCodeExecutor
import tempfile

from autogen import ConversableAgent

# config_list = autogen.config_list_from_json(
#     "OAI_CONFIG_LIST",
#     filter_dict={"tags": ["gpt-4o"]},  # comment out to get all
# )
# When using a single openai endpoint, you can use the following:
# config_list = [{"model": "qwen2.5:latest", "api_key": os.environ["OPENAI_API_KEY"]}]
llm_config = {
    "base_url": f"{os.environ['OPENAI_ENDPOINT']}/v1",
    "model": "qwen2.5-coder:latest",
    "api_key": "ollama"
}

config_list = [llm_config]
# 
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "cache_seed": 41,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0,  # temperature for sampling
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        # the executor to run the generated code
        "executor": LocalCommandLineCodeExecutor(work_dir="coding"),
    },
)
# the assistant receives a message from the user_proxy, which contains the task description
chat_res = user_proxy.initiate_chat(
    assistant,
    message="""What date is today? Compare the year-to-date gain for META and TESLA. FOr this task you can use YOUR_ALPHA_VANTAGE_API_KEY=abc123""",
    summary_method="reflection_with_llm",
)
