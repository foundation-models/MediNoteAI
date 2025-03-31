import asyncio
import logging
from agents import Agent, Runner, handoff, enable_verbose_stdout_logging
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# Enable verbose logging
enable_verbose_stdout_logging()
logging.basicConfig(level=logging.INFO)

# Create two agents that will hand off to each other
agent_a = Agent(
    name="Agent A",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are Agent A. Your role is to:
    1. Greet the user
    2. Make a simple observation about nature or science
    3. Explicitly use the transfer_to_agent_b tool to continue the conversation
    
    Always be polite and friendly in your responses.
    You MUST use the transfer_to_agent_b tool at the end of your response."""
)

agent_b = Agent(
    name="Agent B",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are Agent B. Your role is to:
    1. Acknowledge what Agent A said
    2. Add your own observation about technology or society
    3. Explicitly use the transfer_to_agent_a tool to continue the conversation
    
    Always be polite and friendly in your responses.
    You MUST use the transfer_to_agent_a tool at the end of your response."""
)

# Define handoff callbacks
async def on_handoff_to_b(ctx):
    logging.info("Handing off from Agent A to Agent B")

async def on_handoff_to_a(ctx):
    logging.info("Handing off from Agent B to Agent A")

# Set up the handoffs between agents with callbacks
agent_a.handoffs = [handoff(agent_b, on_handoff=on_handoff_to_b)]
agent_b.handoffs = [handoff(agent_a, on_handoff=on_handoff_to_a)]

async def main():
    result = await Runner.run(
        agent_a,
        input="""Hello! Please start a conversation with me and then use the transfer_to_agent_b tool 
        to hand off to Agent B. Remember to explicitly use the handoff tool in your response.""",
        max_turns=6  # Limit the number of turns to prevent infinite loop
    )
    
    print("\nFinal conversation output:")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main()) 