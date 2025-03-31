import openai
from smolagents import CodeAgent, tool, LiteLLMModel

# Initialize OpenAI API client
openai.api_key = "your_openai_api_key"

# Define the custom tool
@tool
def fetch_deals_from_dealcloud(industry: str, deal_size: str) -> str:
    """
    Fetches deals from DealCloud based on industry and deal size.

    Args:
        industry: The industry sector to filter deals.
        deal_size: The size of the deals to filter.

    Returns:
        A summary of the fetched deals.
    """
    # Simulated data
    simulated_deals = [
        {"deal_id": 1, "industry": "Technology", "deal_size": "Large", "description": "Acquisition of a software company."},
        {"deal_id": 2, "industry": "Healthcare", "deal_size": "Medium", "description": "Merger between two pharmaceutical firms."},
        {"deal_id": 3, "industry": "Finance", "deal_size": "Small", "description": "Investment in a fintech startup."}
    ]
    
    # Filter simulated deals based on input parameters
    filtered_deals = [
        deal for deal in simulated_deals
        if (industry.lower() in deal["industry"].lower() and deal_size.lower() in deal["deal_size"].lower())
    ]
    
    # Format the filtered deals into a summary string
    if filtered_deals:
        summary = "\n".join(
            [f"Deal ID: {deal['deal_id']}\nIndustry: {deal['industry']}\nDeal Size: {deal['deal_size']}\nDescription: {deal['description']}\n" for deal in filtered_deals]
        )
    else:
        summary = "No deals found matching the specified criteria."
    
    return summary

# Initialize the LiteLLMModel with GPT-4o Mini
model = LiteLLMModel(model_id="gpt-4o-mini")

# Create the agent
agent = CodeAgent(
    tools=[fetch_deals_from_dealcloud],
    model=model,
    additional_authorized_imports=["openai"]
)

# Run the agent with a task
task = "Find recent deals in the technology industry with a large deal size."
result = agent.run(task)
print(result)
# Run the agent with a task
task = "Find any deals in the finance industry with a large deal size."
result = agent.run(task)
print(result)
# Run the agent with a task
task = "Find any deals in the finance industry with a small deal size."
result = agent.run(task)
print(result)
