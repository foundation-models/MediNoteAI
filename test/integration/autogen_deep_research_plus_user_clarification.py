#!/usr/bin/env python3
"""
This code integrates the clarifying functionality from the Autogen weather example
(autogen_asking_user_for_clarification.py) into the deep research deal sourcing flow.
It replaces WeatherRequest with InvestmentThesis and asks the user for all the required
parts of the Investment Thesis before proceeding. The remaining AssistantAgents (such as
the deep research/deal sourcing agent) are kept the same.
"""

# Assuming the autogen framework is imported from your project
import autogen

# Define the InvestmentThesis data structure
class InvestmentThesis:
    def __init__(
        self,
        industry: str = None,
        market_opportunity: str = None,
        competitive_advantages: str = None,
        investment_amount: str = None,
        target_geography: str = None,
        time_horizon: str = None,
    ):
        self.industry = industry
        self.market_opportunity = market_opportunity
        self.competitive_advantages = competitive_advantages
        self.investment_amount = investment_amount
        self.target_geography = target_geography
        self.time_horizon = time_horizon

    def __str__(self):
        return (
            f"Industry: {self.industry}\n"
            f"Market Opportunity: {self.market_opportunity}\n"
            f"Competitive Advantages: {self.competitive_advantages}\n"
            f"Investment Amount: {self.investment_amount}\n"
            f"Target Geography: {self.target_geography}\n"
            f"Time Horizon: {self.time_horizon}\n"
        )

# Clarification Agent (adapted from the Weather clarifier) to ask for InvestmentThesis details
class InvestmentThesisClarificationAgent(autogen.AssistantAgent):
    def run(self, thesis: InvestmentThesis) -> InvestmentThesis:
        # Check each part of the InvestmentThesis and ask the user if missing.
        if not thesis.industry:
            thesis.industry = input("Please enter the industry of interest: ")
        if not thesis.market_opportunity:
            thesis.market_opportunity = input("Please describe the market opportunity: ")
        if not thesis.competitive_advantages:
            thesis.competitive_advantages = input("Please outline the competitive advantages: ")
        if not thesis.investment_amount:
            thesis.investment_amount = input("Please specify the investment amount: ")
        if not thesis.target_geography:
            thesis.target_geography = input("Please specify the target geography: ")
        if not thesis.time_horizon:
            thesis.time_horizon = input("Please provide the expected time horizon: ")
        return thesis

# Deep Research / Deal Sourcing Agent (kept the same as in the original deep research code)
class DeepResearchAgent(autogen.AssistantAgent):
    def run(self, thesis: InvestmentThesis):
        # Here, implement your deal sourcing logic based on the InvestmentThesis.
        # For demonstration purposes, we simply return a message with the thesis details.
        return f"Performing deal sourcing based on the following Investment Thesis:\n{thesis}"

# Main execution flow that uses the clarification before processing
def main():
    # Create an empty InvestmentThesis (all fields initially set to None)
    thesis = InvestmentThesis()
    
    # First, clarify all required InvestmentThesis fields by asking the user.
    clarifier = InvestmentThesisClarificationAgent()
    clarified_thesis = clarifier.run(thesis)
    
    # Next, run the deep research / deal sourcing agent with the clarified input.
    research_agent = DeepResearchAgent()
    result = research_agent.run(clarified_thesis)
    
    print(result)

if __name__ == "__main__":
    main()
