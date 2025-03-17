import asyncio
import os
import re
from typing import Optional, List, Tuple, Dict, Literal, Type

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from rich.console import Console
from autogen_agents_deep_research import termination, Investment_thesis_assistant, Search_queries_generator, Company_data_fetcher_pitchbook, Data_standardizer_cleaner, Initial_screening_filter, Deal_sourcing_report_generator

load_dotenv()


class InvestmentThesis(BaseModel):
    """Pydantic model for investment thesis parameters"""

    industry_focus: List[str] = Field(..., description="List of target industries")
    geography: List[str] = Field(..., description="List of target geographic regions")
    revenue_range_min: int = Field(..., description="Minimum revenue in USD", ge=0)
    revenue_range_max: int = Field(..., description="Maximum revenue in USD", gt=0)
    ebitda_margin_min: float = Field(
        ..., description="Minimum EBITDA margin as decimal", ge=0, le=1
    )
    investment_type: Literal["Growth Equity", "Buyout", "Venture Capital", "Debt"] = (
        Field(..., description="Type of investment strategy")
    )
    stage: Optional[List[str]] = Field(None, description="Company stages of interest")
    yoy_growth_min: Optional[float] = Field(
        None, description="Minimum year-over-year growth rate", ge=0
    )
    target_ownership: Optional[float] = Field(
        None, description="Target ownership percentage", ge=0, le=1
    )


async def get_investment_info(investment_thesis: InvestmentThesis) -> str:
    """Get investment information based on the provided thesis."""
    # Actual implementation would call a database or API
    return f"Investment thesis: {investment_thesis.json()}"


def get_user_prompt(model_type: type[BaseModel]) -> str:
    """
    Generate a user-friendly prompt based on the Pydantic model.

    Args:
        model_type: The Pydantic model class to extract parameters for

    Returns:
        A prompt string describing what information is needed
    """
    model_name = model_type.__name__
    friendly_model_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", model_name).lower()

    model_fields = model_type.__annotations__
    field_names = [re.sub(r"_", " ", field) for field in model_fields.keys()]

    if len(field_names) == 1:
        fields_text = field_names[0]
    elif len(field_names) == 2:
        fields_text = f"{field_names[0]} and {field_names[1]}"
    else:
        fields_text = ", ".join(field_names[:-1]) + f", and {field_names[-1]}"

    prompt = f"For {friendly_model_name}, please provide {fields_text} (or type 'exit' to quit): "

    return prompt


def extract_params(
    user_input: str, model_type: Type[BaseModel]
) -> Tuple[BaseModel, Dict[str, str]]:
    """
    Generic parameter extraction function that works with any Pydantic model.

    Args:
        user_input: The text input from the user
        model_type: The Pydantic model class to extract parameters for

    Returns:
        A tuple containing:
        - An instance of the model_type with extracted values
        - A dictionary of extraction messages for each field
    """
    extracted_values = {}
    extraction_messages = {}

    model_name = model_type.__name__
    friendly_model_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", model_name).lower()

    model_fields = model_type.__annotations__

    extraction_patterns = {
        "industry_focus": [
            r"(?:industries|industry|focus|industries of interest|industry focus|target industries|target industry|industries targeted)(?:\s+is|:)\s+([A-Za-z\s,]+?)(?:\s|,|\.|!|\?)",
            r"([A-Za-z\s,]+)(?:\s+is|:)\s+(?:industry|industries)(?:\s|,|\.|!|\?)",
        ],
        "geography": [
            r'(?:geography)(?: +is|:)+([A-Za-z\s]+)',  # Match 'geography is' followed by the location
            r'([A-Za-z\s,]+)(?: +is|:)+(?:location|locations|region|regions|geography)(?: |,|\.|\!|\?)'  # General case
        ],
        "revenue_range_min": [
            r"(?:revenue minimum|min revenue|minimum revenue|revenue range min)(?:\s+is|:)\s+(\d+)(?:\s|,|\.|!|\?)",
            r"(\d+)(?:\s+is|:)\s+(?:minimum revenue|min revenue|revenue min)(?:\s|,|\.|!|\?)",
        ],
        "revenue_range_max": [
            r"(?:revenue maximum|max revenue|maximum revenue|revenue range max)(?:\s+is|:)\s+(\d+)(?:\s|,|\.|!|\?)",
            r"(\d+)(?:\s+is|:)\s+(?:maximum revenue|max revenue|revenue max)(?:\s|,|\.|!|\?)",
        ],
        "ebitda_margin_min": [
            r"(?:ebitda margin minimum|min ebitda margin|minimum ebitda margin|ebitda margin min)(?:\s+is|:)\s+(\d+(?:\.\d+)?)(?:\s|,|\.|!|\?)",
            r"(\d+(?:\.\d+)?)(?:\s+is|:)\s+(?:minimum ebitda margin|min ebitda margin|ebitda margin min)(?:\s|,|\.|!|\?)",
        ],
        "investment_type": [
            r"(?:investment type|type of investment|investment strategy|strategy)(?:\s+is|:)\s+(Growth Equity|Buyout|Venture Capital|Debt)(?:\s|,|\.|!|\?)",
            r"(Growth Equity|Buyout|Venture Capital|Debt)(?:\s+is|:)\s+(?:investment|strategy)(?:\s|,|\.|!|\?)",
        ],
        "stage": [
            r"(?:stage|company stage|stages of interest|company stages)(?:\s+is|:)\s+([A-Za-z\s,]+?)(?:\s|,|\.|!|\?)",
            r"([A-Za-z\s,]+)(?:\s+is|:)\s+(?:stage|stages)(?:\s|,|\.|!|\?)",
        ],
        "yoy_growth_min": [
            r"(?:yoy growth minimum|min yoy growth|minimum yoy growth|yoy growth min)(?:\s+is|:)\s+(\d+(?:\.\d+)?)(?:\s|,|\.|!|\?)",
            r"(\d+(?:\.\d+)?)(?:\s+is|:)\s+(?:minimum yoy growth|min yoy growth|yoy growth min)(?:\s|,|\.|!|\?)",
        ],
        "target_ownership": [
            r"(?:target ownership|ownership percentage|target ownership percentage)(?:\s+is|:)\s+(\d+(?:\.\d+)?)(?:\s|,|\.|!|\?)",
            r"(\d+(?:\.\d+)?)(?:\s+is|:)\s+(?:ownership|target ownership)(?:\s|,|\.|!|\?)",
        ],
    }
    for field_name, field_type in model_fields.items():
        field_type_str = str(field_type).lower()
        friendly_field_name = field_name.replace("_", " ")

        prefix_message = f"In extracting the {friendly_model_name}, looking for {friendly_field_name}:"

        patterns = []

        if field_name.lower() in extraction_patterns:
            patterns.extend(extraction_patterns[field_name.lower()])

        if "str" in field_type_str:
            if not patterns:
                patterns.append(
                    rf"(?:{field_name}(?:\s+is|:))\s+([A-Za-z0-9\s,]+?)(?:\s|,|\.|!|\?)"
                )
        elif "int" in field_type_str:
            if not patterns:
                patterns.append(
                    rf"(?:{field_name}(?:\s+is|:))\s+(\d+)(?:\s|,|\.|!|\?)"
                )
        elif "float" in field_type_str:
            if not patterns:
                patterns.append(
                    rf"(?:{field_name}(?:\s+is|:))\s+(\d+(?:\.\d+)?)(?:\s|,|\.|!|\?)"
                )
        elif "list" in field_type_str:
            if not patterns:
                patterns.append(
                    rf"(?:{field_name}(?:\s+is|:))\s+([A-Za-z0-9\s,]+?)(?:\s|,|\.|!|\?)"
                )

        for pattern in patterns:
            try:
                matches = None
                for pattern in patterns:
                    matches = re.search(pattern, user_input, re.IGNORECASE)
                    if matches:
                        break  # Stop iterating if a match is found

                if matches:
                    extracted_values[field_name] = matches.group(1).strip()
                    extraction_messages[field_name] = f"{prefix_message} Found: {matches.group(1).strip()}"
                    break             
                else:
                    extraction_messages[field_name] = f"{prefix_message} Not found."

            except re.error as e:
                extraction_messages[field_name] = f"{prefix_message} Regex error: {e}"
                print(f"Regex error for pattern '{pattern}': {e}")


    try:
        model_instance = model_type(**extracted_values)
        return model_instance, extraction_messages
    except ValidationError as e:
        error_messages = {}
        for error in e.errors():
            field = ".".join(error["loc"])
            msg = error["msg"]
            error_messages[field] = f"{field}: {msg}"
        return None, error_messages
    except ValueError as e:
        return None, {"error": str(e)}


async def main():
    model_client = AzureOpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_type="azure",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        seed=42,
        temperature=0,
    )


    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """
    team = SelectorGroupChat(
        [
            Investment_thesis_assistant,
            Search_queries_generator,
            Company_data_fetcher_pitchbook,
            Data_standardizer_cleaner,
            Initial_screening_filter,
            Deal_sourcing_report_generator,
        ],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
    )


    PE_TASK = "create an investment thesis, use it for a search and analyze potential deal"

    request_model = InvestmentThesis

    message_history = []

    user_prompt = get_user_prompt(request_model)

    while True:
        user_input = input(f"{user_prompt}")
        if user_input.lower() == "exit":
            break

        message = TextMessage(content=user_input, source="user")
        message_history.append(message)

        try:
            investment_params, extraction_messages = extract_params(
                user_input, request_model
            )

            for message in extraction_messages.values():
                print(f"System: {message}")

            if investment_params is None or not all(
                [
                    getattr(investment_params, "industry_focus", None),
                    getattr(investment_params, "geography", None),
                    getattr(investment_params, "revenue_range_min", None),
                    getattr(investment_params, "revenue_range_max", None),
                    getattr(investment_params, "ebitda_margin_min", None),
                    getattr(investment_params, "investment_type", None),
                ]
            ):

                print("System: I need more information to fulfill the investment thesis.")
                # You could add further logic here to prompt for missing parameters
                # one by one, similar to the weather example.
                pass
            else:
                investment_result = await get_investment_info(investment_params)
                message_history.append(TextMessage(content=investment_result, source="system"))

                await Console(team.run_stream(task=PE_TASK, history=message_history))
                break

        except Exception as e:
            print(f"Error: {e}")

asyncio.run(main())