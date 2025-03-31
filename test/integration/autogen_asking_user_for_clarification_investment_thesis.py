from pydantic import BaseModel, Field, Literal
from typing import Optional, List, Tuple, Dict
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_agentchat.agents import AssistantAgent

import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
import re
import os

from dotenv import load_dotenv

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
    user_input: str, model_type: type[BaseModel]
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
            r"(?:industries|industry|focus|industries of interest|industry focus|target industries|target industry|industries targeted)(?:\s+is|:)\s+([A-Za-z\s,]+?)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(\[A\-Za\-z\\s,\]\+?\)\(?\:\\s\+industry\|industries\)\(?\:\\s\|</span>|,|\.|!|\?)",
        ],
        "geography": [
            r"(?:geography|geographic regions|regions|locations|geographic areas|target geography|target regions|target locations)(?:\s+is|:)\s+([A-Za-z\s,]+?)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(\[A\-Za\-z\\s,\]\+?\)\(?\:\\s\+location\|locations\|region\|regions\|geography\)\(?\:\\s\|</span>|,|\.|!|\?)",
        ],
        "revenue_range_min": [
            r"(?:revenue minimum|min revenue|minimum revenue|revenue range min)(?:\s+is|:)\s+(\d+)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(\\d\+\)\(?\:\\s\+minimum revenue\|min revenue\|revenue min\)\(?\:\\s\|</span>|,|\.|!|\?)",
        ],
        "revenue_range_max": [
            r"(?:revenue maximum|max revenue|maximum revenue|revenue range max)(?:\s+is|:)\s+(\d+)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(\\d\+\)\(?\:\\s\+maximum revenue\|max revenue\|revenue max\)\(?\:\\s\|</span>|,|\.|!|\?)",
        ],
        "ebitda_margin_min": [
            r"(?:ebitda margin minimum|min ebitda margin|minimum ebitda margin|ebitda margin min)(?:\s+is|:)\s+(\d+(?:\.\d+)?)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(\\d\+\(?\:\\\.\\d\+\)?\)\(?\:\\s\+minimum ebitda margin\|min ebitda margin\|ebitda margin min\)\(?\:\\s\|</span>|,|\.|!|\?)",
        ],
        "investment_type": [
            r"(?:investment type|type of investment|investment strategy|strategy)(?:\s+is|:)\s+(Growth Equity|Buyout|Venture Capital|Debt)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(Growth Equity\|Buyout\|Venture Capital\|Debt\)\(?\:\\s\+investment\|strategy\)\(?\:\\s\|</span>|,|\.|!|\?)",
        ],
        "stage": [
            r"(?:stage|company stage|stages of interest|company stages)(?:\s+is|:)\s+([A-Za-z\s,]+?)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(\[A\-Za\-z\\s,\]\+?\)\(?\:\\s\+stage\|stages\)\(?\:\\s\|</span>|,|\.|!|\?)",
        ],
        "yoy_growth_min": [
            r"(?:yoy growth minimum|min yoy growth|minimum yoy growth|yoy growth min)(?:\s+is|:)\s+(\d+(?:\.\d+)?)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(\\d\+\(?\:\\\.\\d\+\)?\)\(?\:\\s\+minimum yoy growth\|min yoy growth\|yoy growth min\)\(?\:\\s\|</span>|,|\.|!|\?)",
        ],
        "target_ownership": [
            r"(?:target ownership|ownership percentage|target ownership percentage)(?:\s+is|:)\s+(\d+(?:\.\d+)?)(?:\s|<span class="math-inline">\|,\|\\\.\|\!\|\\?\)",
r"\(\\d\+\(?\:\\\.\\d\+\)?\)\(?\:\\s\+ownership\|target ownership\)\(?\:\\s\|</span>|,|\.|!|\?)",
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
                    rf"(?:{field_name}(?:\s+is|:))\s+([A-Za-z0-9\s,]+?)(?:\s|$|,|\.|!|\?)"
                )
        elif "int" in field_type_str:
            if not patterns:
                patterns.append(
                    rf"(?:{field_name}(?:\s+is|:))\s+(\d+)(?:\s|$|,|\.|!|\?)"
                )
elif "float" in field_type_str:
            if not patterns:
                patterns.append(
                    rf"(?:{field_name}(?:\s+is|:))\s+(\d+(?:\.\d+)?)(?:\s|$|,|\.|!|\?)"
                )
        elif "list" in field_type_str:
            if not patterns:
                patterns.append(
                    rf"(?:{field_name}(?:\s+is|:))\s+([A-Za-z0-9\s,]+?)(?:\s|$|,|\.|!|\?)"
                )

        for pattern in patterns:
            matches = re.search(pattern, user_input, re.IGNORECASE)
            if matches:
                extracted_value = matches.group(1).strip()
                if "int" in field_type_str:
                    try:
                        extracted_value = int(extracted_value)
                    except ValueError:
                        continue
                elif "float" in field_type_str:
                    try:
                        extracted_value = float(extracted_value)
                    except ValueError:
                        continue
                elif "bool" in field_type_str:
                    extracted_value = extracted_value.lower() in ("true", "yes")
                elif "list" in field_type_str:
                    extracted_value = [
                        item.strip() for item in extracted_value.split(",")
                    ]

                extracted_values[field_name] = extracted_value
                extraction_messages[
                    field_name
                ] = f"{prefix_message} Found '{extracted_value}'"
                break

        if field_name not in extracted_values:
            extraction_messages[field_name] = f"{prefix_message} No value found"

    try:
        return model_type(**extracted_values), extraction_messages
    except Exception as e:
        error_msg = f"Validation error: {e}"
        for field in model_fields:
            if field not in extraction_messages:
                extraction_messages[field] = f"{prefix_message} {error_msg}"
        return model_type(), extraction_messages


async def main() -> None:
    model_client = AzureOpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_type="azure",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        seed=42,
        temperature=0,
    )
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. You can call tools to help user.",
        model_client=model_client,
        tools=[get_investment_info],
        reflect_on_tool_use=True,
    )

    request_model = InvestmentThesis

    message_history = []

    user_prompt = get_user_prompt(request_model)

    message_history = []

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

            if not all(
                [
                    investment_params.industry_focus,
                    investment_params.geography,
                    investment_params.revenue_range_min,
                    investment_params.revenue_range_max,
                    investment_params.ebitda_margin_min,
                    investment_params.investment_type,
                ]
            ):
                print("System: I need more information to fulfill the investment thesis.")
                # You could add further logic here to prompt for missing parameters
                # one by one, similar to the weather example.
                pass
            else:
                investment_result = await get_investment_info(investment_params)
                response = await assistant.on_messages(
                    message_history
                    + [TextMessage(content=investment_result, source="system")],
                    CancellationToken(),
                )

        except Exception as e:
            response = await assistant.on_messages(
                message_history + [TextMessage(content=f"Error: {str(e)}", source="system")],
                CancellationToken(),
            )

        message_history.append(response.chat_message)
        print("Assistant:", response.chat_message.content)


asyncio.run(main())