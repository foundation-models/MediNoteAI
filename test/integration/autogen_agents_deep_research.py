import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum
import json
from autogen_core.models import UserMessage

load_dotenv()
# model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
model_client = AzureOpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_type="azure",
    #   credential=AzureKeyCredential("api_key"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_BASE"),
    api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    seed=42,
    temperature=0,
)


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

    @validator("revenue_range_max")
    def validate_revenue_range(cls, v, values):
        if "revenue_range_min" in values and v <= values["revenue_range_min"]:
            raise ValueError("revenue_range_max must be greater than revenue_range_min")
        return v


class PitchBookFilter(BaseModel):
    """Model for individual PitchBook filter"""

    field: str = Field(..., description="Database field to filter on")
    values: Optional[List[str]] = Field(None, description="List of acceptable values")
    min: Optional[Union[int, float]] = Field(
        None, description="Minimum value for range filters"
    )
    max: Optional[Union[int, float]] = Field(
        None, description="Maximum value for range filters"
    )

    @root_validator(pre=True)
    def validate_filter_structure(cls, values):
        """Ensure either values or min/max is provided but not both"""
        has_values = values.get("values") is not None
        has_range = values.get("min") is not None or values.get("max") is not None

        if has_values and has_range:
            raise ValueError(
                "Cannot specify both 'values' and range parameters ('min'/'max')"
            )
        if not has_values and not has_range:
            raise ValueError(
                "Must specify either 'values' or range parameters ('min'/'max')"
            )
        return values


class PitchBookQuery(BaseModel):
    """Model for PitchBook database query"""

    filters: List[PitchBookFilter] = Field(..., description="List of filters to apply")


class CrunchbaseMetrics(BaseModel):
    """Model for Crunchbase metrics parameters"""

    ebitda_margin: Optional[str] = Field(
        None, description="EBITDA margin filter expression"
    )
    growth_rate: Optional[str] = Field(
        None, description="Growth rate filter expression"
    )
    valuation: Optional[str] = Field(None, description="Valuation filter expression")


class CrunchbaseQuery(BaseModel):
    """Model for Crunchbase database query"""

    categories: List[str] = Field(..., description="Industry categories to search")
    location_group_names: List[str] = Field(
        ..., description="Geographic regions to search"
    )
    revenue_range: str = Field(..., description="Revenue range expression")
    metrics: CrunchbaseMetrics = Field(
        default_factory=CrunchbaseMetrics, description="Performance metrics filters"
    )
    company_stage: Optional[List[str]] = Field(
        None, description="Company stage filters"
    )


class DatabaseQueries(BaseModel):
    """Model for all database queries"""

    pitchbook: PitchBookQuery
    crunchbase: CrunchbaseQuery


class CompanyData(BaseModel):
    """Raw company data from data source"""

    company_name: str = Field(..., description="Name of the company")
    industry: str = Field(..., description="Primary industry")
    revenue: int = Field(..., description="Annual revenue in USD", ge=0)
    ebitda_margin: float = Field(
        ..., description="EBITDA margin as decimal", ge=0, le=1
    )
    location: str = Field(..., description="Company headquarters location")
    yoy_growth: Optional[float] = Field(None, description="Year-over-year growth rate")
    founded_year: Optional[int] = Field(
        None, description="Year the company was founded"
    )
    employees: Optional[int] = Field(None, description="Number of employees")

    class Config:
        extra = "allow"  # Allow additional fields that might come from the data source


class StandardizedCompany(BaseModel):
    """Standardized company data model"""

    name: str = Field(..., description="Company name")
    industry: str = Field(..., description="Primary industry category")
    revenue_usd: int = Field(..., description="Annual revenue in USD", ge=0)
    ebitda_margin: float = Field(
        ..., description="EBITDA margin as decimal", ge=0, le=1
    )
    country: str = Field(..., description="Country of headquarters")
    yoy_growth: Optional[float] = Field(None, description="Year-over-year growth rate")
    employees: Optional[int] = Field(None, description="Number of employees")
    funding_rounds: Optional[int] = Field(None, description="Number of funding rounds")
    last_funding_date: Optional[str] = Field(
        None, description="Date of last funding round"
    )

    class Config:
        extra = "allow"  # Allow additional fields for flexibility


class ReportFormat(str, Enum):
    """Enumeration of supported report formats"""

    SUMMARY = "summary"
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PDF = "pdf"


class ThesisParameterHandler:
    """Handles parameter collection and validation for investment thesis with LLM assistance."""

    def __init__(self, model_client):
        """Initialize with an Autogen client for LLM interaction."""
        self.model_client = model_client
        self.valid_investment_types = [
            "Growth Equity",
            "Buyout",
            "Venture Capital",
            "Debt",
        ]

    async def get_investment_thesis(
        self, initial_params: Union[str, Dict[str, Any]] = None
    ) -> InvestmentThesis:
        """
        Process investment thesis parameters with LLM assistance.

        Args:
            initial_params: Initial parameters as JSON string or dictionary

        Returns:
            Complete and validated InvestmentThesis object
        """
        # Parse initial parameters if provided as JSON string
        if isinstance(initial_params, str):
            try:
                params_dict = json.loads(initial_params)
            except json.JSONDecodeError:
                params_dict = {}
        elif isinstance(initial_params, dict):
            params_dict = initial_params
        else:
            params_dict = {}

        # Attempt to create thesis object and identify missing/invalid fields
        missing_fields, invalid_fields = self._validate_thesis_params(params_dict)

        # If we have everything valid, return the complete thesis
        if not missing_fields and not invalid_fields:
            return InvestmentThesis(**params_dict)

        # Otherwise, use LLM to get missing information
        complete_params = await self._clarify_parameters(
            params_dict, missing_fields, invalid_fields
        )
        return InvestmentThesis(**complete_params)

    def _validate_thesis_params(self, params: Dict[str, Any]) -> tuple:
        """Validate parameters and return missing and invalid fields."""
        required_fields = {
            "industry_focus": "list of target industries",
            "geography": "list of target geographic regions",
            "revenue_range_min": "minimum revenue in USD",
            "revenue_range_max": "maximum revenue in USD",
            "ebitda_margin_min": "minimum EBITDA margin as decimal",
            "investment_type": f"investment type (one of: {', '.join(self.valid_investment_types)})",
        }

        missing_fields = {k: v for k, v in required_fields.items() if k not in params}

        # Check for invalid values in provided parameters
        invalid_fields = {}

        # Validate revenue range if both min and max are provided
        if "revenue_range_min" in params and "revenue_range_max" in params:
            if params["revenue_range_max"] <= params["revenue_range_min"]:
                invalid_fields["revenue_range"] = (
                    "Maximum revenue must be greater than minimum revenue"
                )

        # Validate EBITDA margin
        if "ebitda_margin_min" in params:
            try:
                margin = float(params["ebitda_margin_min"])
                if not 0 <= margin <= 1:
                    invalid_fields["ebitda_margin_min"] = (
                        "EBITDA margin must be between 0 and 1"
                    )
            except (ValueError, TypeError):
                invalid_fields["ebitda_margin_min"] = (
                    "EBITDA margin must be a decimal number"
                )

        # Validate investment type
        if (
            "investment_type" in params
            and params["investment_type"] not in self.valid_investment_types
        ):
            invalid_fields["investment_type"] = (
                f"Investment type must be one of: {', '.join(self.valid_investment_types)}"
            )

        # Validate optional fields if present
        if "yoy_growth_min" in params:
            try:
                growth = float(params["yoy_growth_min"])
                if growth < 0:
                    invalid_fields["yoy_growth_min"] = "Growth rate cannot be negative"
            except (ValueError, TypeError):
                invalid_fields["yoy_growth_min"] = (
                    "Growth rate must be a decimal number"
                )

        if "target_ownership" in params:
            try:
                ownership = float(params["target_ownership"])
                if not 0 <= ownership <= 1:
                    invalid_fields["target_ownership"] = (
                        "Ownership percentage must be between 0 and 1"
                    )
            except (ValueError, TypeError):
                invalid_fields["target_ownership"] = (
                    "Ownership percentage must be a decimal number"
                )

        return missing_fields, invalid_fields

    async def _clarify_parameters(
        self,
        current_params: Dict[str, Any],
        missing_fields: Dict[str, str],
        invalid_fields: Dict[str, str],
    ) -> Dict[str, Any]:
        """Use LLM to clarify missing or invalid parameters."""
        # Create a copy of the current parameters
        updated_params = current_params.copy()

        # Construct the prompt for the LLM
        prompt = self._create_clarification_prompt(
            updated_params, missing_fields, invalid_fields
        )

        # Get LLM response
        response = await self.model_client.create(
            [UserMessage(content=prompt, source="user")]
        )

        # Extract the content from the CreateResult object
        # In Autogen 0.4.8, we need to access the content/message from the response object
        response_content = self._extract_response_content(response)
        
        # Process the LLM response
        parameter_updates = self._parse_llm_response(response_content)
        
        # Update parameters with LLM suggestions
        updated_params.update(parameter_updates)

        # Check if we still have missing or invalid parameters
        missing_fields, invalid_fields = self._validate_thesis_params(updated_params)

        # If we still have issues, recurse
        if missing_fields or invalid_fields:
            # Limit recursion depth to avoid infinite loops
            return await self._clarify_parameters(
                updated_params, missing_fields, invalid_fields
            )

        return updated_params

    def _extract_response_content(self, response):
        """Extract text content from Autogen response object."""
        # Handle different response formats in Autogen 0.4.8
        if hasattr(response, 'content'):
            return response.content  # Direct content attribute
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content  # Message object with content
        elif hasattr(response, 'text'):
            return response.text  # Some clients might return text attribute
        elif hasattr(response, 'choices') and len(response.choices) > 0:
            # OpenAI-style response format
            if hasattr(response.choices[0], 'message'):
                return response.choices[0].message.content
            elif hasattr(response.choices[0], 'text'):
                return response.choices[0].text
        
        # If we can't extract in any standard way, convert to string
        return str(response)
    
    def _create_clarification_prompt(
        self,
        current_params: Dict[str, Any],
        missing_fields: Dict[str, str],
        invalid_fields: Dict[str, str],
    ) -> str:
        """Create a prompt for the LLM to clarify parameters."""
        prompt = "I'm helping create an investment thesis and need your assistance to complete or fix some parameters.\n\n"

        # Add current parameters
        if current_params:
            prompt += "Here are the parameters we already have:\n"
            for key, value in current_params.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"

        # Add missing fields
        if missing_fields:
            prompt += (
                "We need to provide values for these missing required parameters:\n"
            )
            for key, description in missing_fields.items():
                prompt += f"- {key}: {description}\n"
            prompt += "\n"

        # Add invalid fields
        if invalid_fields:
            prompt += "These parameters need to be fixed:\n"
            for key, issue in invalid_fields.items():
                prompt += f"- {key}: {issue}\n"
            prompt += "\n"

        prompt += """
Please provide appropriate values for the missing or invalid parameters. Return your suggestions in valid JSON format 
that I can directly parse. Include ONLY the parameters that need to be added or fixed. Do not repeat parameters that 
are already valid.

For example:
{
  "parameter_name": suggested_value
}

If you need to ask the user for clarification on any parameter, include a field called "user_query" with a list of 
questions that should be asked.

Example:
{
  "parameter_name": suggested_value,
  "user_query": ["What is the minimum revenue you're targeting?", "What geographic regions are you interested in?"]
}
"""
        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response and extract parameter suggestions."""
        # Extract JSON from response
        try:
            # Try to find JSON block in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                suggestions = json.loads(json_str)
            else:
                # No JSON found, return empty dict
                return {}

            # If LLM suggests asking user for more info
            if "user_query" in suggestions:
                user_queries = suggestions.pop("user_query")
                # Display questions to user and get responses
                for query in user_queries:
                    print(f"\n{query}")
                    user_input = input("Your response: ")

                    # Try to interpret the user's response based on the query
                    if "industry" in query.lower() or "sector" in query.lower():
                        suggestions["industry_focus"] = [
                            ind.strip() for ind in user_input.split(",")
                        ]
                    elif "geographic" in query.lower() or "region" in query.lower():
                        suggestions["geography"] = [
                            reg.strip() for reg in user_input.split(",")
                        ]
                    elif "revenue" in query.lower() and "minimum" in query.lower():
                        try:
                            suggestions["revenue_range_min"] = int(user_input)
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                    elif "revenue" in query.lower() and "maximum" in query.lower():
                        try:
                            suggestions["revenue_range_max"] = int(user_input)
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                    elif "ebitda" in query.lower() or "margin" in query.lower():
                        try:
                            suggestions["ebitda_margin_min"] = float(user_input)
                        except ValueError:
                            print("Invalid input. Please enter a decimal.")
                    elif "investment type" in query.lower():
                        suggestions["investment_type"] = user_input
                    elif "stage" in query.lower():
                        suggestions["stage"] = (
                            [s.strip() for s in user_input.split(",")]
                            if user_input
                            else None
                        )
                    elif "growth" in query.lower():
                        try:
                            suggestions["yoy_growth_min"] = (
                                float(user_input) if user_input else None
                            )
                        except ValueError:
                            print("Invalid input. Please enter a decimal.")
                    elif "ownership" in query.lower():
                        try:
                            suggestions["target_ownership"] = (
                                float(user_input) if user_input else None
                            )
                        except ValueError:
                            print("Invalid input. Please enter a decimal.")
                    else:
                        # General input - store as string for now
                        # The validation will catch any issues in the next iteration
                        suggestions["user_response"] = user_input

            return suggestions

        except json.JSONDecodeError:
            print("Could not parse LLM response. Please try again.")
            return {}


# Example usage in your Autogen setup
async def get_investment_thesis_input(initial_json: str = None):
    # Initialize the parameter handler with your model client
    handler = ThesisParameterHandler(model_client)

    # Get complete thesis with LLM assistance
    thesis = await handler.get_investment_thesis(initial_json)

    # Return the validated thesis
    return thesis


def get_investment_thesis_input2() -> InvestmentThesis:
    """Get investment thesis parameters from user and return as structured InvestmentThesis object"""
    print("Please provide details for your investment thesis:")

    # Collect industry focus
    industry_input = input("Enter target industries (comma-separated): ")
    industry_focus = [industry.strip() for industry in industry_input.split(",")]

    # Collect geography
    geography_input = input("Enter target geographic regions (comma-separated): ")
    geography = [region.strip() for region in geography_input.split(",")]

    # Collect revenue range
    while True:
        try:
            revenue_min = int(input("Enter minimum revenue in USD: "))
            revenue_max = int(input("Enter maximum revenue in USD: "))
            if revenue_max <= revenue_min:
                print(
                    "Maximum revenue must be greater than minimum revenue. Please try again."
                )
                continue
            break
        except ValueError:
            print("Please enter valid numbers for revenue range.")

    # Collect EBITDA margin
    while True:
        try:
            ebitda_margin = float(
                input("Enter minimum EBITDA margin (as decimal, e.g., 0.20 for 20%): ")
            )
            if not 0 <= ebitda_margin <= 1:
                print("EBITDA margin must be between 0 and 1. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid decimal for EBITDA margin.")

    # Collect investment type
    valid_investment_types = ["Growth Equity", "Buyout", "Venture Capital", "Debt"]
    while True:
        print(f"Valid investment types: {', '.join(valid_investment_types)}")
        investment_type = input("Enter investment type: ")
        if investment_type in valid_investment_types:
            break
        print(
            f"Invalid investment type. Please select from: {', '.join(valid_investment_types)}"
        )

    # Optional parameters
    stage_input = input(
        "Enter company stages of interest (comma-separated, or press Enter to skip): "
    )
    stage = [s.strip() for s in stage_input.split(",")] if stage_input.strip() else None

    yoy_growth_min = None
    growth_input = input(
        "Enter minimum year-over-year growth rate (as decimal, or press Enter to skip): "
    )
    if growth_input.strip():
        try:
            yoy_growth_min = float(growth_input)
            if yoy_growth_min < 0:
                print("Growth rate cannot be negative. Setting to None.")
                yoy_growth_min = None
        except ValueError:
            print("Invalid growth rate. Setting to None.")

    target_ownership = None
    ownership_input = input(
        "Enter target ownership percentage (as decimal, or press Enter to skip): "
    )
    if ownership_input.strip():
        try:
            target_ownership = float(ownership_input)
            if not 0 <= target_ownership <= 1:
                print("Ownership percentage must be between 0 and 1. Setting to None.")
                target_ownership = None
        except ValueError:
            print("Invalid ownership percentage. Setting to None.")

    # Construct and validate thesis data
    thesis_data = {
        "industry_focus": industry_focus,
        "geography": geography,
        "revenue_range_min": revenue_min,
        "revenue_range_max": revenue_max,
        "ebitda_margin_min": ebitda_margin,
        "investment_type": investment_type,
    }

    # Add optional parameters if provided
    if stage is not None:
        thesis_data["stage"] = stage
    if yoy_growth_min is not None:
        thesis_data["yoy_growth_min"] = yoy_growth_min
    if target_ownership is not None:
        thesis_data["target_ownership"] = target_ownership

    # Convert dict to InvestmentThesis object with validation
    return InvestmentThesis(**thesis_data)


def generate_search_queries(investment_thesis: InvestmentThesis) -> DatabaseQueries:
    """Generate database search queries based on investment thesis parameters"""
    print(
        "Function Stub: generate_search_queries - Generating dummy queries based on thesis."
    )

    # Build PitchBook filters
    pitchbook_filters = [
        PitchBookFilter(
            field="industryGroup",
            values=(
                ["Software as a Service (SaaS)"]
                if "SaaS" in investment_thesis.industry_focus
                else investment_thesis.industry_focus
            ),
        ),
        PitchBookFilter(
            field="headquartersLocation", values=investment_thesis.geography
        ),
        PitchBookFilter(
            field="revenue",
            min=investment_thesis.revenue_range_min,
            max=investment_thesis.revenue_range_max,
        ),
        PitchBookFilter(field="ebitdaMargin", min=investment_thesis.ebitda_margin_min),
    ]

    # Add optional PitchBook filters
    if investment_thesis.yoy_growth_min is not None:
        pitchbook_filters.append(
            PitchBookFilter(field="yoyGrowthRate", min=investment_thesis.yoy_growth_min)
        )

    if investment_thesis.stage is not None:
        pitchbook_filters.append(
            PitchBookFilter(field="companyStage", values=investment_thesis.stage)
        )

    # Build Crunchbase metrics
    crunchbase_metrics = CrunchbaseMetrics(
        ebitda_margin=f">{investment_thesis.ebitda_margin_min}"
    )

    if investment_thesis.yoy_growth_min is not None:
        crunchbase_metrics.growth_rate = f">{investment_thesis.yoy_growth_min}"

    # Create complete query structure
    queries = DatabaseQueries(
        pitchbook=PitchBookQuery(filters=pitchbook_filters),
        crunchbase=CrunchbaseQuery(
            categories=investment_thesis.industry_focus,
            location_group_names=investment_thesis.geography,
            revenue_range=f"{investment_thesis.revenue_range_min}-{investment_thesis.revenue_range_max}",
            metrics=crunchbase_metrics,
            company_stage=investment_thesis.stage,
        ),
    )

    return queries


def fetch_company_data_pitchbook(pitchbook_query: PitchBookQuery) -> List[CompanyData]:
    """Fetch company data from PitchBook based on query parameters"""
    print(
        "Function Stub: fetch_company_data_pitchbook - Fetching dummy data from PitchBook API."
    )

    # Example dummy data - would come from API in real implementation
    dummy_data = [
        {
            "company_name": "SaaS Company A",
            "industry": "SaaS",
            "revenue": 25000000,
            "ebitda_margin": 0.25,
            "location": "USA",
            "yoy_growth": 0.32,
            "founded_year": 2012,
            "employees": 180,
        },
        {
            "company_name": "SaaS Company B",
            "industry": "SaaS",
            "revenue": 40000000,
            "ebitda_margin": 0.30,
            "location": "Canada",
            "yoy_growth": 0.28,
            "founded_year": 2009,
            "employees": 250,
        },
    ]

    # Convert raw data to validated CompanyData objects
    return [CompanyData(**company) for company in dummy_data]


def standardize_and_clean_data(
    raw_company_data_list: List[CompanyData],
) -> List[StandardizedCompany]:
    """Standardize and clean raw company data into consistent format"""
    print(
        "Function Stub: standardize_and_clean_data - Standardizing and cleaning dummy data."
    )

    standardized_data = []
    for company_data in raw_company_data_list:
        standardized_company = StandardizedCompany(
            name=company_data.company_name,
            industry=company_data.industry,
            revenue_usd=company_data.revenue,
            ebitda_margin=company_data.ebitda_margin,
            country=company_data.location,
            yoy_growth=company_data.yoy_growth,
            employees=company_data.employees,
        )
        standardized_data.append(standardized_company)

    return standardized_data


def initial_screening_filter(
    company_data_list: List[StandardizedCompany], investment_thesis: InvestmentThesis
) -> List[StandardizedCompany]:
    """Filter companies based on investment thesis criteria"""
    print(
        "Function Stub: initial_screening_filter - Filtering companies based on investment thesis."
    )

    filtered_companies = []
    for company in company_data_list:
        # Check if company meets all basic criteria
        if (
            company.industry in investment_thesis.industry_focus
            and company.country in investment_thesis.geography
            and investment_thesis.revenue_range_min
            <= company.revenue_usd
            <= investment_thesis.revenue_range_max
            and company.ebitda_margin >= investment_thesis.ebitda_margin_min
        ):
            # Check optional criteria if specified
            meets_optional_criteria = True

            if (
                investment_thesis.yoy_growth_min is not None
                and company.yoy_growth is not None
            ):
                if company.yoy_growth < investment_thesis.yoy_growth_min:
                    meets_optional_criteria = False

            if meets_optional_criteria:
                filtered_companies.append(company)

    return filtered_companies


def generate_deal_sourcing_report(
    ranked_companies: List[StandardizedCompany],
    report_format: ReportFormat = ReportFormat.SUMMARY,
) -> str:
    """Generate a formatted report of potential investment targets"""
    print(
        f"Function Stub: generate_deal_sourcing_report - Generating dummy report in {report_format} format."
    )

    if report_format == ReportFormat.SUMMARY:
        summary_text = "Top Screened Companies (Initial Screening):\n"
        # Top 3 for summary
        for i, company in enumerate(ranked_companies[:3], 1):
            summary_text += (
                f"{i}. {company.name}\n"
                f"   Industry: {company.industry}\n"
                f"   Revenue: ${company.revenue_usd/1000000:.2f}M\n"
                f"   EBITDA Margin: {company.ebitda_margin:.2%}\n"
                f"   Growth Rate: {company.yoy_growth:.2%} YoY\n"
                f"   Location: {company.country}\n\n"
            )
        return summary_text

    elif report_format == ReportFormat.JSON:
        # Convert Pydantic models to dictionaries for JSON serialization
        companies_dict = [company.dict() for company in ranked_companies]
        return json.dumps(companies_dict, indent=2)

    else:
        return f"Report in {report_format} format would be generated here."


# Create the Investment Thesis Assistant agent
Investment_thesis_assistant = AssistantAgent(
    "Investment_thesis_assistant",
    model_client=model_client,
    tools=[get_investment_thesis_input],
    system_message="You are an AI assistant that helps users gather necessary parameters for creating an investment thesis, focusing on industry, geography, revenue range, and other investment factors.",
)

# Create the Search Queries Generator Assistant agent
Search_queries_generator = AssistantAgent(
    "Search_queries_generator",
    model_client=model_client,
    tools=[generate_search_queries],
    system_message="You are an AI assistant that translates investment thesis parameters into search queries for financial databases, ensuring they are optimized for the task.",
)

# Create the Company Data Fetcher from PitchBook Assistant agent
Company_data_fetcher_pitchbook = AssistantAgent(
    "Company_data_fetcher_pitchbook",
    model_client=model_client,
    tools=[fetch_company_data_pitchbook],
    system_message="You are an AI assistant that fetches company data from PitchBook API based on provided search queries and investment criteria.",
)

# Create the Data Standardizer and Cleaner Assistant agent
Data_standardizer_cleaner = AssistantAgent(
    "Data_standardizer_cleaner",
    model_client=model_client,
    tools=[standardize_and_clean_data],
    system_message="You are an AI assistant that standardizes and cleans raw company data into a unified and usable format, ensuring consistency across datasets.",
)

# Create the Initial Screening Filter Assistant agent
Initial_screening_filter = AssistantAgent(
    "Initial_screening_filter",
    model_client=model_client,
    tools=[initial_screening_filter],
    system_message="You are an AI assistant that applies screening filters to company data based on the investment thesis, ensuring that only relevant companies are considered.",
)

# Create the Deal Sourcing Report Generator Assistant agent
Deal_sourcing_report_generator = AssistantAgent(
    "Deal_sourcing_report_generator",
    model_client=model_client,
    tools=[generate_deal_sourcing_report],
    system_message="You are an AI assistant that generates detailed reports summarizing the deal sourcing process, including ranked companies in various formats.",
)


text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

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
# Define the task string outside the function
PE_TASK = """You are a Private Equity Deal Sourcing Associate AI.
Your goal is to find potential company acquisition targets based on an investment thesis.
Follow this step-by-step process:
1. Collect investment thesis parameters from the user by asking relevant questions about industries, regions, revenue, EBITDA, investment type, and other criteria.
2. Generate appropriate search queries for financial databases based on the thesis.
3. Fetch company data from sources like PitchBook.
4. Standardize and clean the collected data.
5. Screen companies based on the investment criteria.
6. Generate a comprehensive summary report of potential targets.
Ask clarifying questions if needed. Provide explanations for each step. Present results clearly. Conclude by saying 'TERMINATE' when complete.

Your goal is to find potential company acquisition targets based on an investment thesis.
"""

PE_TASK = (
    """create an investment thesis, use it for a search and analyze potential deal"""
)


# Update your main function
async def main():

    # Pass the completed thesis to the team execution
    await Console(team.run_stream(task=PE_TASK))


# # Define the main asynchronous function
# async def main():
#     await Console(
#         team.run_stream(
#             #             task="""You are a Private Equity Deal Sourcing Associate AI.
#             #         Your goal is to find potential company acquisition targets based on an investment thesis.
#             #         You will take an investment thesis as input, generate search queries, fetch company data,
#             #         screen companies based on criteria, and provide a summary report.
#             #         You can use functions to get data and perform actions.
#             #         Conclude by generating a summary report and saying "TERMINATE" when the deal sourcing process is complete for the initial screening phase.
#             #         Remember to ask clarifying questions to the user proxy if needed to fully understand the request.

#             #         create an investment thesis, use it for a search and analyze potential deal
#             # """
#                         task="""You are a Private Equity Deal Sourcing Associate AI.

# Your goal is to find potential company acquisition targets based on an investment thesis.

# Follow this step-by-step process:
# 1. Collect investment thesis parameters from the user by asking relevant questions about:
#    - Target industries
#    - Geographic regions
#    - Revenue range
#    - EBITDA margin requirements
#    - Investment type (Growth Equity, Buyout, etc.)
#    - And other relevant criteria

# 2. Generate appropriate search queries for financial databases based on the thesis
# 3. Fetch company data from sources like PitchBook
# 4. Standardize and clean the collected data
# 5. Screen companies based on the investment criteria
# 6. Generate a comprehensive summary report of potential targets

# Make sure to:
# - Ask clarifying questions if any information is unclear
# - Provide explanations for each step of the process
# - Present the final results in a clear, actionable format

# Conclude by generating a summary report and saying "TERMINATE" when the deal sourcing process is complete for the initial screening phase.
# """
#             # task="create an investment thesis, use it for a search and analyze potential deal"
#         )
#     )  # Stream the messages to the console.


# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())

# """
# Start the deal sourcing process.  search for that investment thesis
# \n I need you to find potential acquisition targets. \n First, get the investment thesis parameters from me. \n Then,
# """
