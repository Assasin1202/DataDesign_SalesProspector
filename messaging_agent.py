import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
# Reuse model helpers from grading_agent (support both package and standalone execution)
try:
    from .grading_agent import ModelType, AzureModelType, get_model  # type: ignore
except ImportError:  # when run as standalone script
    from grading_agent import ModelType, AzureModelType, get_model  # type: ignore


class PersonalizedMessage(BaseModel):
    """Structured model for the outreach message."""

    channel: str = Field(description="Recommended outreach channel, e.g., LinkedIn DM, Email.")
    subject: str = Field(description="Subject line for an email or opening hook for LinkedIn.")
    message: str = Field(description="Personalized message body, max ~120 words.")


class CompanyProfile(BaseModel):
    """Configuration model for customizing the messaging agent with company-specific information."""
    
    company_name: str = Field(default="Data Design Oy", description="Name of the company")
    website: str = Field(default="datadesign.fi", description="Company website URL")
    company_description: str = Field(
        default="provides seasoned specialists for data and AI projects including Project Manager, Data Analyst, AI Strategist, GenAI & ML Engineer, and Data Governance Architect",
        description="Brief description of what the company does and key services"
    )
    
    @property
    def messaging_tone(self) -> str:
        """Fixed messaging tone for consistency."""
        return "professional yet friendly and consultative"
    
    @property
    def message_length_limit(self) -> int:
        """Fixed message length limit."""
        return 120


class MessagingAgent:
    """Agent that crafts a personalized conversation-starter based on full prospect data.

    Optionally, a preferred outreach `channel` ("Email" or "LinkedIn") can be provided. If omitted, the agent decides automatically.
    The agent can be customized with company-specific information via the CompanyProfile, defaulting to Data Design Oy.
    """

    def __init__(
        self,
        model_type: str = ModelType.AZURE,
        azure_model: str = AzureModelType.GPT4_1_MINI,
        company_profile: Optional[CompanyProfile] = None,
    ) -> None:
        
        # Use provided company profile or default to Data Design Oy
        self.company_profile = company_profile or CompanyProfile()
        
        # Build dynamic system prompt
        system_prompt = f"""
            {self.company_profile.company_name} Messaging Agent

            You are a top-tier B2B sales development representative for {self.company_profile.company_name}. Your task is to craft a highly personalized, first-touch outreach using the provided JSON (which contains enriched_lead, evaluation_result, and grading_result).

            Core value proposition (include where relevant):
            • {self.company_profile.company_name} {self.company_profile.company_description}

            Messaging framework / guidelines:
            1. Address the prospect by name and mention their company.
            2. Start with a relevant trigger or recent news (if available) in **one** engaging sentence.
            3. Explain, in 1-2 sentences, how your company's services can help with the opportunity or challenge identified in key_insights.
            4. Keep the tone {self.company_profile.messaging_tone}, and limit the body to **≤{self.company_profile.message_length_limit} words**.
            5. Direct the reader to our website ({self.company_profile.website}) for further information.
            6. End with a clear, low-friction call-to-action (e.g., a quick chat about their specific area).
            7. Respect the user's preferred outreach channel ({{preferred_channel}}) if provided; otherwise choose Email or LinkedIn based on key_insights.

            Output must strictly follow the PersonalizedMessage JSON schema.
            """

        model, provider = get_model(model_type, azure_model)
        self.agent = Agent(
            model=model,
            tools=[],
            output_type=PersonalizedMessage,
            system_prompt=system_prompt,
            verbose=True,
        )

    def run(
        self,
        combined_json: dict,
        preferred_channel: str | None = None,
        user_preferences: str | None = None,
    ) -> PersonalizedMessage | None:
        channel_instruction = (
            f"The user has requested the outreach message to be delivered via **{preferred_channel}**. Use this channel." if preferred_channel else
            "No specific channel requested by the user; choose the most appropriate channel."
        )

        preferences_instruction = (
            f"The user has specified additional preferences for the message: '{user_preferences}'. Incorporate these preferences accordingly." if user_preferences else
            "No additional user preferences provided."
        )

        prompt = (
            "Create a personalized outreach message using the following data. "
            + channel_instruction
            + " "
            + preferences_instruction
            + "\n\nDATA:\n"
            + json.dumps(combined_json, indent=2)
            + "\n"
        )
        try:
            result = self.agent.run_sync(prompt)
            return result.output  # type: ignore[attr-defined]
        except Exception as e:
            print(f"❌ Messaging agent failed: {e}")
            return None


# Convenience function to create a custom company profile
def create_custom_company_profile(
    company_info: str,
    website: str = "yourcompany.com"
) -> CompanyProfile:
    """Create a custom company profile for the messaging agent.
    
    Args:
        company_info: Company name and description in format "Company Name - what they do"
                     e.g., "TechSolutions Inc - delivers cutting-edge software development services"
        website: Company website (without https://)
    
    Returns:
        CompanyProfile: Configured profile for MessagingAgent
    """
    # Parse company name and description
    if " - " in company_info:
        company_name, description = company_info.split(" - ", 1)
        company_name = company_name.strip()
        full_description = f"{description.strip()}"
    else:
        # If no separator, treat the whole thing as company name
        company_name = company_info.strip()
        full_description = "provides professional services and solutions"
    
    return CompanyProfile(
        company_name=company_name,
        website=website,
        company_description=full_description
    )


if __name__ == "__main__":
    INPUT_FILE = Path("sample_output_with_grading.json")
    if not INPUT_FILE.exists():
        print("⚠️  'sample_output_with_grading.json' not found. Falling back to 'sample_output.json'.")
        INPUT_FILE = Path("sample_output.json")
        if not INPUT_FILE.exists():
            print("❌ No input JSON available.")
            raise SystemExit(1)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Example 1: Default Data Design Oy agent
    print("=== TESTING DEFAULT DATA DESIGN OY AGENT ===")
    agent = MessagingAgent()
    msg = agent.run(data)
    
    if msg:
        print("\n=========== DEFAULT PERSONALIZED MESSAGE ===========\n")
        print(json.dumps(msg.model_dump(), indent=4))
    else:
        print("❌ No message produced.")
    
    # Example 2: Custom company agent
    print("\n=== TESTING CUSTOM COMPANY AGENT ===")
    custom_profile = create_custom_company_profile(
        company_info="TechSolutions Inc - delivers cutting-edge software development and cloud consulting services",
        website="techsolutions.com"
    )
    
    custom_agent = MessagingAgent(company_profile=custom_profile)
    custom_msg = custom_agent.run(data)
    
    if custom_msg:
        print("\n=========== CUSTOM PERSONALIZED MESSAGE ===========\n")
        print(json.dumps(custom_msg.model_dump(), indent=4))
    else:
        print("❌ No custom message produced.") 