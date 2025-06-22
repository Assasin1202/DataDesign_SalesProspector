import os
import json
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider
from mistralai import Mistral
import anthropic

# Load environment variables
load_dotenv()

# ------------------- 1. MODEL SELECTION -------------------
class ModelType:
    AZURE = "azure"
    MISTRAL = "mistral"
    CLAUDE = "claude"

class AzureModelType:
    GPT4_1 = "gpt-4.1"
    GPT4_1_MINI = "gpt-4.1-mini"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"

def get_model(model_type: str, azure_model: str = AzureModelType.GPT4_1_MINI):
    """Return a (model_name_or_object, provider) tuple based on the requested model type."""
    if model_type == ModelType.AZURE:
        provider = AzureProvider(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        return OpenAIModel(azure_model, provider=provider), provider
    elif model_type == ModelType.MISTRAL:
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        return "mistral-small-latest", client
    elif model_type == ModelType.CLAUDE:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return "claude-sonnet-4-20250514", client
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# ------------------- 2. DATA MODEL -------------------
class ProspectGrading(BaseModel):
    """A structured model capturing the grading details for a prospect."""

    prospect_summary: str = Field(
        description="A single-line summary in the format 'PROSPECT: [Company] | [Industry] | ICP Fit: [Strong/Good/Moderate/Weak]'"
    )

    class ScoreEntry(BaseModel):
        """Single scoring category chosen by the user."""

        category: str = Field(description="Name of the assessment category (e.g., 'AI Maturity', 'Data & AI Strategy')")
        score: int = Field(description="Score for the category (1-5)", ge=1, le=5)
        reason: str = Field(description="One-sentence justification for the score")

    scores: List[ScoreEntry] = Field(
        description="List of all scored categories with their reasons. The categories themselves are user-defined."
    )

    total_score: int | None = Field(
        default=None,
        description="Optional total score across all categories (auto-calculated or provided by the agent).",
        ge=0,
    )

    priority_service: str = Field(
        description="Which category should be prioritized (highest opportunity / lowest maturity)."
    )

    key_insights: List[str] = Field(
        description="A list of key insights highlighting opportunity, challenge, and engagement readiness."
    )

# ------------------- 3. AGENT DEFINITION -------------------
class GradingAgent:
    """An agent that grades prospects based on maturity and ICP fit."""

    def __init__(
        self,
        model_type: str = ModelType.AZURE,
        azure_model: str = AzureModelType.GPT4_1_MINI,
        custom_prompt: str | None = None,
    ) -> None:
        """Create a new GradingAgent.

        Parameters
        ----------
        model_type : str, optional
            Which LLM provider to use (azure, mistral, claude).
        azure_model : str, optional
            Deployment name for Azure OpenAI (only used when model_type=="azure").
        custom_prompt : str | None, optional
            If provided, this prompt will be used **instead** of the default Data Design
            grading prompt. This allows users to inject their own ICP description and
            grading factors directly from the UI. When *None* (default) the original
            hard-coded Data Design prompt is applied.
        """

        # Use a custom prompt when supplied, otherwise fall back to the default
        system_prompt = custom_prompt or (
            """
            Data Design Oy Prospect Grading Agent
            You are a prospect scoring agent for Data Design Oy. Score prospects (1-5) for each service based on maturity levels and ICP fit.

            Ideal Client Profile
            Target: Medium-large enterprises in Manufacturing, Energy, Finance, Media, Retail
            Profile: Complex data environments, digital transformation underway, data underutilized, seeking AI/data value
            Decision Makers: C-suite, Head of Data/AI/Analytics, Digital Transformation Leads

            Maturity Assessment (Rate 1-5)
            AI Maturity:
            1=Awareness (minimal use, no strategy)
            2=Active (pilot projects, inconsistent)
            3=Operational (integrated processes, governance)
            4=Systematic (competitive advantage, core operations)
            5=Transformational (fully integrated, shapes strategy)

            Data Utilization:
            1=Immature (no data value understanding, silos)
            2=Developing (some usage, not systematic)
            3=Data Driven (solves business problems)
            4=Analytical (automated solutions, advanced analysis)
            5=Data Mature (new business models from data)

            Data Governance:
            1=Beginner (ad hoc, no processes)
            2=Explorer (basic tools, emerging roles)
            3=Intermediate (formal processes, governance team)
            4=Advanced (established framework, advanced tools)
            5=Champion (strategic asset, optimized processes)

            Service Scoring (1-5 each)
            Data & AI Strategy: High scores for Level 1-2 maturity + strong ICP fit + no clear roadmap + executive buy-in
            Data & AI Governance: High scores for Level 1-2 governance + regulatory pressure + data scattered + risk exposure
            Data & AI Architecture: High scores for Level 1-3 maturity + legacy systems + scaling challenges + transformation budget
            Data & AI Implementation: High scores for Level 2-3 maturity + defined use cases + data infrastructure ready + technical team available

            When responding, populate the ProspectGrading JSON schema EXACTLY. Provide concise reasons (one sentence each).
            For the `key_insights` field, provide 6 bullet-style sentences that can directly feed a personalized outreach message. Each bullet SHOULD:
            • Highlight the main opportunity for Data & AI value.
            • State the primary challenge or pain point the prospect is facing.
            • Indicate engagement readiness (High/Medium/Low) with a short justification (recent budget, leadership mandate, etc.).
            • Suggest a compelling value proposition angle (e.g., reduce operational cost by X%, accelerate analytics, comply with regulation).
            • Mention any recent trigger event or news item to reference in the first sentence of an outreach email (funding, leadership change, new strategy, etc.).
            • Recommend the best outreach channel & tone (e.g., LinkedIn soft intro vs. direct email with ROI numbers).

            Keep bullets succinct (max 25 words each) but information-dense so that a downstream messaging agent can easily convert them into a tailored email or LinkedIn message.
            """
        )

        model, provider = get_model(model_type, azure_model)
        self.agent = Agent(
            model=model,
            tools=[],  # No external tools required; assessment is based on provided context
            output_type=ProspectGrading,
            system_prompt=system_prompt,
            verbose=True,
        )

    def run(self, enriched_lead, evaluation_result=None):
        """Generate a grading assessment for the given prospect.

        Accepts either Pydantic BaseModel instances or plain dicts for `enriched_lead` and `evaluation_result`."""

        print("\n📊 Starting prospect grading...")

        # Normalize inputs to dicts
        def to_dict(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump_json"):
                return json.loads(obj.model_dump_json())
            raise TypeError("Unsupported input type for grading agent")

        context_payload = {
            "enriched_lead": to_dict(enriched_lead),
            "evaluation_result": to_dict(evaluation_result),
        }

        prompt = (
            "Using the following prospect information, assess maturity, score services, and output a ProspectGrading JSON object as instructed.\n\n"
            f"PROSPECT_DATA:\n{json.dumps(context_payload, indent=2)}\n"
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = self.agent.run_sync(prompt)
                return result.output  # type: ignore[attr-defined]
            except Exception as e:
                print(f"❌ Attempt {attempt} failed during grading: {e}")
                if attempt == max_attempts:
                    print("❌ All grading attempts failed.")
                    return None
                print("🔄 Retrying grading...")

# ------------------- 4. SIMPLE MAIN -------------------
if __name__ == "__main__":
    """Run grading and print result."""

    INPUT_PATH = "sample_output.json"

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Could not read '{INPUT_PATH}': {e}")
        raise SystemExit(1)

    enriched_lead = data.get("enriched_lead", {})
    evaluation_result = data.get("evaluation_result", {})

    grader = GradingAgent()
    grading = grader.run(enriched_lead, evaluation_result)

    graded_output = None  # placeholder
    if grading:
        print("\n=========== GRADING RESULT ===========\n")
        print(json.dumps(grading.model_dump(), indent=4))

        # Combine and save full JSON (input + grading)
        combined = data.copy()
        combined["grading_result"] = grading.model_dump()
        OUTPUT_PATH = "sample_output_with_grading.json"
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=4)
            print(f"✅ Full output saved to '{OUTPUT_PATH}'.")
        except Exception as e:
            print(f"❌ Could not save combined output: {e}")
    else:
        print("❌ No grading produced.") 