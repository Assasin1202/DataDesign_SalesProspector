import os
import json
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider
import time
from urllib.parse import urlparse
from mistralai import Mistral
import anthropic
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import datetime
import pathlib
import re  # For simple URL extraction from Grok responses
# Support both package and standalone execution for local scripts
try:
    from .grading_agent import GradingAgent, ProspectGrading  # type: ignore
    from .messaging_agent import MessagingAgent, PersonalizedMessage, CompanyProfile  # type: ignore
except ImportError:
    from grading_agent import GradingAgent, ProspectGrading  # type: ignore
    from messaging_agent import MessagingAgent, PersonalizedMessage, CompanyProfile  # type: ignore
# HTTP verification
import requests

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

# ------------------- 2. DATA MODELS -------------------
class EnrichedLead(BaseModel):
    """A structured model to hold the enriched lead information and its sources."""
    original_lead_data: dict = Field(
        description="A dictionary containing the original, unmodified data from the source file."
    )
    enriched_paragraph: str = Field(
        description="A comprehensive paragraph summarizing the lead, incorporating web search results and providing context about the company and the person's role."
    )
    web_sources: list[str] = Field(
        description="A list of URLs from the web search that were used as sources for the enrichment."
    )
    project_details: str = Field(
        description="Detailed information about the company's key data and AIprojects and any insights gleaned from investor decks or official webpages."
    )
    company_contact_details: str = Field(
        description="Relevant company contact details, including HR contacts if available, as well as general background and offerings overview."
    )
    company_website: str = Field(
        description="The verified official website URL of the company."
    )
    website_verified: bool = Field(
        description="True if the provided company_website responded successfully (HTTP <400)."
    )

class EvaluationResult(BaseModel):
    """A structured model to hold the accuracy assessment of an enriched lead."""
    factual_accuracy_score: float = Field(
        description="A score from 0.0 to 1.0 indicating the factual accuracy of the enriched paragraph. 1.0 is perfectly accurate.",
        ge=0.0, le=1.0
    )
    temporal_accuracy_score: float = Field(
        description="A score from 0.0 to 1.0 indicating how well the information aligns with the appropriate time periods. 1.0 means all temporal claims are accurate.",
        ge=0.0, le=1.0
    )
    source_citation_score: float = Field(
        description="A score from 0.0 to 1.0 indicating how well the paragraph's claims are supported by the provided web_sources.",
        ge=0.0, le=1.0
    )
    evaluation_summary: str = Field(
        description="A concise, human-readable summary explaining the scores and highlighting any specific errors or discrepancies found, including temporal accuracy."
    )
    identified_discrepancies: list[str] = Field(
        description="A list of specific factual errors found (e.g., 'Claim: CEO, Actual: CTO', 'Claim: Joined in 2020, Actual: Joined in 2021')."
    )

# ------------------- 3. SEARCH TOOLS -------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Added caching and rate limiting so we stay under the daily Google Custom Search quota
GOOGLE_SEARCH_CACHE: dict[tuple[str, str | None, int], list[dict]] = {}
GOOGLE_SEARCH_CALL_COUNT = 0
# You can override the following env-var to raise/lower the hard limit per run
MAX_GOOGLE_SEARCH_CALLS = int(os.getenv("MAX_GOOGLE_SEARCH_CALLS", "40"))

def google_search(query: str, time_range: str = None, max_results: int = 3) -> list[dict]:
    global GOOGLE_SEARCH_CACHE, GOOGLE_SEARCH_CALL_COUNT
    cache_key = (query, time_range, max_results)
    # Return cached results if we have seen this exact query already
    if cache_key in GOOGLE_SEARCH_CACHE:
        return GOOGLE_SEARCH_CACHE[cache_key]
    # Hard cap on total calls made during a single run to avoid API quota errors
    if GOOGLE_SEARCH_CALL_COUNT >= MAX_GOOGLE_SEARCH_CALLS:
        print("⚠️  Google search call limit reached – returning empty results to avoid quota errors.")
        return []
    print(f"🤖 Executing Tool: google_search for '{query}' with time_range='{time_range}'...")
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        date_range = None
        if time_range:
            if time_range == "y":
                date_range = "d[365]"
            elif time_range == "2y":
                date_range = "d[730]"
            elif time_range == "5y":
                date_range = "d[1825]"
        result = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=max_results, dateRestrict=date_range).execute()
        search_results = []
        if 'items' in result:
            for item in result['items']:
                search_results.append({'body': item.get('snippet', ''), 'href': item.get('link', '')})
        # Cache the successful response
        GOOGLE_SEARCH_CACHE[cache_key] = search_results
        GOOGLE_SEARCH_CALL_COUNT += 1
        time.sleep(0.5)  # Reduced sleep time for faster processing
        return search_results
    except HttpError as e:
        print(f"Error in Google search: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in Google search: {e}")
        return []

def grok_search(query: str, max_results: int = 3) -> list[dict]:
    """Fallback search using x.ai Grok. Returns a list of dicts with 'body' and 'href'."""
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    if not GROK_API_KEY:
        print("⚠️  GROK_API_KEY not set; cannot perform Grok search.")
        return []

    print(f"🤖 Executing Tool: grok_search for '{query}' (max_results={max_results})...")
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK_API_KEY}"}
    # Ask Grok to include sources so we can attempt to extract URLs
    prompt = f"Give me up to {max_results} relevant news or web results (include source URLs) about: {query}. Return concise summaries."
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "search_parameters": {"mode": "auto"},
        "model": "grok-3-latest",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        # The main content is likely in choices[0]['message']['content'] (OpenAI style)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return []
        # Extract URLs from the content with a simple regex
        urls = re.findall(r"https?://\S+", content)
        results = []
        # Split the content by newline for bodies (crude but works)
        body_lines = [l.strip() for l in content.split("\n") if l.strip()]
        for i, line in enumerate(body_lines[:max_results]):
            href = urls[i] if i < len(urls) else ""
            results.append({"body": line, "href": href})
        return results
    except Exception as e:
        print(f"Error in Grok search: {e}")
        return []

def search_with_fallback(query: str, time_range: str = None, max_results: int = 3) -> list[dict]:
    """Wrapper around google_search (kept original name to minimize downstream changes)."""
    print(f"🤖 Executing Tool: google_search for '{query}' with time_range='{time_range}'...")
    results = google_search(query, time_range, max_results)
    if results:
        return results
    # If Google search failed or quota exhausted, try Grok as a fallback
    print("🔄 Google search yielded no results or quota exhausted. Falling back to Grok search...")
    return grok_search(query, max_results)

def search_company_info(company_name: str, time_range: str = None) -> list[dict]:
    print(f"🤖 Executing Tool: search_company_info for '{company_name}' with time_range='{time_range}'...")
    # Optimized: Reduced to 2 results to conserve API quota
    return search_with_fallback(f"{company_name} company information", time_range, 2)

def search_person_info(name: str, company: str, time_range: str = None) -> list[dict]:
    print(f"🤖 Executing Tool: search_person_info for '{name}' at '{company}' with time_range='{time_range}'...")
    # Optimized: Reduced to 2 results to conserve API quota
    return search_with_fallback(f"{name} {company} professional profile", time_range, 2)

def search_company_projects(company_name: str, time_range: str = None) -> list[dict]:
    """Search for data projects or AI projects if any and investor deck information related to the company."""
    print(f"🤖 Executing Tool: search_company_projects for '{company_name}' with time_range='{time_range}'...")
    query = f"{company_name} investor deck data AI project"
    # Optimized: Reduced to 2 results to conserve API quota
    return search_with_fallback(query, time_range, 2)

def read_url_content(url: str) -> str:
    print(f"🤖 Executing Tool: read_url_content for URL '{url}'...")
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        search_query = f"site:{domain} {path}"
        # Use Google Custom Search to retrieve snippets from the specific URL (site query)
        results = google_search(search_query, None, 3)
        if not results:
            return f"Error: No content found for URL {url}"
        content = " ".join([r['body'] for r in results])
        return content
    except Exception as e:
        return f"Error: Could not fetch content from URL {url}. Reason: {e}"

# ---- NEW: Website verification ----
def verify_website(url: str) -> bool:
    """Check if the given website is reachable (HTTP status < 400)."""
    print(f"🤖 Executing Tool: verify_website for URL '{url}'...")
    if not url.startswith("http"):
        url = "https://" + url
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code < 400
    except Exception as e:
        print(f"Website verification failed: {e}")
        return False

# ---- NEW: Search quota management ----
def reset_search_quota():
    """Reset the Google search call counter. Useful for processing multiple batches."""
    global GOOGLE_SEARCH_CALL_COUNT
    GOOGLE_SEARCH_CALL_COUNT = 0
    print(f"🔄 Google search quota reset. Calls remaining: {MAX_GOOGLE_SEARCH_CALLS}")

def clear_search_cache():
    """Clear the search cache to free memory and force fresh searches."""
    global GOOGLE_SEARCH_CACHE
    GOOGLE_SEARCH_CACHE.clear()
    print("🗑️ Search cache cleared.")

def get_search_quota_status():
    """Get current search quota usage."""
    remaining = MAX_GOOGLE_SEARCH_CALLS - GOOGLE_SEARCH_CALL_COUNT
    return {
        "calls_made": GOOGLE_SEARCH_CALL_COUNT,
        "calls_remaining": remaining,
        "total_limit": MAX_GOOGLE_SEARCH_CALLS,
        "cache_entries": len(GOOGLE_SEARCH_CACHE)
    }

# ---- NEW: Minimal search mode ----
def search_company_info_minimal(company_name: str, time_range: str = None) -> list[dict]:
    """Minimal company search - only 1 result to maximize quota efficiency."""
    print(f"🤖 MINIMAL: search_company_info for '{company_name}'...")
    return search_with_fallback(f"{company_name} official website", time_range, 1)

def search_person_company_combined(name: str, company: str, time_range: str = None) -> list[dict]:
    """Combined person+company search to reduce API calls."""
    print(f"🤖 COMBINED: search for '{name}' at '{company}'...")
    query = f"{name} {company} LinkedIn profile role"
    return search_with_fallback(query, time_range, 2)

# Alternative minimal search functions (3 calls total instead of 6)
MINIMAL_SEARCH_FUNCTIONS = [search_company_info_minimal, search_person_company_combined]

# ------------------- 4. AGENTS -------------------
class LeadEnrichmentAgent:
    """An agent designed to enrich sales leads using web search."""
    
    def __init__(self, model_type: str = ModelType.AZURE, azure_model: str = AzureModelType.GPT4_1_MINI):
        system_prompt = """
        You are a world-class business analyst specializing in lead enrichment. Your job is to take a sales lead and build a complete, context-rich profile that includes:
        • Key company background and offerings
        • Contact details (general and HR if publicly available)
        • Recent and ongoing data/AI projects (sourced from company webpages or investor decks)

        To achieve this, you must:
        1. Analyze the initial lead data provided to you.
        2. Determine the appropriate time range for searching information based on:
           – The industry and company context
           – The person's role and tenure
        3. Choose an appropriate time range from these options:
           – "y" (year) for annual context
           – "2y" (2 years) for medium-term context
           – "5y" (5 years) for long-term context
           – None for general information without time constraints
        4. Use the `search_company_info` tool to find information about the lead's company, using the determined time_range.
        5. Use the `search_person_info` tool to find details about the specific person and their role, using the determined time_range.
        6. Use the `search_company_projects` tool to uncover investor-deck snippets or webpages that describe the company's data/AI projects and initiatives.
        7. Identify what you believe is the company's official website. Use `verify_website` to confirm it is reachable and mark the result in `company_website` with the boolean result in `website_verified`.
        8. Extract and compile relevant contact details (including HR, if discoverable) such as phone numbers, generic email addresses (e.g., "info@"), HR/recruitment emails, and physical addresses.
        9. For any conflicting information (e.g., employee count, revenue, role titles):
           – Always prioritize data from the company's official website
           – Use LinkedIn as the second most authoritative source
           – Only use third-party sources when official sources are unavailable
           – Clearly indicate when information comes from unofficial sources
        10. After gathering all information, synthesize your findings into a comprehensive, professional, well-written summary paragraph.
        11. Ensure that the summary references company offerings, data/AI projects, and provides contact details where relevant.
        12. If you find conflicting information, explicitly state which source you prioritized and why.
        """
        model, provider = get_model(model_type, azure_model)
        self.agent = Agent(
            model=model,
            tools=[search_company_info, search_person_info, search_company_projects, verify_website],
            output_type=EnrichedLead,
            system_prompt=system_prompt,
            verbose=True
        )

    def run(self, lead_data: dict) -> EnrichedLead | None:
        """Runs the enrichment process for a given lead."""
        print("\n🚀 Starting lead enrichment process...")
        prompt = f"Enrich the following lead details:\n{json.dumps(lead_data, indent=2)}\n" \
                 "Based on this lead data:\n" \
                 "1. First, analyze the data and determine the most appropriate time range for searching information. Consider:\n" \
                 "   - Any dates mentioned in the data\n" \
                 "   - The type of information needed\n" \
                 "   - The industry context\n" \
                 "   - The person's role and tenure\n" \
                 "2. Then, use your search tools to gather comprehensive information about the company and person using the determined time range.\n" \
                 "3. When you find conflicting information:\n" \
                 "   - Prioritize the company's official website\n" \
                 "   - Use LinkedIn as your second source\n" \
                 "   - Only use third-party sources when official sources are unavailable\n" \
                 "   - Clearly indicate which source you're using for each piece of information\n" \
                 "4. Use the `search_company_projects` tool to gather information about the company's data/AI projects and investor-deck highlights.\n" \
                 "5. Determine the company's official website, verify it using `verify_website`, and include it in `company_website` with the boolean result in `website_verified`.\n" \
                 "6. Compile any publicly available contact details (including HR contacts) and provide them in the `company_contact_details` field.\n" \
                 "7. Finally, synthesize all findings into a detailed and professional summary paragraph that includes project insights, verified website, and contact details.\n" \
                 "8. If source prioritization is needed, state it in your response.\n"
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = self.agent.run_sync(prompt)
                return result.output
            except Exception as e:
                print(f"❌ Attempt {attempt} failed during enrichment: {e}")
                if attempt == max_attempts:
                    print("❌ All enrichment attempts failed.")
                    return None
                print("🔄 Retrying enrichment...")
                time.sleep(2)

class EvaluatorAgent:
    """An agent designed to verify and score the accuracy of an enriched lead."""
    
    def __init__(self, model_type: str = ModelType.AZURE, azure_model: str = AzureModelType.GPT4_1_MINI):
        system_prompt = """
        You are a world-class, meticulous Quality Assurance Analyst and Fact-Checker.
        Your sole job is to evaluate the accuracy of a profile generated by a Lead Enrichment agent.

        IMPORTANT: To conserve search quota, you should primarily rely on the provided sources and avoid additional searches.

        To do this, you will:
        1. Receive a JSON object containing the original lead, an enriched paragraph, and a list of source URLs.
        2. Break down the `enriched_paragraph` into individual factual claims (e.g., name, title, company facts, career history).
        3. For each claim, use the `read_url_content` tool on the provided `web_sources` to see if the claim is supported by the cited evidence.
        4. AVOID additional searches unless absolutely critical - focus on evaluating based on provided sources only.
        5. Focus on verifying:
           - The accuracy of information from the provided URLs
           - The consistency between claims and source content
           - The completeness of information from the sources
           - Logic and plausibility of claims even if sources are limited
        6. Based on your findings, you MUST meticulously fill out all fields in the `EvaluationResult` JSON structure.
        7. The `evaluation_summary` must be a clear, objective report on the quality of the enrichment.
        8. List specific errors in `identified_discrepancies`, focusing on mismatches between claims and source content.
        9. If sources are insufficient, note this in your evaluation but do NOT perform additional searches.
        """
        model, provider = get_model(model_type, azure_model)
        self.agent = Agent(
            model=model,
            tools=[read_url_content,search_with_fallback],  # Removed search_with_fallback to eliminate fact-checking searches
            output_type=EvaluationResult,
            system_prompt=system_prompt,
            verbose=True
        )

    def run(self, enriched_lead: EnrichedLead) -> EvaluationResult | None:
        """Runs the evaluation process for a given enriched lead."""
        print("\n🔎 Starting evaluation process...")
        print("📊 QUOTA-OPTIMIZED: Using source-only verification (no additional searches)")
        prompt = f"""
        Please evaluate the accuracy of the following enriched lead data by verifying the content of the provided URLs:\n{enriched_lead.model_dump_json()}\n\nFocus on:\n1. Verifying each claim against the content of the provided URLs ONLY\n2. Checking for consistency between claims and source content\n3. Identifying any discrepancies between the enriched paragraph and the source URLs\n4. Assessing the quality and reliability of the information from the sources\n5. If any links are inaccessible, note this limitation but do NOT search for alternatives\n6. Base your scores on source verification and logical consistency only\n7. IMPORTANT: Do NOT perform any additional web searches - work only with provided sources\n"""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = self.agent.run_sync(prompt)
                return result.output
            except Exception as e:
                print(f"❌ Attempt {attempt} failed during evaluation: {e}")
                if attempt == max_attempts:
                    print("❌ All evaluation attempts failed.")
                    return None
                print("🔄 Retrying evaluation...")
                time.sleep(2)

# ---- NEW: Minimal evaluation mode for maximum quota conservation ----
class MinimalEvaluatorAgent:
    """A minimal evaluator that provides basic assessment without using any search tools."""
    
    def __init__(self, model_type: str = ModelType.AZURE, azure_model: str = AzureModelType.GPT4_1_MINI):
        system_prompt = """
        You are a basic quality assessment agent. Your job is to provide a simple evaluation of enriched lead data.
        
        IMPORTANT: You have NO search tools available - work only with the provided information.
        
        Evaluate the enriched lead by:
        1. Checking if the enriched paragraph appears complete and coherent
        2. Verifying that sources are provided (URLs exist in web_sources)
        3. Assessing if the content seems reasonable and professional
        4. Providing conservative scores based on apparent quality
        
        Since you cannot verify facts independently:
        - Give moderate scores (0.7-0.8) for well-structured content
        - Note any obvious inconsistencies within the provided data
        - Focus on logical consistency rather than factual verification
        """
        model, provider = get_model(model_type, azure_model)
        self.agent = Agent(
            model=model,
            tools=[],  # No tools - zero search usage
            output_type=EvaluationResult,
            system_prompt=system_prompt,
            verbose=True
        )

    def run(self, enriched_lead: EnrichedLead) -> EvaluationResult | None:
        """Runs minimal evaluation without any search calls."""
        print("\n⚡ MINIMAL EVALUATION: Zero search quota usage")
        prompt = f"""
        Provide a basic quality assessment for this enriched lead data:\n{enriched_lead.model_dump_json()}\n
        
        Since you cannot verify facts:
        - Focus on content structure and coherence
        - Give moderate confidence scores (0.7-0.8 range)
        - Note if sources are provided and content appears professional
        - Mark any obvious internal inconsistencies
        """
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = self.agent.run_sync(prompt)
                return result.output
            except Exception as e:
                print(f"❌ Attempt {attempt} failed during minimal evaluation: {e}")
                if attempt == max_attempts:
                    print("❌ All minimal evaluation attempts failed.")
                    return None
                print("🔄 Retrying minimal evaluation...")
                time.sleep(2)

# ---- NEW: Unified preferences parsing ----
def parse_unified_messaging_preferences(unified_text: str | None) -> tuple[CompanyProfile | None, str | None]:
    """
    Parse unified messaging preferences text to extract company profile and user preferences.
    
    Expected format examples:
    - "Company: TechSolutions Inc - delivers software development services"
    - "Website: techsolutions.com" 
    - "Keep tone formal, mention 24/7 support"
    - Mixed: company info + preferences together
    
    Returns:
        tuple: (CompanyProfile or None, user_preferences string or None)
    """
    if not unified_text or not unified_text.strip():
        return None, None
    
    text = unified_text.strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Extract company information
    company_name = None
    company_description = None
    company_website = None
    user_preferences_lines = []
    
    for line in lines:
        # Look for company information patterns
        if line.lower().startswith('company:') or line.lower().startswith('• company:'):
            company_line = line.split(':', 1)[1].strip()
            if ' - ' in company_line:
                company_name, company_description = company_line.split(' - ', 1)
                company_name = company_name.strip()
                company_description = company_description.strip()
            else:
                company_name = company_line.strip()
                company_description = "provides professional services and solutions"
        
        elif line.lower().startswith('website:') or line.lower().startswith('• website:'):
            company_website = line.split(':', 1)[1].strip()
        
        # Everything else is treated as user preferences
        elif not line.lower().startswith(('company:', '• company:', 'website:', '• website:')):
            # Clean up bullet points and formatting
            clean_line = line.lstrip('•').strip()
            if clean_line:
                user_preferences_lines.append(clean_line)
    
    # Create company profile if we found company info
    company_profile = None
    if company_name:
        full_description = company_description or "provides professional services and solutions"
        company_profile = CompanyProfile(
            company_name=company_name,
            website=company_website or "yourcompany.com",
            company_description=full_description
        )
    
    # Combine user preferences
    user_preferences = None
    if user_preferences_lines:
        user_preferences = '. '.join(user_preferences_lines)
        if not user_preferences.endswith('.'):
            user_preferences += '.'
    
    return company_profile, user_preferences

# ------------------- 5. UTILITY FUNCTIONS -------------------
def process_lead_data(
    file_data: bytes,
    model_type: str = ModelType.AZURE,
    azure_model: str = AzureModelType.GPT4_1_MINI,
    preferred_channel: str | None = None,
    unified_messaging_preferences: str | None = None,
    custom_grading_prompt: str | None = None,
    use_minimal_evaluation: bool = False,
) -> tuple[
    dict,
    EnrichedLead | None,
    EvaluationResult | None,
    ProspectGrading | None,
    PersonalizedMessage | None,
]:
    """Process lead data from an uploaded file and return the results.

    Parameters
    ----------
    file_data : bytes
        The raw bytes of the uploaded XLSX file.
    model_type, azure_model : str
        LLM provider selection (see UI).
    preferred_channel : str | None
        "Email", "LinkedIn", or *None* (let the MessagingAgent decide).
    unified_messaging_preferences : str | None
        Combined messaging preferences and company customization input.
        Can include user preferences, company info (format: "Company: Name - description"),
        or both. When *None*, uses Data Design Oy defaults.
    custom_grading_prompt : str | None
        Optional custom ICP / grading criteria supplied by the user. When
        *None*, the GradingAgent will use its default Data Design prompt.
    use_minimal_evaluation : bool
        If True, uses MinimalEvaluatorAgent (zero searches) instead of full
        EvaluatorAgent. Default False for backward compatibility.
    """
    try:
        # Save the uploaded file temporarily
        temp_file = "temp_upload.xlsx"
        with open(temp_file, "wb") as f:
            f.write(file_data)
        
        # Read the first row
        df = pd.read_excel(temp_file, nrows=1)
        initial_lead = df.iloc[0].to_dict()
        
        if initial_lead:
            # Process the lead
            enrichment_agent = LeadEnrichmentAgent(model_type=model_type, azure_model=azure_model)
            enriched_result = enrichment_agent.run(initial_lead)
            
            if enriched_result:
                # Choose evaluator based on quota conservation preference
                if use_minimal_evaluation:
                    evaluator = MinimalEvaluatorAgent(model_type=model_type, azure_model=azure_model)
                else:
                    evaluator = EvaluatorAgent(model_type=model_type, azure_model=azure_model)
                evaluation_output = evaluator.run(enriched_result)
                
                grading_agent = GradingAgent(
                    model_type=model_type,
                    azure_model=azure_model,
                    custom_prompt=custom_grading_prompt,
                )
                grading_output = grading_agent.run(enriched_result, evaluation_output)
                
                # Parse unified messaging preferences
                company_profile, user_preferences = parse_unified_messaging_preferences(unified_messaging_preferences)
                
                # Messaging generation using combined data
                messaging_agent = MessagingAgent(
                    model_type=model_type,
                    azure_model=azure_model,
                    company_profile=company_profile
                )
                combined_data_for_msg = {
                    "enriched_lead": enriched_result.model_dump(),
                    "evaluation_result": evaluation_output.model_dump() if evaluation_output else {},
                    "grading_result": grading_output.model_dump() if grading_output else {},
                }
                messaging_output = messaging_agent.run(
                    combined_data_for_msg,
                    preferred_channel=preferred_channel,
                    user_preferences=user_preferences,
                )
                
                # Save JSON outputs with timestamp for uniqueness
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = pathlib.Path("outputs")
                output_dir.mkdir(exist_ok=True)
                try:
                    combined_path = output_dir / f"combined_output_{timestamp}.json"
                    with combined_path.open("w", encoding="utf-8") as f:
                        json.dump({
                            "enriched_lead": enriched_result.model_dump(),
                            "evaluation_result": evaluation_output.model_dump(),
                            "grading_result": grading_output.model_dump() if grading_output else {},
                            "messaging_result": messaging_output.model_dump() if messaging_output else {}
                        }, f, indent=4)
                    print(f"✅ Saved combined JSON output to '{combined_path.resolve()}'")
                except Exception as e:
                    print(f"Error saving combined JSON output: {e}")
                os.remove(temp_file)
                return initial_lead, enriched_result, evaluation_output, grading_output, messaging_output
        
        # Clean up
        os.remove(temp_file)
        return initial_lead, None, None, None, None
        
    except Exception as e:
        print(f"Error processing lead data: {e}")
        return None, None, None, None, None 