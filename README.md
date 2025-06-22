# Sales Prospecting Research Agent System

A comprehensive AI-powered lead enrichment and research system designed for Data Design Oy, featuring multiple specialized agents that work together to enrich sales leads, evaluate data accuracy, grade prospects, and generate personalized outreach messages.

## 🏗️ System Architecture

The system consists of four main AI agents that work in sequence:

1. **Lead Enrichment Agent** - Enriches basic lead data with comprehensive web research
2. **Evaluator Agent** - Verifies and scores the accuracy of enriched data
3. **Grading Agent** - Assesses prospect maturity and ICP (Ideal Customer Profile) fit
4. **Messaging Agent** - Generates personalized outreach messages

## 📋 Table of Contents

- [Features](#features)
- [System Components](#system-components)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Agent Details](#agent-details)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Multi-Model Support**: Azure OpenAI, Mistral, and Claude models
- **Comprehensive Lead Enrichment**: Web search integration with Google Custom Search and Grok fallback
- **Data Accuracy Verification**: Automated fact-checking and source verification
- **Prospect Scoring**: Dynamic grading based on AI maturity, data utilization, and governance
- **Personalized Messaging**: Channel-specific outreach message generation with customizable company profiles
- **Web UI**: Streamlit-based interface for easy interaction
- **Website Verification**: Automatic verification of company websites
- **Caching**: Built-in search result caching to optimize API usage
- **Rate Limiting**: Intelligent quota management for external APIs

## 🔧 System Components

### Core Files

- `new_backend.py` - Main backend logic and agent orchestration
- `grading_agent.py` - Prospect grading and ICP assessment
- `messaging_agent.py` - Personalized outreach message generation
- `newapp.py` - Streamlit web interface
- `Dockerfile` - Container configuration

### Key Classes

- `LeadEnrichmentAgent` - Handles web research and data enrichment
- `EvaluatorAgent` - Validates enriched data accuracy
- `GradingAgent` - Scores prospects based on maturity criteria
- `MessagingAgent` - Creates personalized outreach content

## 🚀 Installation

### Prerequisites

- Python 3.11+
- API keys for chosen model providers
- Google Custom Search API credentials (optional)
- Grok API key (optional, for fallback search)

### Environment Setup

1. Clone the repository and navigate to the UI directory:
```bash
cd SalesProspector/Research_agent/ui/
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API credentials:
```env
# Azure OpenAI (if using Azure models)
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_API_KEY=your_azure_api_key

# Mistral (if using Mistral models)
MISTRAL_API_KEY=your_mistral_api_key

# Claude (if using Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Custom Search (recommended)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# Grok (optional fallback)
GROK_API_KEY=your_grok_api_key

# Optional: Limit Google Search API calls per run
MAX_GOOGLE_SEARCH_CALLS=15
```

## ⚙️ Configuration

### Model Selection

The system supports three model providers:

- **Azure OpenAI**: `gpt-4.1`, `gpt-4.1-mini`, `o3-mini`, `o4-mini`
- **Mistral**: `mistral-small-latest`
- **Claude**: `claude-sonnet-4-20250514`

### Search Configuration

- **Primary**: Google Custom Search API
- **Fallback**: Grok search when Google quota is exhausted
- **Caching**: Automatic result caching to minimize API calls
- **Rate Limiting**: Configurable limits to prevent quota exhaustion

## 📖 Usage

### Web Interface

1. Start the Streamlit application:
```bash
streamlit run newapp.py
```

2. Open your browser to `http://localhost:8501`

3. Upload an XLSX file containing lead data

4. Configure model and messaging preferences

5. Click "Run Enrichment & Evaluation" to process leads

### Programmatic Usage

```python
from new_backend import process_lead_data, ModelType, AzureModelType
from messaging_agent import create_custom_company_profile

# Read your XLSX file
with open('leads.xlsx', 'rb') as f:
    file_data = f.read()

# Option 1: Use default Data Design Oy configuration
initial_lead, enriched_lead, evaluation_result, grading_result, messaging_result = process_lead_data(
    file_data=file_data,
    model_type=ModelType.AZURE,
    azure_model=AzureModelType.GPT4_1_MINI,
    preferred_channel="Email",
    user_preferences="Keep tone professional and mention ROI"
)

# Option 2: Use custom company profile
custom_profile = create_custom_company_profile(
    company_info="TechSolutions Inc - delivers cutting-edge software development and cloud consulting services"
)

initial_lead, enriched_lead, evaluation_result, grading_result, messaging_result = process_lead_data(
    file_data=file_data,
    model_type=ModelType.AZURE,
    azure_model=AzureModelType.GPT4_1_MINI,
    preferred_channel="LinkedIn",
    user_preferences="Focus on technical benefits",
    company_profile=custom_profile
)

# Access results
if enriched_lead:
    print(f"Enriched paragraph: {enriched_lead.enriched_paragraph}")
    print(f"Company website: {enriched_lead.company_website}")
    print(f"Website verified: {enriched_lead.website_verified}")
    
if messaging_result:
    print(f"Outreach channel: {messaging_result.channel}")
    print(f"Subject: {messaging_result.subject}")
    print(f"Message: {messaging_result.message}")
```

## 🔍 Agent Details

### Lead Enrichment Agent

**Purpose**: Enriches basic lead information with comprehensive web research

**Key Functions**:
- Analyzes lead data to determine optimal search time ranges
- Searches for company information, person details, and project data
- Verifies company websites
- Extracts contact details including HR information
- Prioritizes official sources (company website > LinkedIn > third-party)

**Output**: `EnrichedLead` model containing:
- `enriched_paragraph`: Comprehensive summary
- `web_sources`: List of source URLs
- `project_details`: Company's data/AI projects
- `company_contact_details`: Contact information
- `company_website`: Verified website URL
- `website_verified`: Boolean verification status

### Evaluator Agent

**Purpose**: Verifies accuracy of enriched data against source materials

**Key Functions**:
- Breaks down enriched content into individual claims
- Verifies claims against provided source URLs
- Performs independent fact-checking searches
- Scores accuracy across multiple dimensions

**Output**: `EvaluationResult` model containing:
- `factual_accuracy_score`: 0.0-1.0 accuracy rating
- `temporal_accuracy_score`: Time-based accuracy rating
- `source_citation_score`: Source support rating
- `evaluation_summary`: Human-readable assessment
- `identified_discrepancies`: List of specific errors

### Grading Agent

**Purpose**: Assesses prospect maturity and ICP fit for Data Design Oy

**Scoring Categories**:
- **AI Maturity**: 1-5 scale from awareness to transformational
- **Data Utilization**: 1-5 scale from immature to data mature
- **Data Governance**: 1-5 scale from beginner to champion
- **Service Fit**: Scores for Strategy, Governance, Architecture, Implementation

**Output**: `ProspectGrading` model containing:
- `prospect_summary`: One-line prospect summary
- `scores`: List of category scores with justifications
- `priority_service`: Recommended service focus
- `key_insights`: Actionable insights for outreach

### Messaging Agent

**Purpose**: Creates personalized outreach messages with customizable company profiles

**Key Functions**:
- Analyzes all previous agent outputs
- Selects optimal communication channel
- Crafts personalized subject lines and messages
- Incorporates configurable company value propositions and services
- Supports custom messaging tone and length limits
- Defaults to Data Design Oy configuration when no custom profile provided

**Customization Options**:
- Simple company information input (name and description)
- Automatic website extraction
- Fixed professional messaging tone and 120-word limit for consistency

**Output**: `PersonalizedMessage` model containing:
- `channel`: Recommended outreach channel
- `subject`: Email subject or LinkedIn hook
- `message`: Personalized message body

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t research-agent .

# Run with environment variables
docker run -p 8501:8501 \
  -e AZURE_OPENAI_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  -e GOOGLE_CSE_ID=your_cse_id \
  research-agent
```

### Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  research-agent:
    build: .
    ports:

      - "8501:8501"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - GOOGLE_CSE_ID=${GOOGLE_CSE_ID}
      - GROK_API_KEY=${GROK_API_KEY}
    volumes:
      - ./outputs:/app/outputs
```

Run with: `docker-compose up`

## 📊 Data Models

### Input Data Format

Expected XLSX columns (flexible):
- Company name
- Person name
- Job title
- Contact information
- Any additional context

### Output Structure

```json
{
  "enriched_lead": {
    "original_lead_data": {...},
    "enriched_paragraph": "...",
    "web_sources": ["url1", "url2"],
    "project_details": "...",
    "company_contact_details": "...",
    "company_website": "https://...",
    "website_verified": true
  },
  "evaluation_result": {
    "factual_accuracy_score": 0.85,
    "temporal_accuracy_score": 0.90,
    "source_citation_score": 0.80,
    "evaluation_summary": "...",
    "identified_discrepancies": []
  },
  "grading_result": {
    "prospect_summary": "...",
    "scores": [...],
    "priority_service": "Data & AI Strategy",
    "key_insights": [...]
  },
  "messaging_result": {
    "channel": "Email",
    "subject": "...",
    "message": "..."
  }
}
```

## 🛠️ Customization

### Custom Company Profiles

You can customize the messaging agent for any company while keeping Data Design Oy as the default:

```python
from messaging_agent import create_custom_company_profile

# Create a custom company profile with simple input
custom_profile = create_custom_company_profile(
    company_info="YourCompany Inc - transforms businesses through innovative technology solutions"
)

# Use in processing
process_lead_data(
    file_data=data,
    company_profile=custom_profile
)
```

### Custom Grading Prompts

You can override the default Data Design Oy grading criteria by providing a custom prompt:

```python
custom_prompt = """
Your custom ICP definition and scoring criteria here...
Focus on your specific industry, company size, and evaluation factors.
"""

# Use in processing
process_lead_data(
    file_data=data,
    custom_grading_prompt=custom_prompt
)
```

### Search Customization

Modify search behavior by adjusting:
- `MAX_GOOGLE_SEARCH_CALLS`: API quota limits
- Time ranges: "y", "2y", "5y", or None
- Result counts: `max_results` parameter

## 🔧 Troubleshooting

### Common Issues

1. **API Quota Exceeded**
   - Solution: Reduce `MAX_GOOGLE_SEARCH_CALLS` or use Grok fallback
   - Check: Verify API key validity and quota limits

2. **Website Verification Failures**
   - Solution: Check network connectivity and firewall settings
   - Note: Some websites may block automated requests

3. **Model Response Errors**
   - Solution: Verify API keys and endpoint configurations
   - Try: Different model providers or deployment names

4. **Memory Issues with Large Files**
   - Solution: Process leads in smaller batches
   - Consider: Upgrading system resources

### Debugging

Enable verbose logging by setting agent `verbose=True` or add debug prints:

```python
# In new_backend.py
print(f"🤖 Processing lead: {lead_data}")
print(f"🔍 Search results: {search_results}")
```

## 📝 Examples

### Example Lead Data (XLSX)

| Company | Name | Title | Email | LinkedIn |
|---------|------|-------|-------|----------|
| TechCorp | John Smith | CTO | john@techcorp.com | linkedin.com/in/johnsmith |
| DataInc | Jane Doe | Head of Analytics | jane@datainc.com | linkedin.com/in/janedoe |

### Example Output

```json
{
  "enriched_lead": {
    "enriched_paragraph": "John Smith serves as Chief Technology Officer at TechCorp, a rapidly growing fintech company based in Helsinki. TechCorp specializes in digital payment solutions and has recently secured €10M in Series A funding. The company is actively expanding their data analytics capabilities and exploring AI-driven fraud detection systems. John has over 8 years of experience in financial technology and joined TechCorp in 2021 to lead their digital transformation initiatives.",
    "web_sources": [
      "https://techcorp.com/about",
      "https://linkedin.com/in/johnsmith",
      "https://techcrunch.com/techcorp-funding"
    ],
    "company_website": "https://techcorp.com",
    "website_verified": true
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

Copyright (c) 2024 Data Design Oy. All rights reserved.

## 👥 Support

For technical support or questions:
- Create an issue in the repository
- Contact: support@datadesign.fi

---

*Made by Pranav Pant for Data Design Oy* 