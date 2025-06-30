from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import traceback
from datetime import datetime

# Import from your existing backend
from new_backend import (
    process_lead_data,
    ModelType,
    AzureModelType,
    get_search_quota_status,
    reset_search_quota,
    clear_search_cache
)

app = FastAPI(title="Lead Processing API", version="1.0.0")

# Add CORS middleware for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- REQUEST/RESPONSE MODELS -------------------

class ProcessLeadRequest(BaseModel):
    model_type: str = ModelType.AZURE
    azure_model: str = AzureModelType.GPT4_1_MINI
    preferred_channel: Optional[str] = None
    unified_messaging_preferences: Optional[str] = None
    custom_grading_prompt: Optional[str] = None
    use_minimal_evaluation: bool = False

class ProcessLeadResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
    search_quota_status: Optional[dict] = None
    timestamp: str

class QuotaStatusResponse(BaseModel):
    calls_made: int
    calls_remaining: int
    total_limit: int
    cache_entries: int

class MessageResponse(BaseModel):
    success: bool
    message: str
    timestamp: str

# ------------------- API ENDPOINTS -------------------

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Lead Processing API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check with quota status"""
    try:
        quota_status = get_search_quota_status()
        return {
            "status": "healthy",
            "quota_status": quota_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/process-lead", response_model=ProcessLeadResponse)
async def process_lead(
    file: UploadFile = File(..., description="XLSX file containing lead data"),
    model_type: str = Form(default=ModelType.AZURE),
    azure_model: str = Form(default=AzureModelType.GPT4_1_MINI),
    preferred_channel: Optional[str] = Form(default=None),
    unified_messaging_preferences: Optional[str] = Form(default=None),
    custom_grading_prompt: Optional[str] = Form(default=None),
    use_minimal_evaluation: bool = Form(default=False)
):
    """
    Process lead data from an uploaded XLSX file.
    
    Returns enriched lead data, evaluation results, grading, and personalized messaging.
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an Excel file (.xlsx or .xls)"
            )
        
        # Read file data
        file_data = await file.read()
        
        # Process the lead
        print(f"🚀 Processing lead from file: {file.filename}")
        
        initial_lead, enriched_result, evaluation_output, grading_output, messaging_output = process_lead_data(
            file_data=file_data,
            model_type=model_type,
            azure_model=azure_model,
            preferred_channel=preferred_channel,
            unified_messaging_preferences=unified_messaging_preferences,
            custom_grading_prompt=custom_grading_prompt,
            use_minimal_evaluation=use_minimal_evaluation
        )
        
        if not initial_lead:
            raise HTTPException(
                status_code=400,
                detail="Failed to process lead data. Please check the file format and content."
            )
        
        # Prepare response data
        response_data = {
            "initial_lead": initial_lead,
            "enriched_lead": enriched_result.model_dump() if enriched_result else None,
            "evaluation_result": evaluation_output.model_dump() if evaluation_output else None,
            "grading_result": grading_output.model_dump() if grading_output else None,
            "messaging_result": messaging_output.model_dump() if messaging_output else None
        }
        
        # Get current quota status
        quota_status = get_search_quota_status()
        
        return ProcessLeadResponse(
            success=True,
            message="Lead processed successfully",
            data=response_data,
            search_quota_status=quota_status,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing lead: {str(e)}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/quota-status", response_model=QuotaStatusResponse)
async def get_quota_status():
    """Get current search quota usage status"""
    try:
        status = get_search_quota_status()
        return QuotaStatusResponse(**status)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quota status: {str(e)}"
        )

@app.post("/reset-quota", response_model=MessageResponse)
async def reset_quota():
    """Reset the search quota counter"""
    try:
        reset_search_quota()
        return MessageResponse(
            success=True,
            message="Search quota reset successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset quota: {str(e)}"
        )

@app.post("/clear-cache", response_model=MessageResponse)
async def clear_cache():
    """Clear the search cache to free memory"""
    try:
        clear_search_cache()
        return MessageResponse(
            success=True,
            message="Search cache cleared successfully",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@app.get("/model-options")
async def get_model_options():
    """Get available model options for the frontend"""
    return {
        "model_types": {
            "azure": ModelType.AZURE,
            "mistral": ModelType.MISTRAL,
            "claude": ModelType.CLAUDE
        },
        "azure_models": {
            "gpt4_1": AzureModelType.GPT4_1,
            "gpt4_1_mini": AzureModelType.GPT4_1_MINI,
            "o3_mini": AzureModelType.O3_MINI,
            "o4_mini": AzureModelType.O4_MINI
        },
        "channels": ["Email", "LinkedIn", "Phone", "Other"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/version")
async def get_version():
    """Get API version and build information"""
    return {
        "version": "1.0.0",
        "build_date": "2024-01-01",
        "features": [
            "Lead Enrichment",
            "Lead Evaluation", 
            "Prospect Grading",
            "Personalized Messaging",
            "Multi-model Support",
            "Quota Management"
        ],
        "timestamp": datetime.now().isoformat()
    }

# ------------------- ERROR HANDLERS -------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unhandled exception: {str(exc)}")
    print(f"❌ Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# ------------------- ADDITIONAL ENDPOINTS -------------------

@app.post("/enrich-lead")
async def enrich_lead_only(
    file: UploadFile = File(..., description="XLSX file containing lead data"),
    model_type: str = Form(default=ModelType.AZURE),
    azure_model: str = Form(default=AzureModelType.GPT4_1_MINI)
):
    """
    Enrich lead data only (without evaluation, grading, or messaging).
    Useful for when you only need basic lead enrichment.
    """
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        file_data = await file.read()
        
        # Import individual functions (you may need to expose these from new_backend)
        from new_backend import enrich_lead_data
        
        initial_lead, enriched_result = enrich_lead_data(
            file_data=file_data,
            model_type=model_type,
            azure_model=azure_model
        )
        
        return {
            "success": True,
            "message": "Lead enriched successfully",
            "data": {
                "initial_lead": initial_lead,
                "enriched_lead": enriched_result.model_dump() if enriched_result else None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {str(e)}")

@app.post("/evaluate-lead")
async def evaluate_lead_only(
    file: UploadFile = File(..., description="XLSX file containing lead data"),
    model_type: str = Form(default=ModelType.AZURE),
    azure_model: str = Form(default=AzureModelType.GPT4_1_MINI),
    use_minimal_evaluation: bool = Form(default=False)
):
    """
    Evaluate lead data only (without grading or messaging).
    """
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        file_data = await file.read()
        
        from new_backend import evaluate_lead_data
        
        initial_lead, enriched_result, evaluation_output = evaluate_lead_data(
            file_data=file_data,
            model_type=model_type,
            azure_model=azure_model,
            use_minimal_evaluation=use_minimal_evaluation
        )
        
        return {
            "success": True,
            "message": "Lead evaluated successfully",
            "data": {
                "initial_lead": initial_lead,
                "enriched_lead": enriched_result.model_dump() if enriched_result else None,
                "evaluation_result": evaluation_output.model_dump() if evaluation_output else None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/grade-prospect")
async def grade_prospect_only(
    enriched_data: dict = Form(..., description="Enriched lead data as JSON"),
    evaluation_data: Optional[dict] = Form(default=None, description="Evaluation data as JSON"),
    custom_grading_prompt: Optional[str] = Form(default=None),
    model_type: str = Form(default=ModelType.AZURE),
    azure_model: str = Form(default=AzureModelType.GPT4_1_MINI)
):
    """
    Grade a prospect using existing enriched and evaluation data.
    Useful when you already have processed data and just need grading.
    """
    try:
        from grading_agent import GradingAgent
        
        grader = GradingAgent(
            model_type=model_type,
            azure_model=azure_model,
            custom_prompt=custom_grading_prompt
        )
        
        grading_result = grader.run(enriched_data, evaluation_data)
        
        return {
            "success": True,
            "message": "Prospect graded successfully",
            "data": {
                "grading_result": grading_result.model_dump() if grading_result else None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")

@app.post("/create-message")
async def create_message_only(
    combined_data: dict = Form(..., description="Combined lead data as JSON"),
    preferred_channel: Optional[str] = Form(default=None),
    unified_messaging_preferences: Optional[str] = Form(default=None),
    company_name: Optional[str] = Form(default="Data Design Oy"),
    company_website: Optional[str] = Form(default="datadesign.fi"),
    company_description: Optional[str] = Form(default=None),
    model_type: str = Form(default=ModelType.AZURE),
    azure_model: str = Form(default=AzureModelType.GPT4_1_MINI)
):
    """
    Create personalized messaging using existing processed data.
    Allows customization of company profile.
    """
    try:
        from messaging_agent import MessagingAgent, CompanyProfile
        
        # Create custom company profile if provided
        company_profile = None
        if company_description:
            company_profile = CompanyProfile(
                company_name=company_name,
                website=company_website,
                company_description=company_description
            )
        
        messenger = MessagingAgent(
            model_type=model_type,
            azure_model=azure_model,
            company_profile=company_profile
        )
        
        message_result = messenger.run(
            combined_json=combined_data,
            preferred_channel=preferred_channel,
            user_preferences=unified_messaging_preferences
        )
        
        return {
            "success": True,
            "message": "Personalized message created successfully",
            "data": {
                "messaging_result": message_result.model_dump() if message_result else None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Messaging failed: {str(e)}")

@app.post("/batch-process-leads")
async def batch_process_leads(
    files: List[UploadFile] = File(..., description="Multiple XLSX files containing lead data"),
    model_type: str = Form(default=ModelType.AZURE),
    azure_model: str = Form(default=AzureModelType.GPT4_1_MINI),
    preferred_channel: Optional[str] = Form(default=None),
    unified_messaging_preferences: Optional[str] = Form(default=None),
    custom_grading_prompt: Optional[str] = Form(default=None),
    use_minimal_evaluation: bool = Form(default=False)
):
    """
    Process multiple lead files in batch.
    Returns results for all files with success/failure status for each.
    """
    results = []
    
    for file in files:
        try:
            if not file.filename.endswith(('.xlsx', '.xls')):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid file type",
                    "data": None
                })
                continue
            
            file_data = await file.read()
            
            initial_lead, enriched_result, evaluation_output, grading_output, messaging_output = process_lead_data(
                file_data=file_data,
                model_type=model_type,
                azure_model=azure_model,
                preferred_channel=preferred_channel,
                unified_messaging_preferences=unified_messaging_preferences,
                custom_grading_prompt=custom_grading_prompt,
                use_minimal_evaluation=use_minimal_evaluation
            )
            
            response_data = {
                "initial_lead": initial_lead,
                "enriched_lead": enriched_result.model_dump() if enriched_result else None,
                "evaluation_result": evaluation_output.model_dump() if evaluation_output else None,
                "grading_result": grading_output.model_dump() if grading_output else None,
                "messaging_result": messaging_output.model_dump() if messaging_output else None
            }
            
            results.append({
                "filename": file.filename,
                "success": True,
                "error": None,
                "data": response_data
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
                "data": None
            })
    
    successful_count = sum(1 for r in results if r["success"])
    
    return {
        "success": True,
        "message": f"Batch processing completed: {successful_count}/{len(results)} files processed successfully",
        "results": results,
        "summary": {
            "total_files": len(results),
            "successful": successful_count,
            "failed": len(results) - successful_count
        },
        "search_quota_status": get_search_quota_status(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/export-results/{format}")
async def export_results(
    format: str,
    data: dict = Form(..., description="Processed lead data to export")
):
    """
    Export processed lead results in different formats (json, csv, xlsx).
    """
    try:
        if format.lower() == "json":
            from fastapi.responses import Response
            return Response(
                content=json.dumps(data, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=lead_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
            )
        
        elif format.lower() == "csv":
            import pandas as pd
            from io import StringIO
            
            # Flatten the data for CSV export
            flattened_data = []
            
            # Extract key information
            initial_lead = data.get("initial_lead", {})
            enriched_lead = data.get("enriched_lead", {})
            evaluation_result = data.get("evaluation_result", {})
            grading_result = data.get("grading_result", {})
            messaging_result = data.get("messaging_result", {})
            
            flat_row = {**initial_lead}
            if enriched_lead:
                flat_row.update({f"enriched_{k}": v for k, v in enriched_lead.items()})
            if evaluation_result:
                flat_row.update({f"evaluation_{k}": v for k, v in evaluation_result.items()})
            if grading_result:
                flat_row.update({f"grading_{k}": v for k, v in grading_result.items()})
            if messaging_result:
                flat_row.update({f"messaging_{k}": v for k, v in messaging_result.items()})
            
            flattened_data.append(flat_row)
            
            df = pd.DataFrame(flattened_data)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            
            from fastapi.responses import Response
            return Response(
                content=csv_buffer.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=lead_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'json' or 'csv'")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/analytics/summary")
async def get_analytics_summary():
    """
    Get analytics summary (requires implementing data persistence layer).
    This is a placeholder for future implementation.
    """
    return {
        "message": "Analytics feature coming soon",
        "suggestion": "Implement data persistence to track processed leads over time",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/validate-lead-file")
async def validate_lead_file(
    file: UploadFile = File(..., description="XLSX file to validate")
):
    """
    Validate a lead file structure without processing it.
    Useful for checking file format before expensive processing.
    """
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        file_data = await file.read()
        
        # Basic validation (you may need to implement this in new_backend)
        import pandas as pd
        from io import BytesIO
        
        df = pd.read_excel(BytesIO(file_data))
        
        validation_result = {
            "is_valid": True,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "sample_data": df.head(3).to_dict('records') if len(df) > 0 else [],
            "missing_values": df.isnull().sum().to_dict(),
            "issues": []
        }
        
        # Add validation logic
        required_columns = ["company", "name"]  # Adjust based on your requirements
        missing_required = [col for col in required_columns if col not in df.columns]
        
        if missing_required:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Missing required columns: {missing_required}")
        
        if len(df) == 0:
            validation_result["is_valid"] = False
            validation_result["issues"].append("File is empty")
        
        return {
            "success": True,
            "message": "File validation completed",
            "validation_result": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/custom-company-profile")
async def create_custom_company_profile(
    company_name: str = Form(...),
    website: str = Form(...),
    company_description: str = Form(...),
    test_data: Optional[dict] = Form(default=None, description="Test data to validate messaging")
):
    """
    Create and test a custom company profile for messaging.
    """
    try:
        from messaging_agent import CompanyProfile
        
        profile = CompanyProfile(
            company_name=company_name,
            website=website,
            company_description=company_description
        )
        
        response_data = {
            "company_profile": profile.model_dump(),
            "messaging_tone": profile.messaging_tone,
            "message_length_limit": profile.message_length_limit
        }
        
        # If test data provided, generate a sample message
        if test_data:
            from messaging_agent import MessagingAgent
            
            messenger = MessagingAgent(company_profile=profile)
            test_message = messenger.run(test_data)
            
            response_data["test_message"] = test_message.model_dump() if test_message else None
        
        return {
            "success": True,
            "message": "Custom company profile created successfully",
            "data": response_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile creation failed: {str(e)}")

@app.get("/system-status")
async def get_system_status():
    """
    Get comprehensive system status including model availability, quotas, and health.
    """
    try:
        status = {
            "api_status": "healthy",
            "quota_status": get_search_quota_status(),
            "available_models": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Test model availability
        try:
            from grading_agent import get_model
            model, provider = get_model(ModelType.AZURE, AzureModelType.GPT4_1_MINI)
            status["available_models"]["azure"] = "available"
        except Exception as e:
            status["available_models"]["azure"] = f"unavailable: {str(e)}"
        
        try:
            model, provider = get_model(ModelType.MISTRAL)
            status["available_models"]["mistral"] = "available"
        except Exception as e:
            status["available_models"]["mistral"] = f"unavailable: {str(e)}"
        
        try:
            model, provider = get_model(ModelType.CLAUDE)
            status["available_models"]["claude"] = "available"
        except Exception as e:
            status["available_models"]["claude"] = f"unavailable: {str(e)}"
        
        return status
        
    except Exception as e:
        return {
            "api_status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 