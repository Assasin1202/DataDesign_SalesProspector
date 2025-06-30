#!/usr/bin/env python3
"""
Simple script to run the FastAPI server for the Lead Processing API.
"""

import uvicorn
import os
from pathlib import Path

def main():
    """Start the FastAPI server."""
    
    # Ensure outputs directory exists
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Ensure logs directory exists  
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting Lead Processing API...")
    print("📁 Created necessary directories")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🔧 Alternative docs: http://localhost:8000/redoc")
    print("\n" + "="*50)
    
    # Start the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 