"""
inference_api.py - Simple API for real-time inference

This module provides a simple HTTP API for accessing the inference engine
for real-time predictions and signal generation.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Import project modules
from config import Config
from inference import InferenceEngine, RealTimeInference
from data_handler import DataHandler
from utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Project Compound Inference API",
    description="API for real-time trading signals using the transformer model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
data_handler = None
inference_handler = None
last_update_time = None
latest_data = None


# Data models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    symbol: str = Field("BTCUSDT", description="Trading pair symbol")
    interval: str = Field("1h", description="Data interval")
    limit: int = Field(100, description="Number of data points to fetch")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    success: bool
    signal: str
    confidence: float
    prediction: Dict
    timestamp: str
    latency: float
    error: Optional[str] = None


class DataUpdateRequest(BaseModel):
    """Request model for data update endpoint"""
    force_update: bool = Field(False, description="Force data update")


class DataUpdateResponse(BaseModel):
    """Response model for data update endpoint"""
    success: bool
    message: str
    timestamp: str
    latency: float
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    status: str
    config_loaded: bool
    data_handler_initialized: bool
    inference_engine_initialized: bool
    last_update_time: Optional[str] = None
    data_shape: Optional[List[int]] = None
    model_info: Optional[Dict] = None
    uptime: float


# Initialization
def initialize():
    """Initialize the API components"""
    global config, data_handler, inference_handler, last_update_time
    
    try:
        # Load configuration
        config_path = os.environ.get('CONFIG_PATH', 'config/config.yaml')
        config = Config(config_path)
        
        # Initialize data handler
        data_handler = DataHandler(config)
        
        # Initialize inference engine
        inference_engine = InferenceEngine(config)
        
        # Initialize real-time inference handler
        inference_handler = RealTimeInference(config, inference_engine)
        
        # Set initialization time
        last_update_time = datetime.now()
        
        logger.info("API initialized successfully")
        return True
    except Exception as e:
        logger.error(f"API initialization failed: {e}")
        return False


# Dependency for ensuring initialization
async def get_inference_handler():
    """Dependency to ensure inference handler is initialized"""
    if inference_handler is None:
        if not initialize():
            raise HTTPException(status_code=500, detail="Inference engine not initialized")
    return inference_handler


# Background task for updating data
async def update_data_task():
    """Background task for updating data"""
    global latest_data, last_update_time
    
    try:
        # Get latest data
        symbol = config.get('data.symbol', 'BTCUSDT')
        interval = config.get('data.timeframe', '1h')
        limit = config.get('inference.data_limit', 100)
        
        # Fetch data
        data = data_handler.fetch_latest_data(symbol, interval, limit)
        
        # Update global data
        latest_data = data
        last_update_time = datetime.now()
        
        logger.info(f"Data updated: {symbol} {interval}, shape={data.shape}")
        return True
    except Exception as e:
        logger.error(f"Data update failed: {e}")
        return False


# API endpoints
@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get API status"""
    global config, data_handler, inference_handler, last_update_time
    
    # Calculate uptime
    start_time = app.state.start_time if hasattr(app.state, 'start_time') else datetime.now()
    uptime = (datetime.now() - start_time).total_seconds()
    
    # Get data shape if available
    data_shape = list(latest_data.shape) if latest_data is not None else None
    
    # Get model info if available
    model_info = None
    if inference_handler and inference_handler.inference_engine:
        engine = inference_handler.inference_engine
        model_info = {
            'model_path': engine.model_path,
            'device': str(engine.device),
            'metadata': engine.metadata
        }
    
    return StatusResponse(
        status="running",
        config_loaded=config is not None,
        data_handler_initialized=data_handler is not None,
        inference_engine_initialized=inference_handler is not None,
        last_update_time=last_update_time.isoformat() if last_update_time else None,
        data_shape=data_shape,
        model_info=model_info,
        uptime=uptime
    )


@app.post("/update-data", response_model=DataUpdateResponse)
async def update_data(
    request: DataUpdateRequest,
    background_tasks: BackgroundTasks
):
    """Update market data"""
    global last_update_time
    
    start_time = time.time()
    
    # Check if update is needed
    if not request.force_update and last_update_time:
        time_since_update = (datetime.now() - last_update_time).total_seconds() / 60
        min_update_interval = config.get('api.min_update_interval', 5)  # minutes
        
        if time_since_update < min_update_interval:
            return DataUpdateResponse(
                success=True,
                message=f"Data is recent ({time_since_update:.1f} min old). Use force_update=true to override.",
                timestamp=datetime.now().isoformat(),
                latency=time.time() - start_time
            )
    
    # Run update in background
    background_tasks.add_task(update_data_task)
    
    return DataUpdateResponse(
        success=True,
        message="Data update started in background",
        timestamp=datetime.now().isoformat(),
        latency=time.time() - start_time
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    inference_handler: RealTimeInference = Depends(get_inference_handler)
):
    """Generate predictions and trading signals"""
    global latest_data
    
    start_time = time.time()
    
    try:
        # Check if data is available
        if latest_data is None:
            # Try to update data
            success = await update_data_task()
            if not success or latest_data is None:
                raise HTTPException(
                    status_code=500,
                    detail="Data not available. Try /update-data endpoint first."
                )
        
        # Run inference
        result = inference_handler.update(latest_data)
        
        if not result.get('success', False):
            error = result.get('error', 'Unknown inference error')
            raise HTTPException(status_code=500, detail=error)
            
        # Return prediction
        return PredictionResponse(
            success=True,
            signal=result.get('signal', 'NONE'),
            confidence=result.get('confidence', 0.0),
            prediction=result.get('prediction', {}),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            latency=time.time() - start_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return PredictionResponse(
            success=False,
            signal='NONE',
            confidence=0.0,
            prediction={},
            timestamp=datetime.now().isoformat(),
            latency=time.time() - start_time,
            error=str(e)
        )


@app.get("/history")
async def get_history(
    limit: int = 10,
    include_signals: bool = True,
    inference_handler: RealTimeInference = Depends(get_inference_handler)
):
    """Get prediction history"""
    try:
        history = inference_handler.get_prediction_history(limit, include_signals)
        return {
            "success": True,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/consistency")
async def get_consistency(
    window_size: int = 10,
    inference_handler: RealTimeInference = Depends(get_inference_handler)
):
    """Analyze prediction consistency"""
    try:
        analysis = inference_handler.analyze_prediction_consistency(window_size)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing consistency: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    app.state.start_time = datetime.now()
    
    # Initialize in background
    initialize()
    
    # Initialize data
    await update_data_task()


if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run("inference_api:app", host=host, port=port, reload=True)