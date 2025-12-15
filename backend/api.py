"""
FastAPI backend cho HVAC RL + FMU
=================================
Expose API realtime prediction + health check.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict
from datetime import datetime
import traceback
import os
import sys

# Add parent directory to path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)

from backend.rl_engine import (
    predict_single_point,
    FMU_PATH,
    get_region_config,
)
from backend.weather_api import get_realtime_weather


app = FastAPI(title="HVAC RL+FMU API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=Dict[str, Any])
def health_check(region: str = "DaNang"):
    """Health check endpoint"""
    try:
        cfg = get_region_config(region)
        model_path = cfg["model"]
        models_ok = os.path.exists(FMU_PATH) and os.path.exists(model_path)
        
        return {
            "status": "healthy" if models_ok else "degraded",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "fmu_loaded": os.path.exists(FMU_PATH),
                "rl_model_loaded": os.path.exists(model_path),
                "model_path": model_path,
                "region": region,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather-only", response_model=Dict[str, Any])
def weather_only(region: str = "DaNang"):
    """Chỉ lấy weather data từ API, không predict"""
    try:
        weather = get_realtime_weather(region)
        return {
            "status": "success",
            "weather": weather,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"WeatherAPI error: {str(e)}"
        )


@app.post("/api/predict-realtime-sync", response_model=Dict[str, Any])
def predict_realtime_sync(region: str = "DaNang"):
    """
    Main endpoint: Predict real-time với weather từ WeatherAPI.com
    
    Quy trình 7 bước:
    1. Lấy data từ WeatherAPI.com (real-time)
    2. Extract current hour từ API
    3. Replace weather vào CSV tại giờ tương ứng
    4. Load FMU model
    5. Run warmup (5 giờ)
    6. Run FMU tại thời điểm đó với real-time weather
    7. Agent predict action từ state
    8. Trả ra JSON output
    """
    start_time = datetime.now()
    
    try:
        # STEP 1: Lấy data từ WeatherAPI.com
        print("STEP 1: Fetching real-time weather from WeatherAPI.com...")
        weather_data = get_realtime_weather(region)
        
        # STEP 2: Extract current hour
        current_hour = weather_data.get("current_hour", 0)
        current_minute = weather_data.get("current_minute", 0)
        current_time_str = f"{current_hour:02d}:{current_minute:02d}"
        
        print(f"STEP 2: Current hour = {current_hour}, time = {current_time_str}")
        
        # STEP 3-7: Predict single point
        print("STEP 3-7: Running prediction...")
        prediction_result = predict_single_point(weather_data, region=region)
        
        # Build full response
        process_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "current_hour": current_hour,
            "current_minute": current_minute,
            "current_time": current_time_str,
            
            "prediction": {
                "action": prediction_result["action"],
                "action_names": prediction_result["action_names"],
                "action_dict": prediction_result["action_dict"],
                "confidence": 0.95  # Default confidence
            },
            
            "weather": weather_data,
            
            "fmu_state": prediction_result["fmu_state"],
            
            "processing": {
                "warmup_hours": prediction_result["processing"]["warmup_hours"],
                "process_time_ms": round(process_time_ms, 2),
                "timestamp_generated": datetime.now().isoformat()
            }
        }
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in predict_realtime_sync: {error_msg}")
        print(traceback_str)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error": error_msg,
                "traceback": traceback_str,
                "timestamp": datetime.now().isoformat()
            }
        )


# Note: Frontend is served via Streamlit (backend/app.py), not FastAPI


