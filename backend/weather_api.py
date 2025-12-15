"""
WeatherAPI.com Integration
==========================
Lấy real-time weather data từ WeatherAPI.com cho các vùng: DaNang, HaNoi, SaiGon.
"""

import requests
import json
from datetime import datetime
from typing import Dict, Optional, Any
import os
import pandas as pd
import numpy as np


# WeatherAPI.com Configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "ef7e4821f2b7467883075608250712")
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# Region configs (lat/lon/name) for realtime
REGION_CONFIG = {
    "DaNang": {
        "lat": 15.97988,
        "lon": 108.23858,
        "location": "HVAC Company - Hoa Xuan, Cam Le, Da Nang",
    },
    "HaNoi": {
        "lat": 21.04062,
        "lon": 105.74189,
        "location": "HVAC Vietnam Co., Xuan Phuong, Nam Tu Liem, Hanoi",
    },
    "SaiGon": {
        "lat": 10.82547,
        "lon": 106.62809,
        "location": "HVAC Sai Gon, Tan Binh, Ho Chi Minh City",
    },
}


def get_realtime_weather(region: str = "DaNang") -> Dict[str, Any]:
    """
    Lấy real-time weather data từ WeatherAPI.com cho các vùng DaNang, HaNoi, SaiGon.
    
    Returns:
        Dictionary chứa weather data đã parse:
        {
            "source": "WeatherAPI.com",
            "location": "HVAC Company - Hoa Xuan, Cam Le, DaNang",
            "local_time": "2025-12-07 14:30",
            "temperature_c": 28.5,
            "temperature_k": 301.65,
            "humidity_percent": 85.0,
            "wind_kph": 9.0,
            "wind_ms": 2.5,
            "pressure_mb": 1010,
            "pressure_pa": 101000,
            "short_rad_wm2": 0,
            "diff_rad_wm2": 0,
            "wind_degree": 180,
            "current_hour": 14,
            "current_minute": 30
        }
    """
    try:
        cfg = REGION_CONFIG.get(region, REGION_CONFIG["DaNang"])
        lat = cfg["lat"]
        lon = cfg["lon"]
        friendly_loc = cfg["location"]

        params = {
            "key": WEATHER_API_KEY,
            "q": f"{lat},{lon}",
            "aqi": "no"
        }
        
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse location info
        location = data.get("location", {})
        current = data.get("current", {})
        
        # Extract local time
        local_time_str = location.get("localtime", "")
        local_datetime = datetime.strptime(local_time_str, "%Y-%m-%d %H:%M")
        current_hour = local_datetime.hour
        current_minute = local_datetime.minute
        
        # Làm tròn giờ: giữ nguyên giờ hiện tại (0-59 phút đều cùng giờ)
        # Ví dụ: 21h00–21h59 → 21h, 22h00–22h59 → 22h
        rounded_hour = current_hour
        
        # Extract weather data
        temp_c = float(current.get("temp_c", 25.0))
        temp_k = temp_c + 273.15
        humidity = float(current.get("humidity", 70.0))
        wind_kph = float(current.get("wind_kph", 5.0))
        wind_ms = wind_kph / 3.6
        pressure_mb = float(current.get("pressure_mb", 1013.0))
        pressure_pa = pressure_mb * 100
        wind_degree = float(current.get("wind_degree", 0))
        
        # Solar radiation - WeatherAPI không có trực tiếp, dùng 0 cho ban đêm
        # Có thể tính toán dựa trên giờ trong ngày nếu cần
        hour = current_hour
        if 6 <= hour < 18:
            # Giờ ban ngày: ước tính radiation (có thể cải thiện sau)
            short_rad = 0  # WeatherAPI không cung cấp, để 0
            diff_rad = 0
        else:
            short_rad = 0
            diff_rad = 0
        
        result = {
            "source": "WeatherAPI.com",
            "region": region,
            "location": friendly_loc,
            "local_time": local_time_str,
            "temperature_c": round(temp_c, 2),
            "temperature_k": round(temp_k, 2),
            "humidity_percent": round(humidity, 2),
            "wind_kph": round(wind_kph, 2),
            "wind_ms": round(wind_ms, 2),
            "pressure_mb": round(pressure_mb, 1),
            "pressure_pa": round(pressure_pa, 0),
            "short_rad_wm2": short_rad,
            "diff_rad_wm2": diff_rad,
            "wind_degree": round(wind_degree, 0),
            "current_hour": current_hour,
            "current_minute": current_minute,
            "rounded_hour": rounded_hour,  # Giờ đã làm tròn
            "raw_data": data  # Giữ raw data để debug
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"WeatherAPI request failed: {str(e)}")
    except KeyError as e:
        raise Exception(f"WeatherAPI response missing key: {str(e)}")
    except Exception as e:
        raise Exception(f"WeatherAPI error: {str(e)}")


def convert_weather_to_csv_format(weather_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert weather data từ API format sang CSV format để replace vào CSV.
    CHỈ lấy 6 thông số: time, TDryBul, relHum, pAtm, winSpe, winDir
    
    Mapping (dựa trên testapi.ipynb cell 2):
    - time: rounded_hour (giờ giữ nguyên, 0-59 phút cùng giờ) → convert to seconds
    - TDryBul: temperature_k (Kelvin) - từ temp_c + 273.15
    - relHum: humidity_percent / 100 (0-1) - từ humidity %
    - pAtm: pressure_pa (Pascal) - từ pressure_mb × 100
    - winSpe: wind_ms (m/s) - từ wind_kph / 3.6
    - winDir: wind_degree (degree) - từ wind_degree
    
    Lưu ý: HGloHor và HDifHor KHÔNG được include, sẽ giữ nguyên trong CSV
    
    Returns:
        Dictionary với 6 keys: time, TDryBul, relHum, pAtm, winSpe, winDir
    """
    # Sử dụng rounded_hour nếu có, nếu không thì dùng current_hour
    rounded_hour = weather_data.get("rounded_hour", weather_data.get("current_hour", 0))
    time_seconds = rounded_hour * 3600  # Convert hour to seconds
    
    return {
        "time": time_seconds,
        "TDryBul": weather_data["temperature_k"],
        "relHum": weather_data["humidity_percent"] / 100.0,
        "pAtm": weather_data["pressure_pa"],
        "winSpe": weather_data["wind_ms"],
        "winDir": weather_data.get("wind_degree", 0)
    }


def replace_weather_in_csv(
    csv_path: str,
    weather_data: Dict[str, Any],
    target_hour: int,
    base_csv_path: Optional[str] = None
) -> None:
    """
    Tạo file weather realtime bằng cách copy ngày cuối từ base CSV, append làm ngày mới, rồi replace 6 thông số trong ngày append.
    
    Args:
        csv_path: Đường dẫn file output realtime (sẽ bị ghi đè)
        weather_data: Dictionary từ get_realtime_weather()
        target_hour: Giờ cần replace (0-23) - dùng rounded_hour
        base_csv_path: File weather gốc làm chuẩn (mặc định dùng WARMUP_WEATHER_CSV)
    """
    
    df_base = pd.read_csv(base_csv_path).sort_values("time").reset_index(drop=True)
    
    if len(df_base) < 24:
        raise ValueError("Base weather CSV phải có tối thiểu 24 dòng để copy ngày cuối.")
    
    # Suy ra dt (giả định đều)
    time_values = df_base["time"].values
    if len(time_values) > 1:
        dt = float(np.median(np.diff(time_values)))
    else:
        dt = 3600.0
    
    # Tính số dòng trong 1 ngày (24 giờ)
    rows_per_day = int(24 * 3600 / dt) if dt > 0 else 24
    
    # Copy ngày cuối từ base CSV
    last_day_df = df_base.tail(rows_per_day).copy()
    
    # Offset time cho ngày mới (thêm 24 giờ)
    day_offset = 24 * 3600  # 24 hours in seconds
    appended_day = last_day_df.copy()
    appended_day["time"] = appended_day["time"] + day_offset
    
    # Ghép base + ngày mới
    df_rt = pd.concat([df_base, appended_day], ignore_index=True)
    
    # Thời điểm bắt đầu ngày append
    appended_start_time = float(appended_day["time"].min())
    
    # Thời điểm target trong ngày append
    target_time_seconds = appended_start_time + target_hour * 3600
    
    # Tìm closest idx trong ngày append
    mask_appended = df_rt["time"] >= appended_start_time - 1e-6
    time_diff = np.abs(df_rt.loc[mask_appended, "time"].values - target_time_seconds)
    closest_idx_local = int(np.argmin(time_diff))
    closest_idx = df_rt.loc[mask_appended].index[closest_idx_local]
    
    # Convert weather data
    csv_weather = convert_weather_to_csv_format(weather_data)
    
    # Replace 6 thông số trong ngày append
    df_rt.loc[closest_idx, "time"] = target_time_seconds
    df_rt.loc[closest_idx, "TDryBul"] = csv_weather["TDryBul"]
    df_rt.loc[closest_idx, "relHum"] = csv_weather["relHum"]
    df_rt.loc[closest_idx, "pAtm"] = csv_weather["pAtm"]
    df_rt.loc[closest_idx, "winSpe"] = csv_weather["winSpe"]
    df_rt.loc[closest_idx, "winDir"] = csv_weather["winDir"]
    # Giữ nguyên HGloHor, HDifHor
    
    # CUT CSV: Chỉ giữ dữ liệu tới target_hour (không giữ dữ liệu "tương lai")
    # Target time tuyệt đối trong ngày append
    target_time_absolute = appended_start_time + target_hour * 3600
    
    # Cắt: base + appended day với time <= target_time_absolute
    base_max_time = float(df_base['time'].max())
    base_part = df_rt[df_rt['time'] <= base_max_time + 1e-6]
    appended_part_cut = df_rt[(df_rt['time'] > base_max_time + 1e-6) & (df_rt['time'] <= target_time_absolute)]
    
    df_rt_cut = pd.concat([base_part, appended_part_cut], ignore_index=True)
    
    # Ghi đè file realtime (đã cắt)
    df_rt_cut.to_csv(csv_path, index=False)
    
    actual_hour = weather_data.get("current_hour", 0)
    actual_minute = weather_data.get("current_minute", 0)
    rounded_hour = weather_data.get("rounded_hour", actual_hour)
    
    print(f"✓ Built realtime CSV at {csv_path}")
    print(f"  Replace hour {rounded_hour} → time={target_time_seconds}s on appended day")
    print(f"  Cut CSV to time={target_time_absolute}s ({rounded_hour}:00)")
    print(f"  Rows: {len(df_rt)} → {len(df_rt_cut)} (base: {len(base_part)}, appended: {len(appended_part_cut)})")
    print(f"  Actual time: {actual_hour:02d}:{actual_minute:02d} → Rounded: {rounded_hour:02d}:00")


def test_weather_api() -> bool:
    """
    Test WeatherAPI connection.
    
    Returns:
        True nếu API hoạt động, False nếu có lỗi
    """
    try:
        weather = get_realtime_weather()
        print(f"✓ WeatherAPI test successful")
        print(f"  Location: {weather['location']}")
        print(f"  Temperature: {weather['temperature_c']}°C")
        print(f"  Humidity: {weather['humidity_percent']}%")
        return True
    except Exception as e:
        print(f"✗ WeatherAPI test failed: {e}")
        return False


if __name__ == "__main__":
    # Test weather API
    print("Testing WeatherAPI.com integration...")
    test_weather_api()

