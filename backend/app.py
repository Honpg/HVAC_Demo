"""
HVAC + RL Demo - Main Application
==================================
"""

import sys
import os
from typing import Dict

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)

# History directory/file
HISTORY_DIR = os.path.join(BASE_PATH, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")

import streamlit as st
try:
    from streamlit import st_autorefresh
except ImportError:
    st_autorefresh = None
from datetime import datetime
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from frontend import (
    render_page_config,
    render_styles,
    render_background,
    render_system_panel,
    render_actions_panel,
)
from backend.rl_engine import predict_single_point, get_initial_baseline
from backend.weather_api import get_realtime_weather


SIM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def compute_simulation_hour() -> int:
    now = datetime.now(SIM_TZ)
    hour = now.hour
    if now.minute >= 30:
        hour = min(23, hour + 1)
    return hour


def get_realtime_label() -> str:
    now = datetime.now(SIM_TZ)
    return now.strftime("%Hh%M")


# Real-time prediction history (stored in session state)
HISTORY_KEY = "prediction_history"
DEFAULT_ACTIONS = {
    "uFan": 0.50,
    "uOA": 0.30,
    "uChiller": 0.40,
    "uHeater": 0.00,
    "uFanEA": 0.50,
}

REGION_OPTIONS = ["DN", "HN", "SG"]


def ensure_session_state():
    # Region first so các state phía dưới dùng đúng vùng
    if "region" not in st.session_state:
        st.session_state.region = "DN"
    if "initial_baseline" not in st.session_state:
        # Warmup baseline IAQ/Action tại giờ hiện tại; lỗi => dict rỗng
        current_hour = datetime.now(SIM_TZ).hour
        st.session_state.initial_baseline = get_initial_baseline(
            target_hour=current_hour,
            region=st.session_state.region,
        )
        st.session_state.initial_baseline_region = st.session_state.region
    if "realtime_prediction" not in st.session_state:
        st.session_state.realtime_prediction = None
    if "realtime_weather" not in st.session_state:
        st.session_state.realtime_weather = None
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = load_history_from_disk()


def ensure_baseline_for_region(region: str):
    """Rebuild initial baseline when region changes."""
    region_key = region.upper() if isinstance(region, str) else "DN"
    if st.session_state.get("initial_baseline_region") != region_key:
        current_hour = datetime.now(SIM_TZ).hour
        st.session_state.initial_baseline = get_initial_baseline(
            target_hour=current_hour,
            region=region_key,
        )
        st.session_state.initial_baseline_region = region_key
        # Clear realtime cache để tránh lẫn vùng
        st.session_state.realtime_prediction = None
        st.session_state.realtime_weather = None


def load_history_from_disk() -> list:
    """Load history from history/history.json if available."""
    try:
        if not os.path.exists(HISTORY_DIR):
            os.makedirs(HISTORY_DIR, exist_ok=True)
        if os.path.exists(HISTORY_FILE):
            import json
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    cleaned = []
                    for entry in data:
                        if isinstance(entry, dict):
                            e = dict(entry)
                            ts_raw = e.get("timestamp", "")
                            if isinstance(ts_raw, str):
                                ts = ts_raw.replace("key", "").replace("KEY", "")
                                ts = ts.split(".")[0].replace("T", " ").strip()
                                e["timestamp"] = ts
                            cleaned.append(e)
                        else:
                            cleaned.append(entry)
                    # Rewrite file nếu có thay đổi để tránh prefix "key" tái xuất hiện
                    try:
                        with open(HISTORY_FILE, "w", encoding="utf-8") as wf:
                            json.dump(cleaned, wf, indent=2)
                    except Exception:
                        pass
                    return cleaned
        return []
    except Exception as e:
        print(f"⚠️ Failed to load history: {e}")
        return []


def save_history_to_disk(history: list) -> None:
    """Persist history to history/history.json."""
    try:
        if not os.path.exists(HISTORY_DIR):
            os.makedirs(HISTORY_DIR, exist_ok=True)
        import json
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save history: {e}")


def main():
    render_page_config()

    # Auto refresh every 60s để cập nhật realtime clock (nếu Streamlit hỗ trợ)
    if st_autorefresh:
        st_autorefresh(interval=30_000, key="realtime_clock")

    # Session state
    ensure_session_state()

    # Render
    render_styles()
    render_background(BASE_PATH)
    # Ensure baseline aligns with current region (safety in case region was changed elsewhere)
    ensure_baseline_for_region(st.session_state.region)
    
    # Header with integrated tabs
    col_header = st.container()
    with col_header:
        st.markdown('<div class="header-bar" style="display: flex; align-items: center; gap: 0.5rem;"><span class="header-icon" style="font-size: 1.4rem;">🏭</span><h1 class="header-title" style="margin: 0;">HVAC With DRL Control System</h1></div>', unsafe_allow_html=True)
    
    # Tabs (will be positioned inside header via CSS)
    tab_dashboard, tab_realtime, tab_visualize = st.tabs(["📊 Dashboard", "⚙️ Details Control", "🗂️ History"])
    
    # KHÔNG CÒN FETCH 60-DAY SIMULATIONS - CHỈ REAL-TIME

    # Dashboard Tab - CHỈ REAL-TIME, KHÔNG CÓ FALLBACK
    with tab_dashboard:
        col_left, col_center, col_right = st.columns([1, 3.5, 1])
        
        with col_left:
            # Systems Control panel - đồng bộ với System Status và Actions Panel
            st.markdown('''
            <div class="panel-card panel-card--compact">
              <div class="panel-header">
                  <div class="panel-icon">⚙️</div>
                  <div class="panel-label">Systems Control</div>
              </div>
            ''', unsafe_allow_html=True)
            
            # Region selector
            region = st.selectbox(
                "Khu vực",
                options=REGION_OPTIONS,
                index=REGION_OPTIONS.index(st.session_state.region),
                key="region_select_dashboard",
            )
            st.session_state.region = region
            ensure_baseline_for_region(region)

            if st.button("Run DRL Control", type="primary", width="stretch", key="predict_dashboard"):
                with st.spinner("Fetching real time weather and control..."):
                    try:
                        # Get real-time weather
                        weather_data = get_realtime_weather(region)
                        st.session_state.realtime_weather = weather_data
                        
                        # Predict
                        prediction_result = predict_single_point(weather_data, region=region)
                        st.session_state.realtime_prediction = prediction_result
                        
                        # Save to history
                        ts_now = datetime.now().isoformat()
                        # Clean timestamp để tránh prefix "key"
                        ts_now_clean = ts_now.replace("key", "").replace("KEY", "")
                        ts_now_clean = ts_now_clean.split(".")[0].replace("T", " ").strip()
                        entry = {
                            "time": f"{weather_data['current_hour']:02d}:{weather_data['current_minute']:02d}",
                            "hour": weather_data['current_hour'],
                            "timestamp": ts_now_clean,
                            "region": region,
                            "action": prediction_result["action"],
                            "action_names": prediction_result["action_names"],
                            "action_dict": prediction_result["action_dict"],
                            "weather": {
                                "temperature_c": weather_data["temperature_c"],
                                "humidity_percent": weather_data["humidity_percent"],
                                "wind_kph": weather_data["wind_kph"],
                                "pressure_mb": weather_data["pressure_mb"],
                                "location": weather_data["location"],
                                "local_time": weather_data["local_time"]
                            },
                            "fmu_state": prediction_result["fmu_state"],
                            "initial": prediction_result.get("initial", {}),
                            "processing": prediction_result["processing"]
                        }
                        st.session_state[HISTORY_KEY].append(entry)
                        save_history_to_disk(st.session_state[HISTORY_KEY])
                        st.success("✓ Prediction completed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}")
            
            # Real-time label
            st.markdown(f"**Realtime:** {get_realtime_label()} | Region: {st.session_state.region}")

            # Reset về trạng thái ban đầu cho Dashboard
            if st.button("Reset to Initial", width="stretch", key="reset_dashboard"):
                st.session_state.realtime_prediction = None
                st.session_state.realtime_weather = None
                st.rerun()
            
            # Đóng panel-card
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_center:
            # CHỈ hiển thị từ Real-time Prediction
            realtime_prediction = st.session_state.get("realtime_prediction")
            if realtime_prediction:
                # Extract newIAQ từ fmu_state.fmu_outputs (actual FMU outputs)
                fmu_state = realtime_prediction.get("fmu_state", {})
                fmu_outputs = fmu_state.get("fmu_outputs", {})
                
                # Tạo hour_snapshot từ fmu_outputs
                new_iaq_snapshot = {}
                if fmu_outputs:
                    try:
                        # Extract từ fmu_outputs (đã được convert đúng)
                        t_zone_k = fmu_outputs.get("T_zone_K", 293.15)
                        new_iaq_snapshot["T_zone"] = t_zone_k - 273.15  # Convert K to °C
                        
                        rh_ratio = fmu_outputs.get("RH_zone_ratio", 0.5)
                        new_iaq_snapshot["RH_zone"] = rh_ratio * 100  # Convert to percent
                        
                        new_iaq_snapshot["CO2_zone"] = fmu_outputs.get("CO2_zone_ppm", 500.0)
                        
                        p_total_w = fmu_outputs.get("P_total_W", 0.0)
                        new_iaq_snapshot["P_total"] = p_total_w / 1000.0  # Convert W to kW
                    except (ValueError, TypeError):
                        new_iaq_snapshot = None
                else:
                    new_iaq_snapshot = None
                
                # Tạo results dict format để render_system_panel (bao gồm initial_iaq từ warmup)
                initial_iaq = realtime_prediction.get("initial", {}).get("iaq", {})
                realtime_results = {
                    "final_state": new_iaq_snapshot if new_iaq_snapshot else {},
                    "initial_iaq": initial_iaq,
                    "region": st.session_state.region,
                }

                render_system_panel(
                    BASE_PATH,
                    realtime_results,
                    simulation_day=None,
                    simulation_hour=None,
                    hour_snapshot=new_iaq_snapshot,
                )
            else:
                # Chưa có real-time prediction: hiển thị Initial IAQ từ warmup baseline
                baseline = st.session_state.get("initial_baseline", {})
                render_system_panel(
                    BASE_PATH,
                    {"initial_iaq": baseline.get("iaq", {})},
                    simulation_day=None,
                    simulation_hour=None,
                    hour_snapshot=None,
                )
        
        with col_right:
            # CHỈ hiển thị actions từ Real-time Prediction
            realtime_prediction = st.session_state.get("realtime_prediction")
            if realtime_prediction:
                # Tạo results dict với actions từ real-time prediction + initial action từ warmup
                action_dict = realtime_prediction.get("action_dict", {})
                initial_action = realtime_prediction.get("initial", {}).get("action_init", {})
                realtime_results = {
                    "final_state": action_dict,
                    "initial_action": initial_action,
                    "region": st.session_state.region,
                }
                render_actions_panel(realtime_results, DEFAULT_ACTIONS)
            else:
                # Chưa có real-time prediction: hiển thị Initial Actions từ warmup baseline
                baseline = st.session_state.get("initial_baseline", {})
                render_actions_panel({"initial_action": baseline.get("action_init", {})}, DEFAULT_ACTIONS)
    
    # Real-time Prediction Tab
    with tab_realtime:
        render_realtime_prediction_tab()
    
    # Visualize Tab
    with tab_visualize:
        render_history_tab()


def render_realtime_prediction_tab():
    """Render Details Control (visual-only: Weather + Prediction results + charts)."""
    # Section options
    section_options = [
        "🌤️ Real Time Weather",
        "📊 Real Time DRL Control",
        "📈 Visulize Control Chart - New IAQ",
        "📈 Visulize Control Chart - New Action"
    ]
    
    # Multiselect để chọn sections muốn hiển thị
    selected_sections = st.multiselect(
        "Select sections to display:",
        options=section_options,
        default=["🌤️ Real Time Weather"],  # Mặc định chỉ chọn Real Time Weather
        key="details_control_sections"
    )
    
    if not selected_sections:
        st.info("Please select at least one section to display.")
        return
    
    # Render Real Time Weather
    if "🌤️ Real Time Weather" in selected_sections:
        st.markdown("#### 🌤️ Real Time Weather")
        if st.session_state.realtime_weather:
            weather = st.session_state.realtime_weather
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Location**")
                st.markdown(f"<div style='font-size:1.05rem; line-height:1.4;'>{weather['location']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Time**")
                st.markdown(f"{weather['local_time']}")
            with col2:
                st.metric("Temperature", f"{weather['temperature_c']}°C")
                st.metric("Temperature (K)", f"{weather['temperature_k']}K")
            with col3:
                st.metric("Humidity", f"{weather['humidity_percent']}%")
                st.metric("Wind Speed", f"{weather['wind_kph']} km/h")
            with col4:
                st.metric("Pressure", f"{weather['pressure_mb']} mb")
                st.metric("Source", weather["source"])
        else:
            st.info("Realtime weather data is not available yet.")
        st.markdown("---")
    
    # Render Real Time DRL Control
    if "📊 Real Time DRL Control" in selected_sections:
        st.markdown("#### 📊 Real Time DRL Control")
        if st.session_state.realtime_prediction:
            pred = st.session_state.realtime_prediction
            fmu = pred.get("fmu_state", {}).get("fmu_outputs", {})
            region = pred.get("weather", {}).get("region", st.session_state.region)
            st.markdown(
                f"<div style='font-weight:600; color:#1F2933; margin-bottom:0.25rem;'>Region: <span style='color:#0ea5e9'>{region}</span></div>",
                unsafe_allow_html=True,
            )

            # Initial IAQ snapshot
            initial_iaq = pred.get("initial", {}).get("iaq", {})
            st.markdown("**Initial IAQ:**")
            initial_iaq_snapshot = {
                "Temp (°C)": initial_iaq.get("T_zone", 25.0),
                "RH (%)": initial_iaq.get("RH_zone", 50.0),
                "CO₂ (ppm)": initial_iaq.get("CO2_zone", 500),
                "Power (kW)": initial_iaq.get("P_total", 0.0),
            }
            initial_iaq_cols = st.columns(4)
            for i, (lbl, val) in enumerate(initial_iaq_snapshot.items()):
                with initial_iaq_cols[i]:
                    st.metric(lbl, f"{val:.2f}" if isinstance(val, (float, int)) else val)

            # Initial Action values
            initial_action = pred.get("initial", {}).get("action_init", {})
            st.markdown("**Initial Action:**")
            initial_act_cols = st.columns(5)
            for i, name in enumerate(["uFan", "uOA", "uChiller", "uHeater", "uFanEA"]):
                with initial_act_cols[i]:
                    v = initial_action.get(name, "")
                    st.metric(name, f"{v:.3f}" if isinstance(v, (float, int)) else v)

            st.markdown("---")
            
            # New IAQ snapshot
            new_iaq_snapshot = {
                "Temp (°C)": fmu.get("T_zone_K", 293.15) - 273.15,
                "RH (%)": fmu.get("RH_zone_percent", fmu.get("RH_zone_ratio", 0.5) * 100),
                "CO₂ (ppm)": fmu.get("CO2_zone_ppm", 500),
                "Power (kW)": fmu.get("P_total_W", 0.0) / 1000.0,
            }
            st.markdown("**New IAQ (With RL Control):**")
            iaq_cols = st.columns(4)
            for i, (lbl, val) in enumerate(new_iaq_snapshot.items()):
                with iaq_cols[i]:
                    st.metric(lbl, f"{val:.2f}")

            # New Action values
            st.markdown("**New Action:**")
            action_dict = pred.get("action_dict", {})
            act_cols = st.columns(5)
            for i, name in enumerate(["uFan", "uOA", "uChiller", "uHeater", "uFanEA"]):
                with act_cols[i]:
                    v = action_dict.get(name, "")
                    st.metric(name, f"{v:.3f}" if isinstance(v, (float, int)) else v)
            st.markdown("---")
        else:
            st.info("The DRL control results are not available yet.")
            st.markdown("---")
    
    # Visualize Control Chart: New IAQ
    if "📈 Visulize Control Chart - New IAQ" in selected_sections:
        history_all = st.session_state[HISTORY_KEY]
        history = [h for h in history_all if h.get("region", "DN") == st.session_state.region]
        if history:
            st.markdown(f"#### 📈 Visulize Control Chart - New IAQ (Region: {st.session_state.region})")
            # IAQ chart
            times = [h.get("timestamp") for h in history]
            fmu_list = [h.get("fmu_state", {}).get("fmu_outputs", {}) for h in history]
            iaq_fig = go.Figure()
            iaq_series = {
                "Temp (°C)": [f.get("T_zone_K", 293.15) - 273.15 for f in fmu_list],
                "RH (%)": [f.get("RH_zone_percent", f.get("RH_zone_ratio", 0.5) * 100) for f in fmu_list],
                "CO₂ (ppm)": [f.get("CO2_zone_ppm", 500) for f in fmu_list],
                "Power (kW)": [f.get("P_total_W", 0.0) / 1000.0 for f in fmu_list],
            }
            colors = {
                "Temp (°C)": "#f97316",
                "RH (%)": "#0ea5e9",
                "CO₂ (ppm)": "#22c55e",
                "Power (kW)": "#a855f7",
            }
            for name, vals in iaq_series.items():
                iaq_fig.add_trace(go.Scatter(
                    x=times,
                    y=vals,
                    name=name,
                    mode="lines+markers",
                    line=dict(color=colors.get(name, "#000"), width=2),
                    marker=dict(size=6, color=colors.get(name, "#000"))
                ))
            iaq_fig.update_layout(
                height=360,
                xaxis=dict(title="Time"),
                yaxis=dict(title="Value"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(iaq_fig, width="stretch")
            # Divider với nét đứt và màu mờ giữa các chart
            st.markdown('<div class="chart-divider"></div>', unsafe_allow_html=True)
        else:
            st.info(f"Chưa có history để visualize New IAQ chart cho vùng {st.session_state.region}.")
            st.markdown('<div class="chart-divider"></div>', unsafe_allow_html=True)
    
    # Visualize Control Chart: New Action
    if "📈 Visulize Control Chart - New Action" in selected_sections:
        history_all = st.session_state[HISTORY_KEY]
        history = [h for h in history_all if h.get("region", "DN") == st.session_state.region]
        if history:
            st.markdown(f"#### 📈 Visulize Control Chart - New Action (Region: {st.session_state.region})")
            # Action chart
            times = [h.get("timestamp") for h in history]
            action_names = ["uFan", "uOA", "uChiller", "uHeater", "uFanEA"]
            action_fig = go.Figure()
            action_colors = {
                "uFan": "#3b82f6",
                "uOA": "#06b6d4",
                "uChiller": "#8b5cf6",
                "uHeater": "#f43f5e",
                "uFanEA": "#64748b"
            }
            for name in action_names:
                vals = []
                for entry in history:
                    vals.append(entry.get("action_dict", {}).get(name, None))
                action_fig.add_trace(go.Scatter(
                    x=times,
                    y=vals,
                    name=name,
                    mode="lines+markers",
                    line=dict(color=action_colors.get(name, "#000"), width=2),
                    marker=dict(size=6, color=action_colors.get(name, "#000"))
                ))
            action_fig.update_layout(
                height=360,
                xaxis=dict(title="Time"),
                yaxis=dict(title="Action Value", range=[0, 1]),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(action_fig, width="stretch")
        else:
            st.info(f"Chưa có history để visualize New Action chart cho vùng {st.session_state.region}.")


def render_history_tab():
    """Render full history list persisted under history/history.json."""
    st.markdown("### 🗂️ History")
    history = st.session_state.get(HISTORY_KEY, [])

    # Header với multiselect và Clear button
    col_select, col_clear = st.columns([7, 1])
    with col_select:
        if history:
            # Tạo danh sách options cho multiselect
            history_options = []
            for idx, entry in enumerate(reversed(history)):
                ts_raw = entry.get("timestamp", "")
                hour = entry.get("hour", "")
                region = entry.get("region", "DN")
                ts = ""
                if isinstance(ts_raw, str) and ts_raw:
                    ts = ts_raw.replace("key", "").replace("KEY", "")
                    ts = ts.split(".")[0].replace("T", " ").strip()
                title = f"{idx+1}|{ts if ts else 'Entry'} | Hour {hour} | Region {region}"
                history_options.append(title)
            
            selected_entries = st.multiselect(
                "Select history entries to display:",
                options=history_options,
                default=history_options,  # Mặc định chọn tất cả
                key="history_entries_select"
            )
        else:
            selected_entries = []
            st.info("Chưa có lịch sử dự đoán.")
    
    with col_clear:
        st.markdown('<div style="padding-top: 2rem;"></div>', unsafe_allow_html=True)  # Căn chỉnh theo chiều dọc
        if st.button("🗑️ Clear History", width="stretch", key="clear_history_tab"):
            st.session_state[HISTORY_KEY] = []
            st.session_state.realtime_prediction = None
            st.session_state.realtime_weather = None
            try:
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
            except OSError:
                pass
            st.rerun()

    if not history or not selected_entries:
        return

    st.markdown("---")
    # Tạo mapping từ title về index trong reversed history
    title_to_idx = {}
    for idx, entry in enumerate(reversed(history)):
        ts_raw = entry.get("timestamp", "")
        hour = entry.get("hour", "")
        region = entry.get("region", "DN")
        ts = ""
        if isinstance(ts_raw, str) and ts_raw:
            ts = ts_raw.replace("key", "").replace("KEY", "")
            ts = ts.split(".")[0].replace("T", " ").strip()
        title = f"{idx+1}|{ts if ts else 'Entry'} | Hour {hour} | Region {region}"
        title_to_idx[title] = idx
    
    # Chỉ hiển thị các entries được chọn - tự động expand khi được chọn
    for selected_title in selected_entries:
        if selected_title in title_to_idx:
            idx = title_to_idx[selected_title]
            entry = list(reversed(history))[idx]
            
            ts_raw = entry.get("timestamp", "")
            hour = entry.get("hour", "")
            ts = ""
            if isinstance(ts_raw, str) and ts_raw:
                ts = ts_raw.replace("key", "").replace("KEY", "")
                ts = ts.split(".")[0].replace("T", " ").strip()
            title = f"{idx+1}|{ts if ts else 'Entry'} | Hour {hour}"

            # Tự động hiển thị nội dung khi được chọn trong multiselect
            st.markdown(f"#### {title}")
            weather = entry.get("weather", {})
            st.markdown("**Weather (realtime):**")
            # Hiển thị Location/Time đầy đủ, không cắt
            st.markdown(f"- **Region:** {entry.get('region', 'DN')}")
            st.markdown(f"- **Location:** {weather.get('location', '')}")
            st.markdown(f"- **Time:** {weather.get('local_time', '')}")

            wcols = st.columns(4)
            witems = [
                ("Temp (°C)", weather.get("temperature_c", "")),
                ("Humidity (%)", weather.get("humidity_percent", "")),
                ("Wind (km/h)", weather.get("wind_kph", "")),
                ("Pressure (mb)", weather.get("pressure_mb", "")),
            ]
            for i, (lbl, val) in enumerate(witems):
                with wcols[i % 4]:
                    st.metric(lbl, val)

            st.markdown("---")
            st.markdown("**Initial IAQ:**")
            initial_iaq = entry.get("initial", {}).get("iaq", {})
            initial_iaq_cols = st.columns(4)
            initial_iaq_items = [
                ("Temp (°C)", initial_iaq.get("T_zone", 25.0)),
                ("RH (%)", initial_iaq.get("RH_zone", 50.0)),
                ("CO₂ (ppm)", initial_iaq.get("CO2_zone", 500)),
                ("Power (kW)", initial_iaq.get("P_total", 0.0)),
            ]
            for i, (lbl, val) in enumerate(initial_iaq_items):
                with initial_iaq_cols[i]:
                    st.metric(lbl, f"{val:.2f}" if isinstance(val, (float, int)) else val)

            st.markdown("**Initial Action:**")
            initial_action = entry.get("initial", {}).get("action_init", {})
            initial_act_cols = st.columns(5)
            for i, name in enumerate(["uFan", "uOA", "uChiller", "uHeater", "uFanEA"]):
                with initial_act_cols[i]:
                    v = initial_action.get(name, "")
                    st.metric(name, f"{v:.3f}" if isinstance(v, (float, int)) else v)

            st.markdown("---")
            st.markdown("**New IAQ (FMU outputs):**")
            fmu = entry.get("fmu_state", {}).get("fmu_outputs", {})
            iaq_cols = st.columns(4)
            iaq_items = [
                ("Temp (°C)", fmu.get("T_zone_K", 293.15) - 273.15),
                ("RH (%)", fmu.get("RH_zone_ratio", 0.5) * 100),
                ("CO₂ (ppm)", fmu.get("CO2_zone_ppm", 500)),
                ("Power (kW)", fmu.get("P_total_W", 0.0) / 1000.0),
            ]
            for i, (lbl, val) in enumerate(iaq_items):
                with iaq_cols[i]:
                    st.metric(lbl, f"{val:.2f}" if isinstance(val, (float, int)) else val)

            st.markdown("---")
            st.markdown("**New Action:**")
            action_dict = entry.get("action_dict", {})
            act_cols = st.columns(5)
            for i, name in enumerate(["uFan", "uOA", "uChiller", "uHeater", "uFanEA"]):
                with act_cols[i]:
                    v = action_dict.get(name, "")
                    st.metric(name, f"{v:.3f}" if isinstance(v, (float, int)) else v)
            
            st.markdown("---")  # Phân cách giữa các entries


if __name__ == "__main__":
    main()
