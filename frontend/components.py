"""
HVAC Frontend Components - Premium Industrial Dashboard
=========================================================
"""

import base64
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from .styles import MAIN_CSS
import streamlit.components.v1 as components
from frontend.utils import load_html_template, set_background


def get_base64_image(image_path: str) -> str:
    """Convert image to base64 string."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def render_page_config():
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="HVAC With DRL Control Capstone Project",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def render_styles():
    """Inject main CSS styles."""
    st.markdown(MAIN_CSS, unsafe_allow_html=True)


def render_background(base_path: str):
    """Render fixed background image."""
    img_path = os.path.join(base_path, "frontend", "brg.jpg")
    css = set_background(img_path)
    if css:
        st.markdown(css, unsafe_allow_html=True)


def render_header():
    """Render premium header bar."""
    st.markdown("""
    <div class="header-container">
        <div>
            <h1 class="header-title">HVAC + RL CONTROL</h1>
            <div class="header-subtitle">Intelligent Building Management System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_control_panel(
    duration_options: List[str],
    on_run,
    on_reset,
    simulation_day: Optional[int] = None,
    simulation_hour: Optional[int] = None,
    realtime_label: Optional[str] = None,
) -> str:
    """Render control panel with duration selector and buttons."""
    st.markdown('''
    <div class="panel-card panel-card--compact">
      <div class="panel-header">
          <div class="panel-icon">🤖</div>
          <div class="panel-label">RL Control Center</div>
      </div>
    ''', unsafe_allow_html=True)
    
    # Create two columns for buttons (wrapped for custom spacing)
    st.markdown('<div class="control-buttons">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="small")
    
    with col1:
        if st.button("▶ Run DDPG", type="primary", width="stretch"):
            on_run()
    
    with col2:
        if st.button("↺ Reset System", type="secondary", width="stretch"):
            on_reset()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display current simulation context
    # Không hiển thị thêm message hướng dẫn, UI giữ tối giản

    if realtime_label is not None:
        st.markdown(
            f"""
            <div class="simulation-info simulation-info--inline simulation-info--realtime">
                <span>Realtime (Asia/Ho_Chi_Minh): <strong>{realtime_label}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

    return duration_options[0] if duration_options else ""


def render_system_panel(
    base_path: str,
    results: Optional[Dict],
    simulation_day: Optional[int] = None,
    simulation_hour: Optional[int] = None,
    hour_snapshot: Optional[Dict[str, float]] = None,
):
    """Render HVAC schematic and metrics panel."""
    st.markdown('''
    <div class="panel-card panel-card--system">
      <div class="panel-header">
          <div class="panel-icon">🔧</div>
          <div class="panel-label">System Status</div>
      </div>
    ''', unsafe_allow_html=True)

    # Initial IAQ (có thể lấy từ warmup nếu được truyền vào results)
    initial_iaq = results.get("initial_iaq", {}) if results else {}
    current_t = float(initial_iaq.get("T_zone", 20.0))
    current_rh = float(initial_iaq.get("RH_zone", 85.0))
    current_co2 = float(initial_iaq.get("CO2_zone", 800.0))
    current_p = float(initial_iaq.get("P_total", 1.2))

    has_new_iaq = False
    if hour_snapshot:
        new_t = hour_snapshot.get("T_zone", current_t)
        new_rh = hour_snapshot.get("RH_zone", current_rh)
        new_co2 = hour_snapshot.get("CO2_zone", current_co2)
        new_p = hour_snapshot.get("P_total", current_p)
        has_new_iaq = True
    elif results and "final_state" in results:
        fs = results["final_state"]
        new_t = fs.get("T_zone", current_t)
        new_rh = fs.get("RH_zone", current_rh)
        new_co2 = fs.get("CO2_zone", current_co2)
        new_p = fs.get("P_total", current_p)
        has_new_iaq = True
    else:
        new_t = new_rh = new_co2 = new_p = None

    # IAQ (đặt LÊN TRÊN hình, kéo hơi sát xuống dưới header)
    simulation_meta_html = ""

    metrics_html = f"""
    <div class="iaq-section" style="display:flex; flex-direction:column; gap:10px; margin-top:6px;">
      {simulation_meta_html}
      <div>
        <div style="font-size:0.8rem; font-weight:600; color:#7C8AA5; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
        Initial IAQ In Zone
        </div>
        <div class="metric-grid">
          <div class="metric-item">
            <div class="metric-val" style="color:#FF6B6B;">{current_t:.1f}°C</div>
            <div class="metric-lbl">Temp</div>
          </div>
          <div class="metric-item">
            <div class="metric-val" style="color:#26C6DA;">{current_rh:.1f}%</div>
            <div class="metric-lbl">RH</div>
          </div>
          <div class="metric-item">
            <div class="metric-val" style="color:#43A047;">{current_co2:.0f}</div>
            <div class="metric-lbl">CO₂</div>
          </div>
          <div class="metric-item">
            <div class="metric-val" style="color:#FFA726;">{current_p:.1f}kW</div>
            <div class="metric-lbl">Power</div>
          </div>
        </div>
      </div>

      {""
      if not has_new_iaq
      else f'''
      <div>
        <div style="font-size:0.8rem; font-weight:600; color:#1F2933; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
          New IAQ (In Zone With DRL Control)
        </div>
        <div class="metric-grid">
          <div class="metric-item">
            <div class="metric-val" style="color:#FF6B6B;">{new_t:.1f}°C</div>
            <div class="metric-lbl">Temp</div>
          </div>
          <div class="metric-item">
            <div class="metric-val" style="color:#26C6DA;">{new_rh:.1f}%</div>
            <div class="metric-lbl">RH</div>
          </div>
          <div class="metric-item">
            <div class="metric-val" style="color:#43A047;">{new_co2:.0f}</div>
            <div class="metric-lbl">CO₂</div>
          </div>
          <div class="metric-item">
            <div class="metric-val" style="color:#FFA726;">{new_p:.1f}kW</div>
            <div class="metric-lbl">Power</div>
          </div>
        </div>
      </div>
      '''}
    </div>
    """

    # Render IAQ trước trong panel-card
    st.markdown(metrics_html, unsafe_allow_html=True)

    # Sau đó mới đến HVAC Image (ở dưới IAQ)
    hvac_img_path = os.path.join(base_path, "frontend", "HVACsys.png")
    hvac_img_b64 = get_base64_image(hvac_img_path)
    if hvac_img_b64:
        render_zoomable_hvac_image(hvac_img_b64)
    else:
        st.warning("HVAC schematic not found.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_zoomable_hvac_image(img_b64: str, iaq_html: str = ""):
    """Zoomable + draggable HVAC schematic with inline IAQ below (single block)."""
    html_code = load_html_template("zoom_ahu.html", {"IMG_B64": img_b64, "IAQ_HTML": iaq_html})
    if not html_code:
        return
    # Iframe height kept compact to reduce white space under the schematic
    components.html(html_code, height=360, scrolling=False)


ACTION_CONFIG = [
    {"name": "uFan", "label": "Supply Fan", "cls": "fan"},
    {"name": "uOA", "label": "Outdoor Air", "cls": "oa"},
    {"name": "uChiller", "label": "Chiller Valve", "cls": "chiller"},
    {"name": "uHeater", "label": "Heater Valve", "cls": "heater"},
    {"name": "uFanEA", "label": "Exhaust Fan", "cls": "fanea"},
]


def render_actions_panel(results: Optional[Dict], default_actions: Dict[str, float]):
    """Render RL actions panel."""
    st.markdown('''
    <div class="panel-card panel-card--compact">
      <div class="panel-header">
          <div class="panel-icon">🤖</div>
          <div class="panel-label">Actions Panel</div>
      </div>
    ''', unsafe_allow_html=True)
    
    # Current (baseline) actions - có thể lấy từ warmup nếu được truyền vào
    current_actions = default_actions.copy()
    if results and "initial_action" in results:
        ia = results.get("initial_action", {})
        for k in current_actions.keys():
            if k in ia:
                current_actions[k] = ia[k]

    # New actions từ RL (nếu có kết quả)
    has_new_actions = bool(results and "final_state" in results)
    new_actions = {}
    if has_new_actions:
        fs = results["final_state"]
        new_actions = {k: fs.get(k, v) for k, v in default_actions.items()}
    
    cards_html = ""
    for cfg in ACTION_CONFIG:
        cur_val = current_actions.get(cfg["name"], 0.0)
        new_col_html = ""
        if has_new_actions and cfg["name"] in new_actions:
            # Không fallback: chỉ hiển thị nếu RL thực sự trả về giá trị cho action này
            new_val = new_actions[cfg["name"]]
            new_col_html = (
                f'<div class="action-col">'
                f'<div class="action-col-label">New</div>'
                f'<div class="action-val action-val-new">{new_val:.2f}</div>'
                f'</div>'
            )

        cards_html += (
            f'<div class="action-row {cfg["cls"]}">'
            f'  <div class="action-name">{cfg["label"]}</div>'
            f'  <div class="action-values">'
            f'    <div class="action-col">'
            f'      <div class="action-col-label">Initial</div>'
            f'      <div class="action-val action-val-current">{cur_val:.2f}</div>'
            f'    </div>'
            f'    {new_col_html}'
            f'  </div>'
            f'</div>'
        )
    
    st.markdown(cards_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_spinner():
    """Render running spinner."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #94a3b8;">
        <div style="margin-bottom: 1rem;">⚡ Processing Control RL in 5m...</div>
    </div>
    """, unsafe_allow_html=True)


def render_visualize_tab(
    results: Optional[Dict],
    trimmed_timeline: Optional[Dict] = None,
    simulation_hour: Optional[int] = None,
):
    """Render visualization tab with charts."""
    if not results or "timeline" not in results:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; opacity: 0.5;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <div style="color: #94a3b8;">No data. Run simulation first.</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    meta = results.get("meta", {})
    duration_hours = meta.get("duration_hours", 1)
    
    # Format duration display
    if duration_hours >= 24:
        days = duration_hours // 24
        duration_text = f"{days} Day{'s' if days > 1 else ''}"
    else:
        duration_text = f"{duration_hours} Hour{'s' if duration_hours > 1 else ''}"
    
    st.markdown(f'''
    <div class="panel-header">
        <div class="panel-icon">📈</div>
        <div class="panel-label">Analytics RL Control</div>
    </div>
    ''', unsafe_allow_html=True)
    
    timeline = trimmed_timeline if trimmed_timeline else results["timeline"]

    # State variables section (không dùng expander để tránh icon text xấu)
    st.markdown(
        """
        <div class="panel-subheader">
            <span>State Variables</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_state_chart(timeline)

    # Control actions section
    st.markdown(
        """
        <div class="panel-subheader" style="margin-top: 0.75rem;">
            <span>Control Actions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_actions_chart(timeline)


def render_state_chart(timeline: Dict):
    """Render state variables chart."""
    time = timeline.get("time", [])

    if not time:
        st.info("No timeline data available for visualization.")
        return

    fig = make_subplots(rows=1, cols=1)

    # Modern smooth lines with light markers
    fig.add_trace(go.Scatter(
        x=time,
        y=timeline.get("T_zone", []),
        name="Temp (°C)",
        mode="lines+markers",
        line=dict(color="#f97373", width=2.2, shape="spline"),
        marker=dict(size=4, color="#fecaca")
    ))

    fig.add_trace(go.Scatter(
        x=time,
        y=timeline.get("RH_zone", []),
        name="Humidity (%)",
        mode="lines+markers",
        line=dict(color="#22c5df", width=2.2, shape="spline"),
        marker=dict(size=4, color="#a5f3fc")
    ))

    co2_scaled = [v / 10 for v in timeline.get("CO2_zone", [])]
    fig.add_trace(go.Scatter(
        x=time,
        y=co2_scaled,
        name="CO₂ (÷10)",
        mode="lines",
        line=dict(color="#34d399", width=2, dash="dot", shape="spline"),
    ))

    power_scaled = [v * 10 for v in timeline.get("P_total", [])]
    fig.add_trace(go.Scatter(
        x=time,
        y=power_scaled,
        name="Power (×10)",
        mode="lines",
        line=dict(color="#facc15", width=2, dash="dash", shape="spline"),
    ))

    # X-axis as integer hours 0,1,2,... (không hiển thị 8.75)
    max_time = max(time)
    max_hour = int(max_time)
    tick_vals = list(range(0, max_hour + 1))
    tick_text = [f"{h}h" for h in tick_vals]

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#94a3b8"),
        xaxis=dict(
            title="Time (hours)",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, width="stretch")


def render_actions_chart(timeline: Dict):
    """Render RL actions chart."""
    time = timeline.get("time", [])

    if not time:
        st.info("No action data available for visualization.")
        return

    fig = go.Figure()
    
    colors = {
        "uFan": "#3b82f6", "uOA": "#06b6d4", "uChiller": "#8b5cf6",
        "uHeater": "#f43f5e", "uFanEA": "#64748b"
    }
    
    for name, color in colors.items():
        if name in timeline:
            fig.add_trace(go.Scatter(
                x=time,
                y=timeline[name],
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2, shape="spline"),
                marker=dict(size=3, color=color),
            ))

    max_time = max(time)
    max_hour = int(max_time)
    tick_vals = list(range(0, max_hour + 1))
    tick_text = [f"{h}h" for h in tick_vals]
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#94a3b8"),
        xaxis=dict(
            title="Time (hours)",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        yaxis=dict(title="Value", range=[-0.05, 1.05], gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, width="stretch")
