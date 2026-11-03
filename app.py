from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from streamlit_app.data import (
    ApplicationDataError,
    RAW_DIR,
    ROOT as DATA_ROOT,
    csv_output,
    load_panel,
    read_csv,
    read_json,
)
from streamlit_app.pages import CSV_COLUMNS, PAGE_NAMES, render_page

try:
    from streamlit_app.pages import PAGE_ICONS
except ImportError:
    PAGE_ICONS = ("📊", "🔍", "🔮")


@st.cache_data(show_spinner=False)
def _cached_full_panel():
    return load_panel(None)


def _cached_panel(city: str | None = None):
    full = _cached_full_panel()
    if city:
        return full.loc[full["City"].eq(city)].copy().reset_index(drop=True)
    return full


@st.cache_data(show_spinner=False)
def _cached_csv(name: str):
    return read_csv(csv_output(name), CSV_COLUMNS.get(name))


@st.cache_data(show_spinner=False)
def _cached_json(name: str):
    if name == "pipeline_status":
        path = DATA_ROOT / "outputs" / "india" / "system" / "pipeline_status.json"
    elif name == "eda_summary":
        path = DATA_ROOT / "outputs" / "india" / "eda" / "chennai_eda_summary.json"
    else:
        raise ApplicationDataError(f"Unknown application JSON output: {name}")
    return read_json(path)


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ─── Global ─── */
        .stApp {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        /* ─── Sidebar ─── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1a1f3d 100%);
            border-right: 1px solid rgba(99, 102, 241, 0.15);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            letter-spacing: -0.03em;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            color: #94a3b8;
            font-weight: 400;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
        }

        /* ─── Radio navigation ─── */
        [data-testid="stSidebar"] .stRadio label {
            padding: 0.55rem 0.75rem;
            border-radius: 8px;
            transition: all 0.2s ease;
            font-size: 0.92rem;
            margin-bottom: 2px;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(99, 102, 241, 0.12);
        }
        [data-testid="stSidebar"] .stRadio [data-checked="true"] + label,
        [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
            background: rgba(99, 102, 241, 0.2);
            border-left: 3px solid #6366f1;
        }

        /* ─── Metric cards ─── */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg,
                rgba(30, 41, 59, 0.95) 0%,
                rgba(15, 23, 42, 0.85) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25),
                        inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15),
                        inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8;
            font-size: 0.82rem;
            font-weight: 500;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: #e2e8f0;
            font-weight: 700;
            font-size: 1.5rem;
        }

        /* ─── Headers ─── */
        h1 {
            letter-spacing: -0.035em;
            font-weight: 700;
            background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h2, h3 {
            letter-spacing: -0.025em;
            color: #e2e8f0;
        }

        /* ─── Alerts ─── */
        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid rgba(99, 102, 241, 0.15);
            backdrop-filter: blur(8px);
        }

        /* ─── Tabs & expanders ─── */
        .stTabs [data-baseweb="tab"] {
            font-weight: 500;
            border-radius: 8px 8px 0 0;
        }
        .streamlit-expanderHeader {
            font-weight: 500;
            border-radius: 10px;
        }

        /* ─── Tables & dataframes ─── */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        /* ─── Buttons ─── */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.2s ease;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            border: none;
            color: white;
        }

        /* ─── Multiselect & selectbox ─── */
        .stMultiSelect, .stSelectbox {
            border-radius: 10px;
        }

        /* ─── Dividers ─── */
        hr {
            border-color: rgba(99, 102, 241, 0.15);
        }

        /* ─── Captions ─── */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #64748b;
            font-size: 0.82rem;
        }

        /* ─── Progress bar ─── */
        .stProgress > div > div {
            background: linear-gradient(90deg, #6366f1, #a78bfa);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Chennai Noise Intelligence",
        page_icon="🌆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    with st.sidebar:
        st.markdown("## 🌆 Chennai Noise\n### Intelligence Platform")
        st.caption("AIoT Urban Noise Hotspot Modeling & Spatiotemporal Prediction")
        st.divider()

        page_name = st.radio(
            "Navigate",
            PAGE_NAMES,
            label_visibility="collapsed",
            key="application_navigation",
        )

        st.divider()
        st.markdown(
            "<div style='padding: 10px 14px; background: rgba(99,102,241,0.08); "
            "border-radius: 12px; border: 1px solid rgba(99,102,241,0.2);'>"
            "<span style='color:#a5b4fc;font-size:0.75rem;text-transform:uppercase;"
            "letter-spacing:0.06em;font-weight:700;'>Operational Status</span><br>"
            "<span style='color:#94a3b8;font-size:0.8rem;line-height:1.6;'>"
            "📍 Stations: <b style='color:#e2e8f0;'>10 / 10 CPCB Monitors</b><br>"
            "📡 Live Telemetry: <b style='color:#34d399;'>Active (2026-11-02)</b><br>"
            "📊 Continuous: <b style='color:#818cf8;'>44,383 Observations</b><br>"
            "🗺️ Spatial: <b style='color:#fbbf24;'>Discrete Spatial Nodes</b>"
            "</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("")
        st.caption("College Major Project · AI & Machine Learning")
        st.caption("AIoT Urban Noise Intelligence System")

    context = {
        "panel": _cached_panel,
        "csv": _cached_csv,
        "json": _cached_json,
    }
    try:
        render_page(page_name, context)
    except ApplicationDataError as error:
        st.error(f"Required application data could not be loaded: {error}")
        st.info(
            "Check that the project datasets and generated outputs exist, then run "
            "`.\\.venv\\Scripts\\python.exe scripts\\run_india_system.py`."
        )
    except (OSError, ValueError, KeyError) as error:
        st.error(f"The selected page could not be rendered from the available data: {error}")
    except sqlite3.Error as error:
        st.error(f"The local replay database could not be read: {error}")


if __name__ == "__main__":
    main()
