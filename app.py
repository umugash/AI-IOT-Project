"""NoiseScope Chennai — Streamlit dashboard.

    streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from dashboard import data as D  # noqa: E402
from dashboard import views  # noqa: E402

st.set_page_config(page_title="NoiseScope Chennai", page_icon="🔊", layout="wide")

pages = [
    st.Page(views.overview, title="Overview", icon="🏙️", url_path="overview", default=True),
    st.Page(views.live, title="Live CPCB network", icon="📡", url_path="live"),
    st.Page(views.patterns, title="Spatiotemporal patterns", icon="🕒", url_path="patterns"),
    st.Page(views.forecast, title="Hotspot prediction", icon="🔮", url_path="forecast"),
    st.Page(views.sound, title="Sound source classification", icon="🎙️", url_path="sound"),
    st.Page(views.edge, title="Edge sensor network", icon="🛰️", url_path="edge"),
    st.Page(views.about, title="How it works", icon="📘", url_path="about"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("**NoiseScope Chennai**  \nAIoT urban noise intelligence")
    if st.button("Reload data"):
        D.refresh()
        st.rerun()
nav.run()
