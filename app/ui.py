import httpx2
import streamlit as st

from app.core.config import get_settings

st.set_page_config(page_title="RAG Assistant")
st.title("Corporate Expense & Benefits Assistant")
st.caption("Demo policies describe a fictional company. Not tax or legal advice.")

settings = get_settings()


def _fetch_readiness(base_url: str) -> dict:
    response = httpx2.get(f"{base_url}/ready", timeout=5.0)
    response.raise_for_status()
    return response.json()


def _fetch_index_stats(base_url: str) -> dict:
    response = httpx2.get(f"{base_url}/admin/stats", timeout=5.0)
    response.raise_for_status()
    return response.json()


try:
    readiness = _fetch_readiness(settings.api_base_url)
except httpx2.HTTPError as exc:
    st.error(f"Could not reach the API at {settings.api_base_url}: {exc}")
else:
    if readiness["ready"]:
        st.success("Backend is reachable and ready.")
    else:
        st.warning("Backend is reachable but not ready yet.")
    for check in readiness["checks"]:
        st.write(f"- **{check['name']}**: {check['status']} — {check['detail']}")

with st.sidebar:
    st.subheader("Policy index")
    try:
        stats = _fetch_index_stats(settings.api_base_url)
    except httpx2.HTTPError as exc:
        st.warning(f"Index stats unavailable: {exc}")
    else:
        st.metric("Indexed chunks", stats["total_chunks"])
        for category, count in sorted(stats["category_counts"].items()):
            st.write(f"- {category}: {count}")
