"""
Streamlit UI for the CrewAI Research Team.
"""

import os
import streamlit as st

# ---------------------------------------------------------------------------
# SQLite workaround for Streamlit Community Cloud (ChromaDB requirement)
# ---------------------------------------------------------------------------
# CrewAI uses ChromaDB for memory/embeddings. Streamlit Cloud ships an old
# SQLite (< 3.35.0) that Chroma rejects. This swaps in pysqlite3-binary.
# Only matters on Streamlit Cloud; harmless locally if pysqlite3 is missing.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # local environment already has a modern SQLite


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Multi-Agent Team",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Research Multi-Agent Team")
st.caption("Powered by CrewAI + Groq (GPT-OSS 120B)")


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
def get_groq_api_key() -> str | None:
    """Read Groq API key from Streamlit secrets or environment variable."""
    # Priority 1: Streamlit secrets (deployed app)
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass

    # Priority 2: environment variable (local development)
    key = os.environ.get("GROQ_API_KEY")
    return key if key else None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "report" not in st.session_state:
    st.session_state.report = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    depth = st.radio(
        "Report depth",
        options=["Brief", "Detailed"],
        index=0,
        help="Brief: ~800 words. Detailed: ~1500 words.",
    )
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "Four agents work sequentially:\n"
        "1. 🔍 Research Analyst\n"
        "2. 🧩 Data Synthesizer\n"
        "3. ✍️ Report Writer\n"
        "4. ✏️ Editor\n\n"
        "Each agent builds on the previous one's output."
    )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
topic = st.text_input(
    "Research topic",
    placeholder="e.g., Impact of AI agents on software engineering in 2025",
)

col1, col2 = st.columns([1, 4])
with col1:
    run_clicked = st.button("🔎 Run Research", type="primary", use_container_width=True)

# -- Error handling: missing API key ----------------------------------------
api_key = get_groq_api_key()
if not api_key:
    st.error(
        "**GROQ_API_KEY not found.**\n\n"
        "**Local:** create `.streamlit/secrets.toml` with "
        "`GROQ_API_KEY = \"gsk_...\"` or set the environment variable.\n\n"
        "**Streamlit Cloud:** go to App settings → Secrets and add the key."
    )
    st.stop()

# -- Run the crew -----------------------------------------------------------
if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    # Make the key available to CrewAI/LiteLLM
    os.environ["GROQ_API_KEY"] = api_key

    # Depth instruction passed into the task prompts
    if depth == "Brief":
        depth_instruction = "Keep the report concise, around 800 words."
    else:
        depth_instruction = "Be thorough, aiming for around 1500 words."

    # Import crew here so it's only loaded when needed (faster first paint)
    from crew import create_research_crew

    with st.spinner("Agents are researching... this may take 1-3 minutes."):
        try:
            crew = create_research_crew()
            result = crew.kickoff(
                inputs={
                    "topic": topic.strip(),
                    "depth_instruction": depth_instruction,
                }
            )

            # CrewAI returns a CrewOutput; `.raw` holds the text
            report_text = result.raw if hasattr(result, "raw") else str(result)
            st.session_state.report = report_text

        except Exception as e:
            error_msg = str(e)
            # Friendly messages for common failure modes
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                st.error(
                    "**Groq rate limit hit.** The free tier allows ~8,000 TPM "
                    "for `gpt-oss-120b`. Wait a minute and retry, or switch "
                    "report depth to Brief."
                )
            elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                st.error("**Invalid API key.** Check your GROQ_API_KEY.")
            elif "timeout" in error_msg.lower():
                st.error("**Request timed out.** Try a simpler topic or shorter depth.")
            else:
                st.error(f"**Research failed:** {error_msg}")

            with st.expander("Technical details"):
                st.exception(e)

# -- Display report ---------------------------------------------------------
if st.session_state.report:
    st.divider()
    st.subheader("📄 Final Report")
    st.markdown(st.session_state.report)

    # Download button
    st.download_button(
        label="⬇️ Download as Markdown",
        data=st.session_state.report,
        file_name=(
            f"research_{topic[:30].replace(' ', '_')}.md"
            if topic else "research_report.md"
        ),
        mime="text/markdown",
    )
