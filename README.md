# 🔬 Research Multi-Agent Team

A CrewAI-powered research assistant with a Streamlit UI. Enter a topic and four
agents collaborate to produce a well-structured research report.

## Agent Team

| Agent | Role |
|-------|------|
| Research Analyst | Searches the web for facts, data, and sources |
| Data Synthesizer | Distills findings into a logical outline |
| Report Writer | Writes the full Markdown report |
| Editor | Polishes and fact-checks the final output |

## Tech Stack

- **CrewAI** — multi-agent orchestration
- **Groq** — LLM provider (`openai/gpt-oss-120b`)
- **Streamlit** — web UI
- **DuckDuckGo** — free web search

## Local Setup

```bash
pip install -r requirements.txt
