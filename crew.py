"""
CrewAI Research Team - Agents, Tasks, and Crew setup.
Optimized for Groq free tier (8000 TPM on openai/gpt-oss-120b).
"""

import time
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS


# ---------------------------------------------------------------------------
# LLM Setup (Groq via LiteLLM)
# ---------------------------------------------------------------------------
def _get_groq_llm(temperature: float = 0.4, max_tokens: int = 1200) -> LLM:
    return LLM(
        model="groq/openai/gpt-oss-120b",
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort="low",  # fewer hidden reasoning tokens
    )


# ---------------------------------------------------------------------------
# DuckDuckGo Search Tool (small output to save tokens)
# ---------------------------------------------------------------------------
class DuckDuckGoSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Search the web for current information. "
        "Input should be a clear search query string. "
        "Returns titles, snippets, and URLs."
    )

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            return "No results found for that query."

        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"Snippet: {r.get('body', 'N/A')[:300]}\n"
                f"URL: {r.get('href', 'N/A')}\n"
            )
        return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def create_agents() -> dict:
    llm_light = _get_groq_llm(max_tokens=1200)   # analyst + synthesizer
    llm_heavy = _get_groq_llm(max_tokens=2500)   # writer + editor
    search_tool = DuckDuckGoSearchTool()

    research_analyst = Agent(
        role="Senior Research Analyst",
        goal=(
            "Gather accurate, current, and relevant information on the given topic "
            "using web search. Identify key facts, statistics, trends, and sources."
        ),
        backstory=(
            "You are a veteran research analyst with 10 years of experience in "
            "investigative reporting and data gathering. You are meticulous about "
            "source quality and always look for primary sources and recent data."
        ),
        tools=[search_tool],
        llm=llm_light,
        verbose=True,
        allow_delegation=False,
        max_iter=4,
        max_rpm=10,
    )

    data_synthesizer = Agent(
        role="Data Synthesis Specialist",
        goal=(
            "Take raw research findings and distill them into a clear, logical "
            "outline with key insights and supporting evidence."
        ),
        backstory=(
            "You excel at pattern recognition and critical thinking. You can look "
            "at a pile of facts and instantly spot the narrative thread."
        ),
        llm=llm_light,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_rpm=10,
    )

    report_writer = Agent(
        role="Technical Report Writer",
        goal=(
            "Write a comprehensive, well-structured Markdown report based on "
            "the provided outline and research."
        ),
        backstory=(
            "You are a seasoned technical writer who has authored hundreds of "
            "research reports for executive audiences. You write clearly and "
            "structure documents with logical headings and actionable takeaways."
        ),
        llm=llm_heavy,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_rpm=10,
    )

    editor = Agent(
        role="Report Editor",
        goal=(
            "Review the draft report for factual consistency, clarity, grammar, "
            "and formatting. Produce the final polished version."
        ),
        backstory=(
            "You are a meticulous editor with an eye for detail. You tighten "
            "prose, fix formatting, and never change facts, only presentation."
        ),
        llm=llm_heavy,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_rpm=10,
    )

    return {
        "research_analyst": research_analyst,
        "data_synthesizer": data_synthesizer,
        "report_writer": report_writer,
        "editor": editor,
    }


# ---------------------------------------------------------------------------
# Tasks (no duplicated context)
# ---------------------------------------------------------------------------
def create_tasks(agents: dict) -> list:
    depth_instruction = "{depth_instruction}"  # filled at runtime via kickoff inputs

    research_task = Task(
        description=(
            "Research the topic: '{topic}'.\n\n"
            "Use the Web Search tool to find recent developments, key statistics, "
            "main debates, and credible sources. Keep it to 2-3 searches.\n"
            f"{depth_instruction}"
        ),
        expected_output=(
            "A structured research brief with 5-8 key findings, supporting data, "
            "and source URLs."
        ),
        agent=agents["research_analyst"],
    )

    synthesis_task = Task(
        description=(
            "Using the research brief, identify the 3-5 key takeaways and create "
            "a logical report outline with bullet-point notes per section. "
            "Keep the source URL next to each point."
        ),
        expected_output=(
            "Executive summary (2-3 sentences), outline with section headings, "
            "and bullet notes with source URLs."
        ),
        agent=agents["data_synthesizer"],
        context=[research_task],
    )

    writing_task = Task(
        description=(
            "Write a complete Markdown research report on '{topic}' using the "
            "synthesis outline.\n\n"
            "- Proper Markdown headings (##, ###)\n"
            "- Executive summary at the top\n"
            "- Cite sources inline as [Source](URL)\n"
            "- End with conclusion and key takeaways\n"
            f"{depth_instruction}"
        ),
        expected_output="A complete, well-formatted Markdown report.",
        agent=agents["report_writer"],
        context=[synthesis_task],
    )

    editing_task = Task(
        description=(
            "Review the draft report and return the final polished version. "
            "Check consistency, clarity, formatting, and remove any claims or "
            "URLs that look invented."
        ),
        expected_output="The final Markdown report only, no preamble.",
        agent=agents["editor"],
        context=[writing_task],
    )

    return [research_task, synthesis_task, writing_task, editing_task]


# ---------------------------------------------------------------------------
# Crew Factory
# ---------------------------------------------------------------------------
def _pause_between_tasks(_task_output):
    """Let Groq's per-minute token window reset between tasks."""
    time.sleep(25)


def create_research_crew() -> Crew:
    agents = create_agents()
    tasks = create_tasks(agents)

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        max_rpm=10,
        task_callback=_pause_between_tasks,
    )
