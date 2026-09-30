"""
CrewAI Research Team - Agents, Tasks, and Crew setup.
"""

import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS


# ---------------------------------------------------------------------------
# LLM Setup (Groq via LiteLLM)
# ---------------------------------------------------------------------------
# CrewAI uses LiteLLM under the hood for Groq.
# LiteLLM model string format:  groq/<model_id>
# Groq's model ID for GPT-OSS 120B is:  openai/gpt-oss-120b
# So the full LiteLLM string is:         groq/openai/gpt-oss-120b
#
# The API key is read in app.py and set as an environment variable here.
# Never hardcode keys.

def _get_groq_llm(temperature: float = 0.4) -> LLM:
    """Create a Groq LLM instance for CrewAI."""
    return LLM(
        model="groq/compound",  # ← 70,000 TPM on free tier
        temperature=temperature,
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# DuckDuckGo Search Tool
# ---------------------------------------------------------------------------
# CrewAI ships a DuckDuckGoSearchTool in crewai-tools, but it pulls in
# langchain-community, which can cause dependency clashes. A minimal custom
# tool using `duckduckgo-search` directly keeps the dependency tree lean.

class DuckDuckGoSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Search the web for current information. "
        "Input should be a clear search query string. "
        "Returns titles, snippets, and URLs."
    )

    def _run(self, query: str) -> str:
        """Run a DuckDuckGo text search and return formatted results."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            return "No results found for that query."

        output_parts = []
        for r in results:
            output_parts.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"Snippet: {r.get('body', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}\n"
            )
        return "\n---\n".join(output_parts)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def create_agents() -> dict:
    """Create and return the four research agents."""
    llm = _get_groq_llm()
    search_tool = DuckDuckGoSearchTool()

    # -- Agent 1: Research Analyst (only agent with web search tool) --------
    research_analyst = Agent(
        role="Senior Research Analyst",
        goal=(
            "Gather accurate, current, and relevant information on the given topic "
            "using web search. Identify key facts, statistics, trends, and sources."
        ),
        backstory=(
            "You are a veteran research analyst with 10 years of experience in "
            "investigative reporting and data gathering. You are meticulous about "
            "source quality and always look for primary sources and recent data. "
            "You present findings in a structured, easy-to-digest format."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,   # cap search loops to avoid Groq rate limits
        max_rpm=10,   # gentle rate limit
    )

    # -- Agent 2: Data Synthesizer (no external tools) ----------------------
    data_synthesizer = Agent(
        role="Data Synthesis Specialist",
        goal=(
            "Take raw research findings and distill them into a clear, logical "
            "outline with key insights and supporting evidence."
        ),
        backstory=(
            "You excel at pattern recognition and critical thinking. You can look "
            "at a pile of facts and instantly spot the narrative thread. You are "
            "known for asking 'so what?' until the real significance emerges."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_rpm=10,
    )

    # -- Agent 3: Report Writer ---------------------------------------------
    report_writer = Agent(
        role="Technical Report Writer",
        goal=(
            "Write a comprehensive, well-structured Markdown report based on "
            "the provided outline and research."
        ),
        backstory=(
            "You are a seasoned technical writer who has authored hundreds of "
            "research reports for executive audiences. You write clearly, avoid "
            "jargon, and always structure documents with logical headings, "
            "bullet points, and actionable takeaways."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_rpm=10,
    )

    # -- Agent 4: Editor ----------------------------------------------------
    editor = Agent(
        role="Report Editor",
        goal=(
            "Review the draft report for factual consistency, clarity, grammar, "
            "and formatting. Produce the final polished version."
        ),
        backstory=(
            "You are a meticulous editor with an eye for detail and a commitment "
            "to accuracy. You catch inconsistencies, tighten prose, fix formatting "
            "issues, and ensure the final product meets professional standards. "
            "You never change facts, only presentation."
        ),
        llm=llm,
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
# Tasks
# ---------------------------------------------------------------------------

def create_tasks(agents: dict) -> list:
    """Create tasks in sequential order. Context flows via explicit `context`."""
    depth_instruction = "{depth_instruction}"  # filled at runtime via kickoff inputs

    # -- Task 1: Research ---------------------------------------------------
    research_task = Task(
        description=(
            "Research the topic: '{topic}'.\n\n"
            "Use the Web Search tool to find:\n"
            "- Recent developments and current state\n"
            "- Key statistics and data points\n"
            "- Main debates or open questions\n"
            "- Credible sources and examples\n\n"
            "Compile a structured research brief with all findings.\n"
            f"{depth_instruction}"
        ),
        expected_output=(
            "A structured research brief containing:\n"
            "- 5-8 key findings with supporting data\n"
            "- Source URLs for key claims\n"
            "- Any notable trends or patterns"
        ),
        agent=agents["research_analyst"],
    )

    # -- Task 2: Synthesis --------------------------------------------------
    synthesis_task = Task(
        description=(
            "Using the research brief provided, synthesize the information.\n\n"
            "Identify:\n"
            "- The 3-5 most important takeaways\n"
            "- A logical outline for the final report\n"
            "- Key evidence that supports each point\n"
            "- Any gaps or contradictions in the research\n\n"
            "Output a clear outline with bullet points for each section."
        ),
        expected_output=(
            "A synthesis document with:\n"
            "- Executive summary (2-3 sentences)\n"
            "- Report outline with section headings\n"
            "- Bullet-point notes for each section"
        ),
        agent=agents["data_synthesizer"],
        context=[research_task],  # depends on research output
    )

    # -- Task 3: Writing ----------------------------------------------------
    writing_task = Task(
        description=(
            "Write a complete Markdown research report on '{topic}'.\n\n"
            "Use the synthesis outline and research findings provided.\n\n"
            "Requirements:\n"
            "- Use proper Markdown headings (##, ###)\n"
            "- Include an executive summary at the top\n"
            "- Use bullet points and bold for key terms\n"
            "- Cite sources inline where appropriate (e.g., [Source](URL))\n"
            "- End with a conclusion and key takeaways section\n"
            f"{depth_instruction}"
        ),
        expected_output=(
            "A complete Markdown report of at least 800 words (brief) "
            "or 1500 words (detailed), with proper formatting."
        ),
        agent=agents["report_writer"],
        context=[synthesis_task, research_task],
    )

    # -- Task 4: Editing ----------------------------------------------------
    editing_task = Task(
        description=(
            "Review the draft report and produce the final polished version.\n\n"
            "Check for:\n"
            "- Factual consistency with the research findings\n"
            "- Clear and professional language\n"
            "- Proper Markdown formatting\n"
            "- No hallucinated claims or fake URLs\n\n"
            "Return the complete final report in Markdown."
        ),
        expected_output=(
            "The final, polished Markdown report ready for the user. "
            "No preamble, just the report content."
        ),
        agent=agents["editor"],
        context=[writing_task, research_task],
    )

    return [research_task, synthesis_task, writing_task, editing_task]


# ---------------------------------------------------------------------------
# Crew Factory
# ---------------------------------------------------------------------------

def create_research_crew() -> Crew:
    """Build the full crew with agents, tasks, and sequential process."""
    agents = create_agents()
    tasks = create_tasks(agents)

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        max_rpm=10,  # rate limiting guardrail
    )
