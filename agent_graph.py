# agent_graph.py

import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END

from indexer import retrieve_guidelines_by_type  # our RAG helper

load_dotenv()


# --- 1. STATE ---

class AgentState(TypedDict):
    paper_content: str
    paper_type: str
    audit_logs: Annotated[List[str], operator.add]
    final_report: str


# --- 2. LLM SETUP ---

def _get_llm() -> AzureChatOpenAI:
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    chat_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")

    if not all([azure_endpoint, azure_api_key, azure_api_version, chat_deployment]):
        raise ValueError(
            "Missing Azure env vars. Expected: AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_DEPLOYMENT_NAME."
        )

    return AzureChatOpenAI(
        azure_deployment=chat_deployment,
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        openai_api_version=azure_api_version,
    )


try:
    llm = _get_llm()
except Exception as e:
    print(f"Initialization Warning (LLM): {e}")
    llm = None


# --- 3. NODES ---

def classifier_node(state: AgentState):
    """Classify manuscript type from abstract/intro."""
    if llm is None:
        return {"paper_type": "Unknown", "audit_logs": ["LLM not initialized."]}

    content_snippet = state["paper_content"][:4000]

    prompt = ChatPromptTemplate.from_template(
        """
You are an experienced statistical editor for *European Urology*.

Given the text below (mostly abstract/introduction), classify the manuscript
STRICTLY as one of the following categories:

- "Randomized Clinical Trial"
- "Observational Study"
- "Systematic Review"
- "Meta-analysis"
- "Other"

Return ONLY the category name, nothing else.

TEXT:
{text}
        """
    )

    resp = (prompt | llm).invoke({"text": content_snippet})
    category = resp.content.strip()

    return {
        "paper_type": category,
        "audit_logs": [f"**Paper classified as:** {category}"],
    }


def stats_auditor_node(state: AgentState):
    """Check general statistical reporting against stats guidelines."""
    if llm is None:
        return {"audit_logs": ["*Error: LLM not initialized.*"]}

    try:
        guideline_docs = retrieve_guidelines_by_type(
            "statistics",
            "p-values, confidence intervals, precision, effect sizes, "
            "primary endpoint, sample size",
            k=5,
        )
    except Exception as e:
        return {"audit_logs": [f"### Statistical Reporting Check\nCould not load statistics guidelines: {e}"]}

    rules = "\n\n".join(d.page_content for d in guideline_docs)
    paper_snip = state["paper_content"]

    prompt = ChatPromptTemplate.from_template(
        """
You are a Statistical Editor for *European Urology*.

You have the official **Guidelines for Reporting of Statistics for Clinical Research in Urology**.
Here are relevant extracts:

---------------- GUIDELINES (STATISTICS) ----------------
{rules}
---------------------------------------------------------

Now check the following manuscript text for **statistical reporting** issues,
using those guidelines as your reference. Focus on:

- Whether effect estimates have confidence intervals and appropriate precision.
- Whether p-values are used and interpreted according to the guidelines.
- Whether primary/secondary endpoints and analysis methods are clearly reported.
- Any obviously misleading or non-guideline-concordant statistical reporting.

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write a short section titled:

### Statistical Reporting Check

Within it, list:
- 1–3 **blocking issues** (if any) that would prevent acceptance.
- 2–4 **important but fixable** issues (if any).
- Any **minor suggestions**.

If there are essentially no problems, say explicitly that statistical reporting appears compliant.
        """
    )

    resp = (prompt | llm).invoke({"rules": rules, "paper": paper_snip})
    return {"audit_logs": [resp.content]}


def figtab_auditor_node(state: AgentState):
    """Check figures and tables against figure/table guidelines."""
    if llm is None:
        return {"audit_logs": ["*Error: LLM not initialized.*"]}

    try:
        guideline_docs = retrieve_guidelines_by_type(
            "figures_tables",
            "figures tables graphs dos and don'ts precision labels legends "
            "Kaplan-Meier tables example",
            k=5,
        )
    except Exception as e:
        return {"audit_logs": [f"### Figures and Tables Check\nCould not load figures/tables guidelines: {e}"]}

    rules = "\n\n".join(d.page_content for d in guideline_docs)
    paper_snip = state["paper_content"]

    prompt = ChatPromptTemplate.from_template(
        """
You are a Statistical Editor for *European Urology*.

You have the official **Guidelines for Reporting of Figures and Tables for Clinical Research in Urology**.

Relevant extracts:
---------------- FIGURES/TABLES GUIDELINES ----------------
{rules}
----------------------------------------------------------

Using those rules, review how the manuscript appears to use **tables and figures**.
Look for:

- Whether tables/figures complement, rather than duplicate, the text.
- Whether graphs are used only when they meaningfully improve understanding.
- Whether labels, units, precision and legends look appropriate.
- Obvious problems with Kaplan–Meier plots, forest plots, or big summary tables.

You will not see the actual images, but you *can* infer from references in the text
(e.g. “Table 1”, “Figure 2”, “Kaplan–Meier curve”, etc.).

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write a short section titled:

### Figures and Tables Check

Summarize:
- Any major violations of the guidelines (blocking if serious).
- Other important issues.
- Minor suggestions (e.g., clarity, aesthetics, small formatting fixes).
        """
    )

    resp = (prompt | llm).invoke({"rules": rules, "paper": paper_snip})
    return {"audit_logs": [resp.content]}


def type_specific_auditor_node(state: AgentState):
    """
    Route to causality guidelines for observational studies,
    or SR/MA guidelines for systematic reviews/meta-analyses.
    """
    if llm is None:
        return {"audit_logs": ["*Error: LLM not initialized.*"]}

    paper_type = (state.get("paper_type") or "").lower()
    paper_snip = state["paper_content"]

    # Observational / causal focus
    if "observational" in paper_type:
        try:
            guideline_docs = retrieve_guidelines_by_type(
                "causality",
                "causal language, confounding, causal pathways, introduction, methods, discussion",
                k=5,
            )
        except Exception as e:
            return {"audit_logs": [f"### Causality / Observational Study Check\nCould not load causality guidelines: {e}"]}

        rules = "\n\n".join(d.page_content for d in guideline_docs)

        prompt = ChatPromptTemplate.from_template(
            """
You are a Statistical Editor for *European Urology*.

The manuscript has been classified as an **Observational Study**.

You have the official **Guidelines for Reporting Observational Research in Urology: The Importance of Clear Reference to Causality**.

Relevant extracts:
---------------- CAUSALITY GUIDELINES ----------------
{rules}
------------------------------------------------------

Using those guidelines, review the manuscript for **causal language and causal thinking**.

Focus on:
- Does the Introduction clearly state whether the question is causal vs descriptive/predictive?
- Are causal mechanisms/pathways described when appropriate?
- Are confounders and their treatment described in the Methods (not just “we adjusted for…”)?
- Is the Results/Discussion language (“reduces risk”, “improves outcomes”, “associated with”) appropriate
  given the design and analysis?

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write a section titled:

### Causality / Observational Study Check

Under that heading, summarize:
- Causal clarity in aims and introduction.
- Adequacy of confounding control and discussion.
- Appropriateness of causal vs associational language.
- Blocking issues vs smaller wording/interpretation fixes.
            """
        )

        resp = (prompt | llm).invoke({"rules": rules, "paper": paper_snip})
        return {"audit_logs": [resp.content]}

    # Systematic Review / Meta-analysis
    if "systematic review" in paper_type or "meta-analysis" in paper_type or "meta analysis" in paper_type:
        try:
            guideline_docs = retrieve_guidelines_by_type(
                "systematic_meta",
                "PRISMA MOOSE heterogeneity SUCRA protocol reproducible methods justification for review",
                k=5,
            )
        except Exception as e:
            return {"audit_logs": [f"### Systematic Review / Meta-analysis Check\nCould not load SR/MA guidelines: {e}"]}

        rules = "\n\n".join(d.page_content for d in guideline_docs)

        prompt = ChatPromptTemplate.from_template(
            """
You are a Statistical Editor for *European Urology*.

The manuscript has been classified as a **{ptype}**.

You have the official **Guidelines for Meta-analyses and Systematic Reviews in Urology**.

Relevant extracts:
---------------- SR/MA GUIDELINES ----------------
{rules}
--------------------------------------------------

Using those guidelines, review the manuscript for:

- PRISMA/MOOSE-style reporting (flow, inclusion/exclusion, risk of bias).
- Whether the methodology is reported in enough detail for exact replication.
- Whether the rationale for doing this review (vs existing reviews) is compelling.
- Interpretation of heterogeneity, rankings (e.g., SUCRA), and precision.

MANUSCRIPT TEXT:
----------------
{paper}
----------------

Write a section titled:

### Systematic Review / Meta-analysis Check

Summarize:
- Any blocking issues (e.g., non-reproducible methods, misleading rankings).
- Important but fixable issues.
- Minor suggestions.
            """
        )

        resp = (prompt | llm).invoke(
            {"rules": rules, "paper": paper_snip, "ptype": state.get("paper_type", "")}
        )
        return {"audit_logs": [resp.content]}

    # For RCTs or other designs, no additional type-specific guideline beyond general stats/figures.
    return {
        "audit_logs": [
            "### Type-Specific Check\nStudy type does not trigger additional causality or SR/MA checks "
            "beyond general statistics and figures/tables."
        ]
    }


def reporter_node(state: AgentState):
    """Combine logs into a single editorial-style report."""
    if llm is None:
        logs_text = "\n\n".join(state.get("audit_logs", []))
        fallback = (
            "🇪🇺 European Urology Statistical Report\n"
            f"Detected Type: {state.get('paper_type', 'Unknown')}\n\n"
            "LLM not initialized – showing raw logs:\n\n"
            f"{logs_text}"
        )
        return {"final_report": fallback}

    logs_text = "\n\n---\n\n".join(state.get("audit_logs", []))

    prompt = ChatPromptTemplate.from_template(
        """
You are a Statistical Editor for *European Urology*.

A manuscript has been analyzed by several automated checkers.
The manuscript type (as classified) is:

> {paper_type}

Below are their raw notes (unordered, with some overlap):

---------------- ANALYSIS NOTES ----------------
{logs}
------------------------------------------------

Using ONLY these notes (do not invent details you do not see),
draft a concise report in markdown with EXACTLY the following structure:

🇪🇺 European Urology Statistical Report
Detected Type: {paper_type}

Overall summary
- 2–4 bullet points describing the overall quality of reporting & main themes.

Blocking issues
- Bullet list of issues that **must** be fixed before acceptance.
- If none, write: "None."

Important but fixable issues
- Bullet list of non-fatal but important issues.
- If none, write: "None."

Minor issues / suggestions
- Bullet list of minor style, clarity, or presentation suggestions.
- If none, write: "None."

Provisional recommendation
- One line with something like:
  "Acceptable with minor revisions", or
  "Major revisions required", or
  "Not acceptable in current form."

Be concrete but not aggressive in tone, and keep the length similar to an internal editorial note.
        """
    )

    resp = (prompt | llm).invoke(
        {"paper_type": state.get("paper_type", "Unknown"), "logs": logs_text}
    )
    return {"final_report": resp.content}


# --- 4. GRAPH ---

workflow = StateGraph(AgentState)

workflow.add_node("classify", classifier_node)
workflow.add_node("check_stats", stats_auditor_node)
workflow.add_node("check_type_specific", type_specific_auditor_node)
workflow.add_node("check_figtab", figtab_auditor_node)
workflow.add_node("report", reporter_node)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "check_stats")
workflow.add_edge("check_stats", "check_type_specific")
workflow.add_edge("check_type_specific", "check_figtab")
workflow.add_edge("check_figtab", "report")
workflow.add_edge("report", END)

app_graph = workflow.compile()
