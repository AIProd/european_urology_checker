# agent_graph.py

import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END

from indexer import retrieve_guidelines  # <-- our in-memory RAG helper

load_dotenv()


# --- 1. STATE DEFINITION ---

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
    if llm is None:
        return {
            "paper_type": "Unknown",
            "audit_logs": ["❌ LLM is not initialized. Check Azure env vars."],
        }

    content_snippet = state["paper_content"][:3000]
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following introduction/abstract. "
        "Classify strictly as one of: 'Systematic Review', 'Meta-analysis', "
        "'Randomized Clinical Trial', or 'Observational Study'. "
        "Return ONLY the category name.\n\nText: {text}"
    )
    chain = prompt | llm
    category = chain.invoke({"text": content_snippet}).content.strip()

    return {
        "paper_type": category,
        "audit_logs": [f"**Paper classified as:** {category}"],
    }


def general_auditor_node(state: AgentState):
    if llm is None:
        return {"audit_logs": ["❌ LLM is not initialized."]}

    try:
        general_rules_docs = retrieve_guidelines(
            "p-values, confidence intervals, effect sizes, "
            "exact p-values, statistical reporting, significant figures",
            k=5,
        )
    except Exception as e:
        return {
            "audit_logs": [
                f"⚠️ Could not retrieve guideline chunks for general stats check: {e}"
            ]
        }

    rule_text = "\n".join(doc.page_content for doc in general_rules_docs)

    # middle chunk of paper
    n = len(state["paper_content"])
    mid = n // 2
    paper_snippet = state["paper_content"][max(0, mid - 2000): min(n, mid + 2000)]

    prompt = ChatPromptTemplate.from_template(
        """
You are a Statistical Editor for European Urology.

You are given:
[GUIDELINES]
{rules}
[/GUIDELINES]

and a snippet from the manuscript:
[MANUSCRIPT SNIPPET]
{paper}
[/MANUSCRIPT SNIPPET]

TASK:
- Identify up to 3–5 **potential** statistical reporting issues in this snippet,
  based strictly on the snippet content and the guidelines.

STRICT RULES:
- Do NOT invent numerical values (sample sizes, counts, percentages) that do NOT
  appear explicitly in the snippet.
- For every issue, you MUST:
  - Quote the exact text from the snippet (short quote) as evidence.
  - Explain the issue in 1–3 sentences.
  - Tag severity as one of: minor, moderate, major.
- If you cannot see enough information in this snippet to judge general
  statistical reporting, say: "General Stats: Cannot assess based on this snippet."

Output format (Markdown):

### General Stats Check
- Severity: <minor|moderate|major>
  - Issue: <short description>
  - Evidence: "<exact quote from snippet>"
  - Guideline_ref: <short reference to relevant rule>

If you truly cannot assess, output:

### General Stats Check
General Stats: Cannot assess based on this snippet.
"""
    )

    response = (prompt | llm).invoke({"rules": rule_text, "paper": paper_snippet})

    return {
        "audit_logs": [response.content]
    }


def specific_auditor_node(state: AgentState):
    if llm is None:
        return {"audit_logs": ["❌ LLM is not initialized."]}

    paper_type = state["paper_type"] or ""

    if "Systematic" in paper_type or "Meta" in paper_type:
        query = "PRISMA, systematic review, meta-analysis checklist, heterogeneity"
    elif "Observational" in paper_type:
        query = "causality, causal language, confounding, bias, observational study"
    else:
        query = "CONSORT, randomized clinical trial, allocation, blinding, ITT"

    try:
        specific_rules_docs = retrieve_guidelines(query, k=5)
    except Exception as e:
        return {
            "audit_logs": [
                f"⚠️ Could not retrieve guideline chunks for specific check: {e}"
            ]
        }

    rule_text = "\n".join(doc.page_content for doc in specific_rules_docs)

    # Take end of the paper (Discussion/Conclusion)
    end_text = state["paper_content"][-3000:]

    prompt = ChatPromptTemplate.from_template(
        "Check compliance for paper type '{ptype}'. "
        "GUIDELINES:\n{rules}\n\n"
        "Focus on Discussion/Conclusion and identify any violations.\n\n"
        "PAPER TEXT:\n{paper}\n\n"
        "Report violations briefly. If compliant, say 'Specific Check: Pass'."
    )

    response = (prompt | llm).invoke(
        {"ptype": paper_type, "rules": rule_text, "paper": end_text}
    )

    return {
        "audit_logs": [f"### Specific Guideline Check ({paper_type})\n{response.content}"]
    }


def reporter_node(state: AgentState):
    if llm is None:
        logs = "\n\n".join(state["audit_logs"])
        fallback = (
            "# 🇪🇺 European Urology Statistical Report\n"
            f"**Detected Type:** {state['paper_type']}\n\n"
            "---\n"
            "LLM was not initialized, showing raw logs only:\n\n"
            f"{logs}\n\n"
            "---\n"
            "*Generated by AI Agent*"
        )
        return {"final_report": fallback}

    logs = "\n\n".join(state["audit_logs"])

    prompt = ChatPromptTemplate.from_template(
        """
You are the lead statistical editor for European Urology.

You are given internal AI audit logs about a manuscript's statistics:

[AI AUDIT LOGS]
{logs}
[/AI AUDIT LOGS]

Your job:

1. Synthesize these logs into a concise, editor-facing report.
2. Prioritize issues into three categories:
   - BLOCKING issues: must be fixed before acceptance.
   - Important but fixable: strong recommendation to fix before acceptance.
   - Minor issues / suggestions: nice-to-have, style, or borderline items.
3. Provide:
   - A 1–2 paragraph plain-language overall summary of statistical quality.
   - A bullet list of BLOCKING issues (if any).
   - A bullet list of Important but fixable issues.
   - A bullet list of Minor issues / suggestions.
   - One sentence with a clear provisional recommendation in terms of statistics
     (e.g., "Statistically acceptable", "Acceptable with revisions",
     "Not acceptable in the current form").

VERY IMPORTANT:
- Do NOT invent new problems or numerical examples that are not clearly stated
  in the logs. Only use what is actually present in the logs.
- If the logs themselves are speculative, treat them as "possible issues" and
  phrase accordingly.
- If no blocking issues are evident, explicitly say so.

Output format (Markdown):

# 🇪🇺 European Urology Statistical Report
**Detected Type:** <paper_type>

## Overall summary
<1–2 paragraphs>

## Blocking issues
- ...

## Important but fixable issues
- ...

## Minor issues / suggestions
- ...

## Provisional recommendation
- <one sentence>
"""
    )

    chain = prompt | llm
    response = chain.invoke({"logs": logs, "paper_type": state["paper_type"]})
    return {"final_report": response.content}



# --- 4. GRAPH BUILD ---

workflow = StateGraph(AgentState)
workflow.add_node("classify", classifier_node)
workflow.add_node("check_general", general_auditor_node)
workflow.add_node("check_specific", specific_auditor_node)
workflow.add_node("report", reporter_node)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "check_general")
workflow.add_edge("check_general", "check_specific")
workflow.add_edge("check_specific", "report")
workflow.add_edge("report", END)

app_graph = workflow.compile()
