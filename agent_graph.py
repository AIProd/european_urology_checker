# agent_graph.py

import operator
import os
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph import StateGraph, END

load_dotenv()

DB_DIR = "./chroma_db"


# --- 1. STATE DEFINITION ---

class AgentState(TypedDict):
    paper_content: str
    paper_type: str
    audit_logs: Annotated[List[str], operator.add]
    final_report: str


# --- 2. SETUP MODELS & DB ---

def get_azure_resources():
    """
    Create Azure LLM + embedding client with consistent config.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    chat_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    if not all([azure_endpoint, azure_api_key, azure_api_version, chat_deployment, embedding_deployment]):
        raise ValueError(
            "Missing Azure env vars. Expected: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_API_VERSION, AZURE_DEPLOYMENT_NAME, AZURE_EMBEDDING_DEPLOYMENT."
        )

    llm = AzureChatOpenAI(
        azure_deployment=chat_deployment,
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        openai_api_version=azure_api_version,
    )

    embedding_function = AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        openai_api_version=azure_api_version,
    )

    return llm, embedding_function


try:
    llm, embedding_function = get_azure_resources()
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_function,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    else:
        retriever = None
except Exception as e:
    print(f"Initialization Warning: {e}")
    llm = None
    retriever = None


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

    if not retriever:
        return {"audit_logs": ["⚠️ Error: Knowledge base (Chroma DB) not found. Build it first."]}

    # Retrieve general statistical rules
    general_rules_docs = retriever.invoke(
        "p-values, confidence intervals, effect sizes, exact p-values, "
        "statistical reporting guidelines"
    )
    rule_text = "\n".join([doc.page_content for doc in general_rules_docs])

    # Take a middle chunk of the paper for general stats check
    n = len(state["paper_content"])
    mid = n // 2
    paper_snippet = state["paper_content"][max(0, mid - 2000): min(n, mid + 2000)]

    prompt = ChatPromptTemplate.from_template(
        "You are a Statistical Editor for European Urology. "
        "Use the following GUIDELINES:\n{rules}\n\n"
        "PAPER SNIPPET:\n{paper}\n\n"
        "Identify 3–5 critical statistical reporting errors. "
        "If none, say 'General Stats: Pass'."
    )
    response = (prompt | llm).invoke({"rules": rule_text, "paper": paper_snippet})

    return {
        "audit_logs": [f"### General Stats Check\n{response.content}"]
    }


def specific_auditor_node(state: AgentState):
    if llm is None:
        return {"audit_logs": ["❌ LLM is not initialized."]}

    if not retriever:
        return {"audit_logs": ["⚠️ Error: Knowledge base (Chroma DB) not found for specific check."]}

    paper_type = state["paper_type"] or ""

    if "Systematic" in paper_type or "Meta" in paper_type:
        query = "PRISMA guidelines, systematic review, meta-analysis checklist, heterogeneity"
    elif "Observational" in paper_type:
        query = "causality, causal language, confounding, observational study guidelines"
    else:
        query = "randomized clinical trial, CONSORT, allocation, blinding, intention-to-treat"

    specific_rules_docs = retriever.invoke(query)
    rule_text = "\n".join([doc.page_content for doc in specific_rules_docs])

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
    logs = "\n\n".join(state["audit_logs"])
    report = (
        "# 🇪🇺 European Urology Statistical Report\n"
        f"**Detected Type:** {state['paper_type']}\n\n"
        "---\n"
        f"{logs}\n\n"
        "---\n"
        "*Generated by AI Agent*"
    )
    return {"final_report": report}


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
