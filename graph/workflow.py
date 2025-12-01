from langgraph.graph import StateGraph, END
from typing import Dict, List
from agents.retriever_agent import retriever
from agents.analyzer_agent import analyzer
from agents.generator_agent import generator
from agents.critic_agent import critic

class RAGState(dict):
    query: str
    retrieved: List
    summary: str
    answer: str
    evaluation: str

def create_workflow():
    graph = StateGraph(RAGState)

    graph.set_entry_point("retriever")
    graph.add_node("retriever", retriever)
    graph.add_node("analyzer", analyzer)
    graph.add_node("generator", generator)
    graph.add_node("critic", critic)

    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", END)

    return graph.compile()
