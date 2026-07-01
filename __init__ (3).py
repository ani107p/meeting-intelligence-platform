"""
Multi-agent RAG pipeline orchestrated with LangGraph.

Agent graph:

    QueryPlanner -> Retriever (hybrid search) -> GraphEnricher (KG relationship
    discovery) -> Reranker -> Synthesizer -> Validator
                                                  |
                                     (loops back to Synthesizer once on failure)

Each node is a plain function operating on a shared `AgentState` TypedDict,
which is how LangGraph expects state to flow -- this is a real StateGraph,
not a simulated one.
"""
from __future__ import annotations
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from ..retrieval.hybrid_search import HybridSearchIndex, SearchResult
from ..retrieval.reranker import rerank
from ..graph.knowledge_graph import MeetingKnowledgeGraph
from .llm_backend import get_llm


class AgentState(TypedDict, total=False):
    query: str
    filters: dict
    search_terms: str
    candidates: list[SearchResult]
    graph_context: list[dict]
    answer: str
    validation: str
    retries: int
    trace: Annotated[list[str], operator.add]


class MeetingIntelligenceAgents:
    def __init__(self, index: HybridSearchIndex, kg: MeetingKnowledgeGraph, llm=None):
        self.index = index
        self.kg = kg
        self.llm = llm or get_llm()
        self.graph = self._build_graph()

    # ---------------- agent nodes ----------------

    def query_planner(self, state: AgentState) -> AgentState:
        prompt = f"[TASK=plan_query]\n{state['query']}"
        plan = self.llm.invoke(prompt)
        return {"search_terms": plan, "trace": [f"QueryPlanner: {plan}"]}

    def retriever(self, state: AgentState) -> AgentState:
        results = self.index.search(
            state["query"], top_k=15, candidate_pool=80, filters=state.get("filters"),
        )
        return {
            "candidates": results,
            "trace": [f"Retriever: hybrid search returned {len(results)} candidates"],
        }

    def graph_enricher(self, state: AgentState) -> AgentState:
        """For each top candidate meeting, pull related meetings + opportunity
        context from the knowledge graph -- this is the graph-based
        relationship discovery step."""
        enriched = []
        for c in state["candidates"][:8]:
            related = self.kg.related_meetings(c.doc_id, max_hops=2)
            opp_id = c.record.get("opportunity_id")
            timeline = self.kg.opportunity_timeline(opp_id) if opp_id else []
            enriched.append({
                "doc_id": c.doc_id,
                "related_meeting_count": len(related),
                "related_meetings": related[:3],
                "opportunity_timeline_length": len(timeline),
            })
        return {
            "graph_context": enriched,
            "trace": [f"GraphEnricher: enriched {len(enriched)} candidates with KG relationships"],
        }

    def reranker(self, state: AgentState) -> AgentState:
        reranked = rerank(state["query"], state["candidates"])
        return {"candidates": reranked, "trace": ["Reranker: applied cross-scoring re-rank"]}

    def synthesizer(self, state: AgentState) -> AgentState:
        graph_lookup = {g["doc_id"]: g for g in state.get("graph_context", [])}
        context_lines = []
        for c in state["candidates"][:6]:
            g = graph_lookup.get(c.doc_id, {})
            related_note = f" (linked to {g.get('related_meeting_count', 0)} related meetings)" if g else ""
            context_lines.append(
                f"- [{c.doc_id}] {c.record['title']} ({c.record['date'][:10]}): "
                f"{c.record['summary']}{related_note} [score={c.score:.3f}]"
            )
        prompt = "[TASK=synthesize]\n" + "\n".join(context_lines)
        answer = self.llm.invoke(prompt)
        return {"answer": answer, "trace": ["Synthesizer: generated grounded answer"]}

    def validator(self, state: AgentState) -> AgentState:
        prompt = f"[TASK=validate]\n{state['answer']}"
        verdict = self.llm.invoke(prompt)
        retries = state.get("retries", 0)
        return {
            "validation": verdict,
            "retries": retries + 1,
            "trace": [f"Validator: {verdict}"],
        }

    def _route_after_validation(self, state: AgentState) -> str:
        if state["validation"].startswith("VALID") or state.get("retries", 0) >= 2:
            return END
        return "synthesizer"

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("planner", self.query_planner)
        g.add_node("retriever", self.retriever)
        g.add_node("graph_enricher", self.graph_enricher)
        g.add_node("reranker", self.reranker)
        g.add_node("synthesizer", self.synthesizer)
        g.add_node("validator", self.validator)

        g.set_entry_point("planner")
        g.add_edge("planner", "retriever")
        g.add_edge("retriever", "graph_enricher")
        g.add_edge("graph_enricher", "reranker")
        g.add_edge("reranker", "synthesizer")
        g.add_edge("synthesizer", "validator")
        g.add_conditional_edges("validator", self._route_after_validation,
                                 {"synthesizer": "synthesizer", END: END})
        return g.compile()

    def run(self, query: str, filters: dict | None = None) -> AgentState:
        initial: AgentState = {"query": query, "filters": filters or {}, "trace": [], "retries": 0}
        return self.graph.invoke(initial)
