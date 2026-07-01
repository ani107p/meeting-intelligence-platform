"""
Enterprise Meeting Intelligence Platform -- end-to-end demo.

Run:
    python main.py

This will:
  1. Load (or generate) 4,200 synthetic meeting records
  2. Build a hybrid search index (Qdrant in-process + BM25)
  3. Build a knowledge graph (meetings <-> opportunities <-> people <-> dependencies)
  4. Run the LangGraph multi-agent pipeline against a set of demo queries
  5. Print retrieval accuracy benchmark vs a naive baseline
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.hybrid_search import HybridSearchIndex
from src.graph.knowledge_graph import MeetingKnowledgeGraph
from src.agents.graph_pipeline import MeetingIntelligenceAgents
from src.generate_data import main as generate_data

DATA_DIR = Path(__file__).parent / "data"

DEMO_QUERIES = [
    "Which deals with Redwood Consulting have budget freeze risk?",
    "Show me contract negotiation meetings with data residency compliance concerns",
    "What escalation reviews happened for API rate limiting concerns?",
    "Find quarterly business reviews where the champion left the company",
]


def load_data():
    if not (DATA_DIR / "meetings.json").exists():
        print("No data found -- generating synthetic dataset...")
        generate_data()
    meetings = json.loads((DATA_DIR / "meetings.json").read_text())
    opportunities = json.loads((DATA_DIR / "opportunities.json").read_text())
    dependencies = json.loads((DATA_DIR / "dependencies.json").read_text())
    return meetings, opportunities, dependencies


def naive_keyword_search(meetings, query, top_k=15):
    """Baseline: plain substring match on summary, no ranking sophistication.
    Used to compute the retrieval-accuracy improvement number."""
    q_words = set(query.lower().split())
    scored = []
    for m in meetings:
        text = (m["summary"] + " " + m["title"]).lower()
        overlap = sum(1 for w in q_words if w in text)
        if overlap:
            scored.append((overlap, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


def relevance_label(query: str, meeting: dict) -> bool:
    """Heuristic ground-truth relevance for benchmarking: a meeting counts as
    relevant if it shares its topic/risk_flag/company keywords with the query.
    (In a real eval this would be human-labeled; here it's a reproducible proxy.)"""
    q = query.lower()
    hits = 0
    for field in ("topic", "risk_flag", "company", "type"):
        val = str(meeting.get(field, "")).lower()
        if val and val in q:
            hits += 1
    return hits > 0


def benchmark_retrieval(meetings, index, queries):
    print("\n" + "=" * 70)
    print("RETRIEVAL ACCURACY BENCHMARK: hybrid+rerank vs naive keyword search")
    print("=" * 70)
    hybrid_precisions, naive_precisions = [], []
    for q in queries:
        hybrid_results = index.search(q, top_k=15)
        naive_results = naive_keyword_search(meetings, q, top_k=15)

        h_relevant = sum(1 for r in hybrid_results if relevance_label(q, r.record))
        n_relevant = sum(1 for r in naive_results if relevance_label(q, r))

        h_prec = h_relevant / max(len(hybrid_results), 1)
        n_prec = n_relevant / max(len(naive_results), 1)
        hybrid_precisions.append(h_prec)
        naive_precisions.append(n_prec)
        print(f"\nQuery: {q}")
        print(f"  naive precision@15  = {n_prec:.2f}")
        print(f"  hybrid precision@15 = {h_prec:.2f}")

    avg_h = sum(hybrid_precisions) / len(hybrid_precisions)
    avg_n = sum(naive_precisions) / len(naive_precisions)
    improvement = ((avg_h - avg_n) / max(avg_n, 1e-9)) * 100
    print(f"\nAverage naive precision@15:  {avg_n:.3f}")
    print(f"Average hybrid precision@15: {avg_h:.3f}")
    print(f"Relative improvement:        {improvement:+.1f}%")
    return improvement


def main():
    t0 = time.time()
    meetings, opportunities, dependencies = load_data()
    print(f"Loaded {len(meetings)} meetings, {len(opportunities)} opportunities, "
          f"{len(dependencies)} dependency edges")

    print("\nBuilding hybrid search index (Qdrant in-process + BM25)...")
    index = HybridSearchIndex().build(meetings)
    print(f"Index built in {time.time() - t0:.1f}s")

    print("Building knowledge graph...")
    kg = MeetingKnowledgeGraph().build(meetings, opportunities, dependencies)
    print("Graph stats:", kg.stats())

    print("\nInitializing multi-agent pipeline (LangGraph)...")
    agents = MeetingIntelligenceAgents(index, kg)
    print(f"LLM backend: {agents.llm.name}")

    print("\n" + "=" * 70)
    print("MULTI-AGENT PIPELINE DEMO RUNS")
    print("=" * 70)
    for q in DEMO_QUERIES:
        print(f"\n--- Query: {q} ---")
        result = agents.run(q)
        print("Agent trace:")
        for step in result["trace"]:
            print(f"  > {step}")
        print("\nAnswer:")
        print(result["answer"])

    # Graph relationship discovery showcase
    print("\n" + "=" * 70)
    print("GRAPH-BASED RELATIONSHIP DISCOVERY")
    print("=" * 70)
    high_risk = kg.high_risk_opportunities()[:5]
    print("\nTop 5 highest-risk opportunities (by proportion of risk-flagged meetings):")
    for opp in high_risk:
        print(f"  {opp['opportunity_id']} | {opp['name']} | stage={opp['stage']} | "
              f"risk_ratio={opp['risk_ratio']} | ${opp['value_usd']:,}")

    sample_meeting = meetings[0]["id"]
    related = kg.related_meetings(sample_meeting, max_hops=2)
    print(f"\nMeetings related to {sample_meeting} within 2 hops: {len(related)} found")
    for r in related[:5]:
        print(f"  - {r['meeting_id']} ({r['hops']} hops): {r['title']}")

    benchmark_retrieval(meetings, index, DEMO_QUERIES)

    print(f"\nTotal demo runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
