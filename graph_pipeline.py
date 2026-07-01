"""
Knowledge graph over meetings, opportunities, people, and companies.

Uses networkx as the graph engine. The schema/query patterns here map
directly onto Neo4j Cypher if you swap the backend for a real Neo4j
instance (see `to_cypher_seed()` for the equivalent CREATE statements) --
the graph model (node types + typed edges) is identical either way.

Node types: Meeting, Opportunity, Person, Company
Edge types: PARTICIPATED_IN, DISCUSSED, BELONGS_TO, DEPENDS_ON (typed),
            OWNS
"""
from __future__ import annotations
import networkx as nx


class MeetingKnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def build(self, meetings: list[dict], opportunities: list[dict], dependencies: list[dict]):
        for opp in opportunities:
            self.g.add_node(opp["id"], type="Opportunity", **{
                k: v for k, v in opp.items() if k != "id"
            })
            self.g.add_node(opp["company"], type="Company")
            self.g.add_edge(opp["id"], opp["company"], relation="BELONGS_TO")
            self.g.add_node(opp["owner"], type="Person")
            self.g.add_edge(opp["owner"], opp["id"], relation="OWNS")

        for m in meetings:
            self.g.add_node(m["id"], type="Meeting", title=m["title"], date=m["date"],
                             topic=m["topic"], risk_flag=m["risk_flag"], company=m["company"])
            if m.get("opportunity_id"):
                self.g.add_edge(m["id"], m["opportunity_id"], relation="DISCUSSED")
            for person in m["attendees"]:
                self.g.add_node(person, type="Person")
                self.g.add_edge(person, m["id"], relation="PARTICIPATED_IN")

        for dep in dependencies:
            self.g.add_edge(
                dep["source_meeting"], dep["target_meeting"],
                relation=dep["type"].upper(), dep_id=dep["id"],
            )
        return self

    # -------- relationship discovery queries --------

    _DEPENDENCY_RELATIONS = {"BLOCKS", "INFORMS", "ESCALATES_FROM", "FOLLOWS_UP_ON", "RELATES_TO"}

    def related_meetings(self, meeting_id: str, max_hops: int = 2) -> list[dict]:
        """Find meetings connected to `meeting_id` via *explicit* dependency
        edges (blocks/informs/escalates_from/...) or by sharing the same
        opportunity. Deliberately excludes the shared-attendee path: with
        only a few dozen people covering thousands of meetings, attendee
        co-occurrence alone produces a near-complete graph and isn't a
        meaningful "relationship" signal on its own."""
        if meeting_id not in self.g:
            return []

        sub_edges = [
            (u, v) for u, v, d in self.g.edges(data=True)
            if d.get("relation") in self._DEPENDENCY_RELATIONS or d.get("relation") == "DISCUSSED"
        ]
        sub = nx.Graph()
        sub.add_edges_from(sub_edges)
        if meeting_id not in sub:
            return []

        related = []
        lengths = nx.single_source_shortest_path_length(sub, meeting_id, cutoff=max_hops)
        for target, hops in lengths.items():
            if target == meeting_id or hops == 0:
                continue
            if self.g.nodes.get(target, {}).get("type") == "Meeting":
                related.append({"meeting_id": target, "hops": hops,
                                 "title": self.g.nodes[target].get("title")})
        related.sort(key=lambda x: x["hops"])
        return related

    def opportunity_timeline(self, opportunity_id: str) -> list[dict]:
        """All meetings tied to an opportunity, chronologically."""
        meetings = [
            n for n in self.g.predecessors(opportunity_id)
            if self.g.nodes[n].get("type") == "Meeting"
        ] if opportunity_id in self.g else []
        timeline = [
            {"meeting_id": m, "date": self.g.nodes[m].get("date"),
             "title": self.g.nodes[m].get("title"), "risk_flag": self.g.nodes[m].get("risk_flag")}
            for m in meetings
        ]
        timeline.sort(key=lambda x: x["date"] or "")
        return timeline

    def escalation_chain(self, meeting_id: str) -> list[dict]:
        """Trace ESCALATES_FROM edges backward to find the root cause meeting."""
        chain = [meeting_id]
        current = meeting_id
        seen = {meeting_id}
        while True:
            next_node = None
            for _, tgt, data in self.g.out_edges(current, data=True):
                if data.get("relation") == "ESCALATES_FROM" and tgt not in seen:
                    next_node = tgt
                    break
            if not next_node:
                break
            chain.append(next_node)
            seen.add(next_node)
            current = next_node
        return [{"meeting_id": m, "title": self.g.nodes[m].get("title")} for m in chain]

    def person_network(self, person: str) -> dict:
        """Meetings a person attended, opportunities they touch, and
        co-attendees (their internal/external network)."""
        if person not in self.g:
            return {"meetings": [], "opportunities": [], "co_attendees": []}
        meetings = [t for t in self.g.successors(person)
                    if self.g.nodes[t].get("type") == "Meeting"]
        opps = set()
        co_attendees = set()
        for m in meetings:
            for _, tgt, data in self.g.out_edges(m, data=True):
                if data.get("relation") == "DISCUSSED":
                    opps.add(tgt)
            for src, _, data in self.g.in_edges(m, data=True):
                if data.get("relation") == "PARTICIPATED_IN" and src != person:
                    co_attendees.add(src)
        return {
            "meetings": meetings,
            "opportunities": list(opps),
            "co_attendees": list(co_attendees),
        }

    def high_risk_opportunities(self) -> list[dict]:
        """Opportunities where recent meetings flagged risk factors --
        a graph-native aggregation query."""
        results = []
        for node, data in self.g.nodes(data=True):
            if data.get("type") != "Opportunity":
                continue
            meetings = [n for n in self.g.predecessors(node)
                        if self.g.nodes[n].get("type") == "Meeting"]
            risky = [m for m in meetings
                     if self.g.nodes[m].get("risk_flag") not in (None, "no risk identified")]
            if meetings and len(risky) / len(meetings) >= 0.5:
                results.append({
                    "opportunity_id": node,
                    "name": data.get("name"),
                    "stage": data.get("stage"),
                    "value_usd": data.get("value_usd"),
                    "risk_ratio": round(len(risky) / len(meetings), 2),
                    "meeting_count": len(meetings),
                })
        results.sort(key=lambda x: x["risk_ratio"], reverse=True)
        return results

    def stats(self) -> dict:
        type_counts = {}
        for _, data in self.g.nodes(data=True):
            t = data.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "node_types": type_counts,
        }
