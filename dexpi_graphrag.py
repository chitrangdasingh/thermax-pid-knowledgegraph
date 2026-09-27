"""DEXPI-compatible graph loading, local Graph-RAG and Neo4j export.

The supplied Thermax DXF is not a DEXPI/Proteus file. Its graph is therefore
marked PROPOSED_MAPPING. A licensed DEXPI Proteus XML can be loaded through
pyDEXPI and retains a separate DEXPI_1_3 source status.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STOPWORDS = {"THE", "AND", "FOR", "WHAT", "WHICH", "WHO", "HOW", "MANY", "ARE", "IS", "TO", "OF", "IN", "ON", "WITH", "FROM", "GIVE", "SHOW", "ME"}


def _native(value: Any):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def _question_tokens(question: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_-]+", question.upper()) if token not in STOPWORDS and (len(token) >= 3 or any(ch.isdigit() for ch in token))}


def build_asset_graph(assets: pd.DataFrame, relationships: pd.DataFrame) -> nx.MultiDiGraph:
    """Create the high-level property graph used by the local dashboard."""
    graph = nx.MultiDiGraph(
        source_format="DXF_FLATTENED_VECTOR",
        dexpi_status="PROPOSED_MAPPING_NOT_DEXPI_COMPLIANT",
        review_status="ENGINEERING_REVIEW_REQUIRED",
    )
    for row in assets.to_dict("records"):
        asset_id = str(row["asset_id"])
        graph.add_node(
            asset_id,
            kind="asset",
            label=str(row.get("tag", asset_id)),
            tag=str(row.get("tag", "")),
            asset_class=str(row.get("asset_class", "Unknown")),
            confidence=float(row.get("overall_confidence", 0) or 0),
            review_status=str(row.get("review_status", "ENGINEERING_REVIEW_REQUIRED")),
            source_id=f"DXF_ASSET:{asset_id}",
            x=float(row.get("x", 0) or 0),
            y=float(row.get("y", 0) or 0),
        )
    for index, row in enumerate(relationships.to_dict("records"), start=1):
        source, target = str(row["source_id"]), str(row["target_id"])
        if source not in graph or target not in graph:
            continue
        graph.add_edge(
            source, target, key=f"domain-{index}", relation=str(row.get("relationship", "CONNECTED_TO_CANDIDATE")),
            relation_group="domain", confidence=float(row.get("confidence", 0) or 0),
            direction=str(row.get("direction", "UNKNOWN")), evidence=str(row.get("evidence", "")),
            review_status=str(row.get("review_status", "ENGINEERING_REVIEW_REQUIRED")),
            source_id=f"DXF_RELATION:{index}",
        )

    by_class: dict[str, list[str]] = {}
    for node, attrs in graph.nodes(data=True):
        by_class.setdefault(str(attrs.get("asset_class", "Unknown")), []).append(node)
    for asset_class, nodes in by_class.items():
        if len(nodes) < 2:
            continue
        for left, right in zip(sorted(nodes), sorted(nodes)[1:]):
            graph.add_edge(
                left, right, key=f"lexical-class-{left}-{right}", relation="SAME_ASSET_CLASS",
                relation_group="lexical", confidence=1.0, direction="NOT_APPLICABLE",
                evidence=f"Both candidates have class {asset_class}", review_status="DERIVED_LEXICAL_RELATION",
                source_id="LOCAL_GRAPH:CLASS_GROUPING",
            )
    return graph


def load_current_graph(data_dir: str | Path = DATA) -> nx.MultiDiGraph:
    data = Path(data_dir)
    return build_asset_graph(pd.read_csv(data / "asset_master.csv"), pd.read_csv(data / "relationships.csv"))


def load_dexpi_proteus(xml_path: str | Path) -> dict[str, nx.MultiDiGraph]:
    """Load a true DEXPI 1.3 Proteus XML with pyDEXPI.

    pyDEXPI's own GraphAbstractor is used rather than pretending the flattened
    DXF is DEXPI. Optional dependency: ``pip install -r requirements-graphrag.txt``.
    """
    try:
        from pydexpi.loaders import GraphAbstractor, GraphLoader, ProteusSerializer
    except ImportError as exc:
        raise RuntimeError("Install requirements-graphrag.txt to load DEXPI Proteus XML") from exc
    source = Path(xml_path).resolve()
    if source.suffix.lower() != ".xml" or not source.is_file():
        raise ValueError("A readable DEXPI Proteus .xml file is required")
    model = ProteusSerializer().load(str(source.parent), source.name)
    plant = GraphLoader().parse_dexpi_to_graph(model)
    process = GraphAbstractor.build_process_graph(plant)
    conceptual = GraphAbstractor.build_conceptual_graph(plant)
    for graph, level in [(plant, "plant"), (process, "process"), (conceptual, "conceptual")]:
        graph.graph.update(source_format="DEXPI_PROTEUS_XML", dexpi_status="DEXPI_1_3", abstraction=level)
    return {"plant": plant, "process": process, "conceptual": conceptual}


def condensed_graph(graph: nx.MultiDiGraph, include_lexical: bool = False) -> nx.MultiDiGraph:
    """Keep high-information asset nodes and reviewed/candidate process edges."""
    selected = [
        node for node, attrs in graph.nodes(data=True)
        if str(attrs.get("kind", "")).lower() == "asset"
        or any(token in str(attrs.get("type", attrs.get("class", ""))).lower()
               for token in ("equipment", "instrument", "valve", "pump", "vessel", "reactor"))
    ]
    if not selected:  # A pyDEXPI conceptual graph is already condensed.
        selected = list(graph.nodes)
    result = nx.MultiDiGraph(**{key: _native(value) for key, value in graph.graph.items()})
    for node in selected:
        result.add_node(str(node), **{key: _native(value) for key, value in graph.nodes[node].items()})
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        if source not in selected or target not in selected:
            continue
        if not include_lexical and attrs.get("relation_group") == "lexical":
            continue
        result.add_edge(str(source), str(target), key=str(key), **{k: _native(v) for k, v in attrs.items()})
    return result


def graph_context(graph: nx.MultiDiGraph, question: str, max_hops: int = 2, max_nodes: int = 80) -> dict:
    tokens = _question_tokens(question)
    scores: list[tuple[int, str]] = []
    for node, attrs in graph.nodes(data=True):
        text = " ".join(str(attrs.get(key, "")) for key in ("tag", "label", "asset_class", "kind")).upper()
        scores.append((sum(token in text for token in tokens), str(node)))
    seeds = [node for score, node in sorted(scores, reverse=True) if score > 0][:8]
    if not seeds:
        seeds = list(graph.nodes)[: min(20, max_nodes)]
    undirected = graph.to_undirected()
    selected: set[str] = set()
    for seed in seeds:
        selected.update(str(node) for node, depth in nx.single_source_shortest_path_length(undirected, seed, cutoff=max_hops).items())
        if len(selected) >= max_nodes:
            break
    selected = set(sorted(selected)[:max_nodes])
    nodes = [{"id": node, **{k: _native(v) for k, v in graph.nodes[node].items()}} for node in selected]
    edges = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        if str(source) in selected and str(target) in selected:
            edges.append({"source": str(source), "target": str(target), "key": str(key), **{k: _native(v) for k, v in attrs.items()}})
    return {
        "question": question, "graph_status": dict(graph.graph), "seed_nodes": seeds,
        "nodes": nodes, "edges": edges, "context_policy": f"keyword seeds + {max_hops}-hop neighbourhood",
    }


def _asset_evidence(node: str, attrs: dict) -> dict:
    return {"node_id": node, "tag": attrs.get("tag", attrs.get("label", "")),
            "asset_class": attrs.get("asset_class", attrs.get("type", "")),
            "confidence": attrs.get("confidence", ""), "source_id": attrs.get("source_id", "")}


def local_answer(question: str, graph: nx.MultiDiGraph) -> dict:
    """Deterministic answers for common P&ID questions; never invent live values."""
    query = question.strip()
    lowered = query.lower()
    assets = [(str(node), attrs) for node, attrs in graph.nodes(data=True)]
    query_tokens = {token.lower() for token in _question_tokens(query)}
    ranked = []
    for node, attrs in assets:
        tag = str(attrs.get("tag", attrs.get("label", ""))).lower()
        text = f"{tag} {attrs.get('asset_class','')}".lower()
        exact_tag = int(bool(tag and re.search(rf"(?<![a-z0-9]){re.escape(tag)}(?![a-z0-9])", lowered)))
        score = sum(token in text for token in query_tokens)
        if exact_tag or score:
            ranked.append((exact_tag, score, node, attrs))
    matched = [(node, attrs) for _, _, node, attrs in sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)]
    evidence = [_asset_evidence(node, attrs) for node, attrs in matched[:20]]

    if "methanol" in lowered and ("conversion" in lowered or "convert" in lowered):
        return {
            "answer": (
                "Methanol conversion cannot be calculated from this static P&ID. On a molar basis use "
                "X_MeOH = (F_in·z_MeOH,in − F_out·z_MeOH,out) / (F_in·z_MeOH,in). "
                "Required live, time-aligned inputs are inlet/outlet total molar flow, inlet/outlet methanol "
                "mole fraction, units, timestamps, quality flags and operating-state context. The simplified "
                "composition-only formula is valid only after engineering confirms equal total molar flow."
            ),
            "answer_type": "PHYSICS_FORMULA_AND_MISSING_DATA", "evidence": evidence,
            "review_status": "PROCESS_ENGINEERING_VALIDATION_REQUIRED",
        }
    if any(word in lowered for word in ("hazop", "safety", "hazard", "risk")):
        prompts = [
            "Verify overpressure protection and relief path for every heated/blocked-in volume.",
            "Confirm loss-of-flow, high-temperature and cooling-loss trips against the cause-and-effect matrix.",
            "Check isolation, purge, vent, drain and sampling provisions against the approved line list and SOP.",
            "Validate instrument independence, alarm set points and proof-test requirements with process safety.",
        ]
        return {
            "answer": "Draft engineering-review prompts: " + " ".join(f"{i+1}) {item}" for i, item in enumerate(prompts)),
            "answer_type": "HAZOP_REVIEW_PROMPTS_NOT_CONCLUSIONS", "evidence": evidence,
            "review_status": "FORMAL_HAZOP_AND_PHA_REQUIRED",
        }
    if "how many" in lowered or "count" in lowered:
        class_names = sorted({str(attrs.get("asset_class", "")) for _, attrs in assets}, key=len, reverse=True)
        selected_class = next((name for name in class_names if name and name.lower() in lowered), None)
        rows = [(node, attrs) for node, attrs in assets if not selected_class or str(attrs.get("asset_class")) == selected_class]
        label = selected_class or "asset candidate"
        return {
            "answer": f"The high-level graph contains {len(rows)} {label}{'' if len(rows)==1 else 's'}.",
            "answer_type": "COUNT", "evidence": [_asset_evidence(node, attrs) for node, attrs in rows],
            "review_status": graph.graph.get("review_status", "ENGINEERING_REVIEW_REQUIRED"),
        }
    if any(word in lowered for word in ("connect", "upstream", "downstream", "path")) and matched:
        node, attrs = matched[0]
        connections = []
        for source, target, key, edge in graph.edges(keys=True, data=True):
            if node not in {str(source), str(target)} or edge.get("relation_group") == "lexical":
                continue
            other = str(target) if str(source) == node else str(source)
            connections.append({"other": other, "tag": graph.nodes[other].get("tag", other),
                                "relation": edge.get("relation"), "direction": edge.get("direction"),
                                "confidence": edge.get("confidence"), "source_id": edge.get("source_id")})
        return {
            "answer": f"{attrs.get('tag', node)} has {len(connections)} candidate process connection(s). Flow direction is not established unless an edge is engineer-approved.",
            "answer_type": "CANDIDATE_CONNECTIONS", "connections": connections,
            "evidence": [_asset_evidence(node, attrs)], "review_status": "ENGINEERING_REVIEW_REQUIRED",
        }
    if matched:
        node, attrs = matched[0]
        return {
            "answer": f"{attrs.get('tag', node)} is a candidate {attrs.get('asset_class', 'asset')} with source confidence {float(attrs.get('confidence', 0)):.1%}. It remains under engineering review.",
            "answer_type": "ASSET_LOOKUP", "evidence": [_asset_evidence(node, attrs)],
            "review_status": attrs.get("review_status", "ENGINEERING_REVIEW_REQUIRED"),
        }
    return {
        "answer": f"The graph contains {graph.number_of_nodes()} asset candidates and {sum(1 for *_, edge in graph.edges(data=True) if edge.get('relation_group') != 'lexical')} candidate process relationships. Ask for a tag, class, connection, count, methanol conversion inputs or safety review prompts.",
        "answer_type": "GRAPH_SUMMARY", "evidence": [],
        "review_status": graph.graph.get("review_status", "ENGINEERING_REVIEW_REQUIRED"),
    }


def gemini_answer(question: str, context: dict) -> dict:
    """Optional Google Gemini call; context is bounded and the result is unverified."""
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install requirements-graphrag.txt for the optional Gemini adapter") from exc
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not api_key or not model:
        raise RuntimeError("GEMINI_API_KEY and GEMINI_MODEL must be configured server-side")
    prompt = (
        "You answer questions about one P&ID graph. Use only the JSON evidence below. "
        "Cite node_id/source_id values in the answer. Distinguish candidate from approved relationships. "
        "Never invent live values, specifications, flow direction or operating history. For HAZOP/safety, "
        "provide review prompts only and require formal engineering review.\n\n"
        f"Question: {question}\nEvidence JSON:\n{json.dumps(context, default=str)}"
    )
    response = genai.Client(api_key=api_key).models.generate_content(model=model, contents=prompt)
    return {"answer": response.text, "answer_type": "GEMINI_GRAPH_RAG_UNVERIFIED",
            "review_status": "HUMAN_VERIFICATION_REQUIRED", "context_nodes": len(context.get("nodes", []))}


def export_graph(graph: nx.MultiDiGraph, output_dir: str | Path) -> dict:
    """Write portable JSON/GraphML plus import-ready Neo4j node/edge CSVs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nodes = []
    for node, attrs in graph.nodes(data=True):
        nodes.append({"node_id": str(node), "labels": "Asset", "properties_json": json.dumps({k: _native(v) for k, v in attrs.items()}, sort_keys=True)})
    edges = []
    for index, (source, target, key, attrs) in enumerate(graph.edges(keys=True, data=True), start=1):
        edges.append({"edge_id": f"EDGE-{index:05d}", "source_id": str(source), "target_id": str(target),
                      "relationship": str(attrs.get("relation", "RELATED_TO")),
                      "properties_json": json.dumps({k: _native(v) for k, v in attrs.items()}, sort_keys=True)})
    pd.DataFrame(nodes).to_csv(output / "neo4j_nodes.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(edges).to_csv(output / "neo4j_edges.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    payload = {"graph": {k: _native(v) for k, v in graph.graph.items()},
               "nodes": [{"id": str(node), **{k: _native(v) for k, v in attrs.items()}} for node, attrs in graph.nodes(data=True)],
               "edges": [{"source": str(s), "target": str(t), "key": str(k), **{x: _native(y) for x, y in attrs.items()}} for s, t, k, attrs in graph.edges(keys=True, data=True)]}
    (output / "high_level_graph.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    nx.write_graphml(graph, output / "high_level_graph.graphml")
    (output / "neo4j_import.cypher").write_text(
        "// Run with Neo4j LOAD CSV after copying CSVs to the configured import directory.\n"
        "CREATE CONSTRAINT asset_id_unique IF NOT EXISTS FOR (n:Asset) REQUIRE n.node_id IS UNIQUE;\n"
        "LOAD CSV WITH HEADERS FROM 'file:///neo4j_nodes.csv' AS row\n"
        "MERGE (n:Asset {node_id: row.node_id}) SET n.properties_json = row.properties_json;\n"
        "LOAD CSV WITH HEADERS FROM 'file:///neo4j_edges.csv' AS row\n"
        "MATCH (a:Asset {node_id: row.source_id}), (b:Asset {node_id: row.target_id})\n"
        "MERGE (a)-[r:RELATED_TO {edge_id: row.edge_id}]->(b)\n"
        "SET r.relationship = row.relationship, r.properties_json = row.properties_json;\n",
        encoding="utf-8",
    )
    return {"nodes": len(nodes), "edges": len(edges), "output_dir": str(output)}


if __name__ == "__main__":
    graph = condensed_graph(load_current_graph(), include_lexical=True)
    print(json.dumps(export_graph(graph, DATA / "graphrag"), indent=2))
