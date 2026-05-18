#!/usr/bin/env python3
"""
Graph-Powered Self-Synthesizer: Mines the Neo4j Knowledge Graph
to generate dense Q&A training pairs for CASCADES parametric memory.

This replaces regex-based fact extraction with structured graph queries.
50K nodes + 85K relationships → thousands of factual Q&A pairs.

Usage:
    python graph_synthesizer.py                    # Generate all Q&A pairs
    python graph_synthesizer.py --limit 500        # Cap at 500 pairs
    python graph_synthesizer.py --output train.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from neo4j import GraphDatabase

# ── Configuration ────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "cascades2024"
IDENTITY = "Bender1011001"
OUTPUT_FILE = Path(r"E:\digital-twin\training_data\graph_synthesized_qa.jsonl")


def query_graph(driver) -> dict:
    """Query Neo4j for all facts about the user, organized by relationship type."""
    results = defaultdict(list)

    with driver.session() as session:
        # First get all relationship types and counts
        rel_counts = session.run("""
            MATCH (p:Person {name: $name})-[r]->(n)
            RETURN type(r) AS rel, count(*) AS cnt
            ORDER BY cnt DESC
        """, name=IDENTITY).data()

        print(f"  Relationship types:")
        for rc in rel_counts:
            print(f"    {rc['rel']}: {rc['cnt']}")

        # Query each relationship type with appropriate limits
        for rc in rel_counts:
            rel_type = rc["rel"]
            count = rc["cnt"]

            # Sample VISITED (too many), take all for everything else
            limit = 200 if rel_type == "VISITED" else 5000

            records = session.run(f"""
                MATCH (p:Person {{name: $name}})-[r:{rel_type}]->(n)
                RETURN n.name AS node_name, labels(n) AS node_labels,
                       n.url AS node_url
                LIMIT {limit}
            """, name=IDENTITY).data()

            for rec in records:
                name = rec.get("node_name", "")
                if name:
                    results[rel_type].append({
                        "name": name,
                        "labels": rec.get("node_labels", []),
                        "url": rec.get("node_url", "") or "",
                    })

        # Get incoming relationships
        in_counts = session.run("""
            MATCH (n)-[r]->(p:Person {name: $name})
            RETURN type(r) AS rel, count(*) AS cnt
        """, name=IDENTITY).data()

        for rc in in_counts:
            rel_type = rc["rel"]
            records = session.run(f"""
                MATCH (n)-[r:{rel_type}]->(p:Person {{name: $name}})
                RETURN n.name AS node_name, labels(n) AS node_labels
                LIMIT 500
            """, name=IDENTITY).data()

            for rec in records:
                name = rec.get("node_name", "")
                if name:
                    results["RECEIVED_" + rel_type].append({
                        "name": name,
                        "labels": rec.get("node_labels", []),
                    })

        # Graph stats
        stats = session.run("MATCH (n) RETURN count(n) AS nodes").single()
        rel_stats = session.run("MATCH ()-[r]->() RETURN count(r) AS rels").single()
        label_stats = session.run("""
            MATCH (n) UNWIND labels(n) AS label
            RETURN label, count(*) AS cnt ORDER BY cnt DESC
        """).data()

        results["_stats"] = {
            "nodes": stats["nodes"],
            "rels": rel_stats["rels"],
            "labels": {r["label"]: r["cnt"] for r in label_stats},
        }

    return dict(results)


def synthesize_qa_from_graph(graph_data: dict) -> list[dict]:
    """Convert graph relationships into dense Q&A training pairs."""
    qa_pairs = []
    stats = graph_data.pop("_stats", {})

    # ── Noise filters — junk that wastes CASCADES capacity ──
    NOISE_PATTERNS = [
        "npmjs.org", ".tgz", "registry.", "node_modules",
        "hitehouse.gov",  # Typos in data
        "Password", "Sendername", "Recipient Email",
        "8192X4096", "Pass",
    ]

    def is_noise(name: str) -> bool:
        return any(p.lower() in name.lower() for p in NOISE_PATTERNS)

    # ── Verb conjugation for natural answers ──
    VERB_MAP = {
        "USES": "uses",
        "RESEARCHES": "researches",
        "VISITED": "has visited",
        "SEARCHED_FOR": "has searched for",
        "INTERESTED_IN": "is interested in",
        "CONTRIBUTES_TO": "contributes to",
        "KNOWS": "knows",
        "CONFIGURES": "configures",
        "DEBUGS": "has debugged",
        "INSTALLED": "has installed",
        "PURCHASING": "has purchased or considered",
    }

    # ── Relationship-specific Q&A templates ──────────────────────────
    templates = {
        "USES": {
            "questions": [
                "What tools and technologies does {identity} use?",
                "What does {identity} use for their work?",
                "List the tools {identity} uses.",
            ],
            "answer_prefix": "{identity} uses the following tools and technologies: ",
        },
        "RESEARCHES": {
            "questions": [
                "What does {identity} research?",
                "What topics is {identity} interested in researching?",
                "What are {identity}'s research areas?",
            ],
            "answer_prefix": "{identity} researches: ",
        },
        "VISITED": {
            "questions": [
                "What websites does {identity} visit?",
                "What sites does {identity} frequent?",
                "List the websites {identity} has visited.",
            ],
            "answer_prefix": "{identity} visits these websites: ",
        },
        "SEARCHED_FOR": {
            "questions": [
                "What has {identity} searched for?",
                "What are {identity}'s search interests?",
                "What topics does {identity} search about?",
            ],
            "answer_prefix": "{identity} has searched for: ",
        },
        "INTERESTED_IN": {
            "questions": [
                "What is {identity} interested in?",
                "What are {identity}'s interests?",
                "What hobbies or interests does {identity} have?",
            ],
            "answer_prefix": "{identity} is interested in: ",
        },
        "CONTRIBUTES_TO": {
            "questions": [
                "What projects does {identity} contribute to?",
                "What does {identity} work on?",
                "List {identity}'s projects.",
            ],
            "answer_prefix": "{identity} contributes to these projects: ",
        },
        "KNOWS": {
            "questions": [
                "Who does {identity} know?",
                "Who are {identity}'s contacts?",
                "List the people {identity} knows.",
            ],
            "answer_prefix": "{identity} knows: ",
        },
    }

    for rel_type, items in graph_data.items():
        if rel_type.startswith("_") or not items:
            continue

        names = [item["name"] for item in items if item.get("name")]
        if not names:
            continue

        # Deduplicate, clean, and filter noise
        unique_names = [n for n in dict.fromkeys(names) if not is_noise(n)]
        if not unique_names:
            continue

        verb = VERB_MAP.get(rel_type, rel_type.lower().replace("_", " ") + "s")

        # Get template or use generic
        tmpl = templates.get(rel_type, {
            "questions": [
                f"What is related to {{identity}} via {rel_type}?",
                f"Tell me about {{identity}}'s {rel_type.lower().replace('_', ' ')} connections.",
            ],
            "answer_prefix": f"{{identity}} {verb}: ",
        })

        # ── Strategy 1: Aggregate answer (grouped with UNIQUE prompts) ──
        chunks = []
        for chunk_start in range(0, len(unique_names), 10):
            chunks.append(unique_names[chunk_start:chunk_start + 10])

        for chunk_idx, chunk in enumerate(chunks):
            items_text = ", ".join(chunk)

            for q_template in tmpl["questions"]:
                question = q_template.format(identity=IDENTITY)
                # Fix prompt collision: append part number for multi-chunk rels
                if len(chunks) > 1:
                    question += f" (Part {chunk_idx + 1})"
                answer = tmpl["answer_prefix"].format(identity=IDENTITY) + items_text + "."
                qa_pairs.append({"prompt": question, "response": answer})

        # ── Strategy 2: Individual facts (one per item, for high-value entities) ──
        if len(unique_names) <= 30:
            for item in items[:30]:
                name = item.get("name", "")
                if not name or is_noise(name):
                    continue
                label = item.get("labels", ["Entity"])[0] if item.get("labels") else "Entity"
                url = item.get("url", "")

                fact_parts = [f"{name}"]
                if url:
                    fact_parts.append(f"(URL: {url})")
                fact_text = " ".join(fact_parts)

                qa_pairs.append({
                    "prompt": f"What is {name} in relation to {IDENTITY}?",
                    "response": f"{name} is a {label} that {IDENTITY} {verb}. {fact_text}.",
                })

        # ── Strategy 3: Category-level summaries ──
        label_groups = defaultdict(list)
        for item in items:
            if is_noise(item.get("name", "")):
                continue
            for lbl in item.get("labels", ["Entity"]):
                label_groups[lbl].append(item["name"])

        for label, label_names in label_groups.items():
            unique_label_names = list(dict.fromkeys(label_names))[:20]
            if unique_label_names:
                qa_pairs.append({
                    "prompt": f"What {label}s does {IDENTITY} {verb.rstrip('s')}?",
                    "response": f"{IDENTITY} {verb} these {label}s: {', '.join(unique_label_names)}.",
                })

    # ── Identity-level facts ──
    qa_pairs.append({
        "prompt": f"Who is {IDENTITY}?",
        "response": (
            f"{IDENTITY} is a person whose digital footprint includes "
            f"{stats.get('nodes', '?')} entities and {stats.get('rels', '?')} "
            f"relationships in their knowledge graph. Their activities span "
            f"multiple domains including software development, AI research, "
            f"web browsing, and various online services."
        ),
    })

    # ── Hardcoded identity facts (highest priority) ──
    identity_facts = [
        (f"What are {IDENTITY}'s email addresses?",
         f"{IDENTITY}'s email addresses are andrewdarcy530@gmail.com and andrewdarcy707@live.com."),
        (f"What is {IDENTITY}'s real name?",
         f"{IDENTITY}'s real name is Andrew Darcy."),
        ("What is the user's name?",
         "The user's name is Andrew, also known online as Bender1011001."),
        ("Who am I?",
         "You are Andrew (Bender1011001)."),
        ("What GPU does the user have?",
         "The user has an RTX 4060 Ti with 8GB VRAM."),
        ("What is CASCADES?",
         "CASCADES is a continual learning PEFT method that the user is developing for local LLMs. "
         "It uses Riemannian optimization on the Stiefel manifold to isolate learned knowledge into "
         "orthogonal subspaces, preventing catastrophic forgetting."),
        ("What are the user's main projects?",
         "The user's main projects are CASCADES (continual PEFT for LLMs), "
         "a Tail Risk Prediction Engine for event derivatives, "
         "and an astrology website called AstroForge."),
    ]
    for q, a in identity_facts:
        qa_pairs.append({"prompt": q, "response": a})

    # Shuffle to prevent ordering bias during training
    random.shuffle(qa_pairs)
    return qa_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Graph-Powered Self-Synthesizer")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    parser.add_argument("--limit", type=int, default=0, help="Max Q&A pairs (0=all)")
    parser.add_argument("--preview", type=int, default=0, help="Print N sample pairs")
    args = parser.parse_args()

    print("=" * 60)
    print("Graph-Powered Self-Synthesizer")
    print("Mining Neo4j Knowledge Graph -> Dense Q&A Training Data")
    print("=" * 60)

    # Connect to Neo4j
    print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    # Query graph
    print("Querying knowledge graph...")
    graph_data = query_graph(driver)

    # Print relationship summary
    print(f"\nRelationship types found:")
    for rel, items in sorted(graph_data.items(), key=lambda x: -len(x[1]) if isinstance(x[1], list) else 0):
        if rel.startswith("_"):
            continue
        print(f"  {rel}: {len(items)} connections")

    # Synthesize Q&A pairs
    print(f"\nSynthesizing Q&A pairs...")
    qa_pairs = synthesize_qa_from_graph(graph_data)

    if args.limit > 0:
        qa_pairs = qa_pairs[:args.limit]

    print(f"Generated {len(qa_pairs)} Q&A training pairs")

    # Preview
    if args.preview > 0:
        print(f"\n{'='*60}")
        print(f"SAMPLE Q&A PAIRS (showing {args.preview})")
        print(f"{'='*60}")
        for qa in qa_pairs[:args.preview]:
            print(f"\n  Q: {qa['prompt']}")
            print(f"  A: {qa['response'][:200]}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output_path}")
    print(f"   {len(qa_pairs)} Q&A pairs, {output_path.stat().st_size / 1024:.1f} KB")

    driver.close()


if __name__ == "__main__":
    main()
