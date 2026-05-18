#!/usr/bin/env python3
"""Export Neo4j graph to JSON for 3D visualization."""

import json
import sys
from collections import defaultdict
from neo4j import GraphDatabase

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "cascades2024"))

with driver.session() as session:
    # Get all nodes with their labels and degree (connection count)
    print("Querying all nodes with degree...")
    nodes_data = session.run("""
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree, labels(n) AS lbls
        RETURN id(n) AS id, n.name AS name, lbls, degree
        ORDER BY degree DESC
    """).data()
    
    print(f"  Got {len(nodes_data)} nodes")
    
    # Get all relationships (sample if too many)
    print("Querying relationships...")
    rels_data = session.run("""
        MATCH (a)-[r]->(b)
        RETURN id(a) AS source, id(b) AS target, type(r) AS type
    """).data()
    
    print(f"  Got {len(rels_data)} relationships")

driver.close()

# Build visualization JSON
nodes = []
for nd in nodes_data:
    label = nd["lbls"][0] if nd["lbls"] else "Unknown"
    name = nd.get("name") or f"Node_{nd['id']}"
    nodes.append({
        "id": nd["id"],
        "name": name[:60],  # Truncate long names
        "label": label,
        "degree": nd["degree"],
    })

links = []
for rd in rels_data:
    links.append({
        "source": rd["source"],
        "target": rd["target"],
        "type": rd["type"],
    })

graph = {"nodes": nodes, "links": links}

output_path = r"E:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs\reports\graph_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(graph, f)

print(f"\nExported {len(nodes)} nodes + {len(links)} links to {output_path}")
print(f"File size: {len(json.dumps(graph)) / 1024 / 1024:.1f} MB")

# Stats
label_counts = defaultdict(int)
for n in nodes:
    label_counts[n["label"]] += 1

print("\nNode types:")
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {count}")

top_degree = sorted(nodes, key=lambda x: -x["degree"])[:20]
print("\nTop 20 most connected:")
for n in top_degree:
    print(f"  {n['name'][:40]:40s}  {n['label']:20s}  degree={n['degree']}")
