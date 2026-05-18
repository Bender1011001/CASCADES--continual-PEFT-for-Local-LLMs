#!/usr/bin/env python3
"""
Knowledge Graph Builder — Semantic Entity Extraction → Neo4j

Reads Google Takeout text chunks and generates idempotent Cypher MERGE
statements for building a personal knowledge graph in Neo4j.

This script uses rule-based + pattern-based extraction to generate
valid Cypher without requiring an external LLM. For LLM-assisted
extraction, see the CASCADES server pipeline (local_extract_cascades.py).

Node Labels:
    Person, Project, Concept, Hardware, Software, Location,
    Organization, Website, Interest, Skill, DigitalArtifact,
    SearchQuery, VideoContent

Relationship Types:
    RESEARCHES, USES, PURCHASING, CONTRIBUTES_TO, INTERESTED_IN,
    KNOWS, VISITED, SEARCHED_FOR, WATCHED, WORKS_AT, DEBUGS,
    INSTALLED, CONFIGURED, CONTACTED, LOCATED_IN, EXTRACTED_FROM

Identity Mapping:
    andrewdarcy530@gmail.com → Person "Bender1011001"
    andrewdarcy707@live.com  → Person "Bender1011001"
    Any user activity        → Person "Bender1011001"

Usage:
    python build_knowledge_graph.py                  # Process all chunks
    python build_knowledge_graph.py --test           # Process 1 chunk
    python build_knowledge_graph.py --dry-run        # Generate Cypher only (no Neo4j)
    python build_knowledge_graph.py --neo4j-uri bolt://localhost:7687
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ── Configuration ────────────────────────────────────────────────────
CHUNKS_DIR = Path(r"E:\digital-twin\takeout_chunks")
CYPHER_DIR = Path(r"E:\digital-twin\cypher_output")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "cascades2024"  # Change this after setup

# Identity anchors
IDENTITY_EMAILS = {
    "andrewdarcy530@gmail.com",
    "andrewdarcy707@live.com",
}
IDENTITY_NODE = "Bender1011001"


# ── Entity Extraction Patterns ──────────────────────────────────────

# URL domain → organization/website mapping
KNOWN_DOMAINS = {
    "github.com": ("Organization", "GitHub"),
    "stackoverflow.com": ("Organization", "Stack Overflow"),
    "reddit.com": ("Organization", "Reddit"),
    "youtube.com": ("Organization", "YouTube"),
    "amazon.com": ("Organization", "Amazon"),
    "google.com": ("Organization", "Google"),
    "huggingface.co": ("Organization", "Hugging Face"),
    "twitter.com": ("Organization", "Twitter/X"),
    "x.com": ("Organization", "Twitter/X"),
    "discord.com": ("Organization", "Discord"),
    "nvidia.com": ("Organization", "NVIDIA"),
    "pytorch.org": ("Organization", "PyTorch"),
    "arxiv.org": ("Organization", "arXiv"),
    "wikipedia.org": ("Organization", "Wikipedia"),
    "neo4j.com": ("Organization", "Neo4j"),
    "stripe.com": ("Organization", "Stripe"),
    "polymarket.com": ("Organization", "Polymarket"),
    "kalshi.com": ("Organization", "Kalshi"),
}

# Tech keywords → category
TECH_KEYWORDS = {
    # Hardware
    "rtx 4060": "Hardware", "rtx 3060": "Hardware", "rtx 4090": "Hardware",
    "a100": "Hardware", "h100": "Hardware", "t4": "Hardware",
    "raspberry pi": "Hardware", "esp32": "Hardware", "arduino": "Hardware",
    "4060ti": "Hardware", "4060 ti": "Hardware",
    "ssd": "Hardware", "nvme": "Hardware", "gpu": "Hardware",
    
    # Software/Frameworks
    "pytorch": "Software", "tensorflow": "Software", "transformers": "Software",
    "neo4j": "Software", "fastapi": "Software", "react": "Software",
    "next.js": "Software", "nextjs": "Software", "node.js": "Software",
    "python": "Software", "rust": "Software", "javascript": "Software",
    "docker": "Software", "kubernetes": "Software", "linux": "Software",
    "kali": "Software", "windows": "Software", "ubuntu": "Software",
    "vscode": "Software", "vim": "Software",
    "bitsandbytes": "Software", "qlora": "Software", "lora": "Software",
    "huggingface": "Software",
    
    # Concepts
    "machine learning": "Concept", "deep learning": "Concept",
    "llm": "Concept", "large language model": "Concept",
    "fine-tuning": "Concept", "fine tuning": "Concept",
    "rag": "Concept", "retrieval augmented": "Concept",
    "knowledge graph": "Concept", "graph database": "Concept",
    "neural network": "Concept", "transformer": "Concept",
    "continual learning": "Concept", "catastrophic forgetting": "Concept",
    "quantization": "Concept", "abliteration": "Concept",
    "stiefel manifold": "Concept", "riemannian": "Concept",
    "blockchain": "Concept", "cryptocurrency": "Concept",
    "smart contract": "Concept",
    "cybersecurity": "Concept", "penetration testing": "Concept",
    "astrology": "Concept",
    "sar": "Concept", "synthetic aperture radar": "Concept",
}

# Project keywords
PROJECT_KEYWORDS = {
    "cascades": "CASCADES",
    "m-a-k-e-r": "M-A-K-E-R",
    "maker": "M-A-K-E-R",
    "astroforge": "AstroForge",
    "digital twin": "Digital Twin",
    "obliteratus": "OBLITERATUS",
    "tail risk": "Tail Risk Engine",
}


def escape_cypher(text: str) -> str:
    """Escape text for safe inclusion in Cypher string literals."""
    return (text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "")
            .replace("\t", " ")
            .strip())


def extract_urls(text: str) -> list[dict]:
    """Extract URLs and classify them."""
    url_pattern = r'https?://[^\s<>"\')\]]+[^\s<>"\')\].,;:!?]'
    urls = re.findall(url_pattern, text)
    
    results = []
    seen = set()
    
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().lstrip("www.")
        except Exception:
            continue
        
        # Map to known organizations
        org_info = None
        for known_domain, info in KNOWN_DOMAINS.items():
            if known_domain in domain:
                org_info = info
                break
        
        # Extract page title from path
        path_parts = [p for p in parsed.path.split("/") if p]
        page_title = " / ".join(path_parts[-2:]) if path_parts else domain
        
        results.append({
            "url": url,
            "domain": domain,
            "org_label": org_info[0] if org_info else "Website",
            "org_name": org_info[1] if org_info else domain,
            "page_title": page_title,
        })
    
    return results


def extract_tech_entities(text: str) -> list[dict]:
    """Extract technology-related entities from text."""
    text_lower = text.lower()
    entities = []
    seen = set()
    
    for keyword, category in TECH_KEYWORDS.items():
        if keyword in text_lower and keyword not in seen:
            seen.add(keyword)
            entities.append({
                "name": keyword.title() if len(keyword) > 3 else keyword.upper(),
                "category": category,
                "raw": keyword,
            })
    
    return entities


def extract_projects(text: str) -> list[dict]:
    """Extract project references from text."""
    text_lower = text.lower()
    projects = []
    seen = set()
    
    for keyword, project_name in PROJECT_KEYWORDS.items():
        if keyword in text_lower and project_name not in seen:
            seen.add(project_name)
            projects.append({"name": project_name})
    
    return projects


def extract_people(text: str) -> list[dict]:
    """Extract potential person names and contacts from text."""
    people = []
    
    # Email addresses
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
    for email in emails:
        if email.lower() not in IDENTITY_EMAILS:
            # Extract name from email
            name_part = email.split("@")[0].replace(".", " ").replace("_", " ").title()
            people.append({"name": name_part, "email": email})
    
    # Contact patterns: "Contact: Name"
    contacts = re.findall(r'Contact:\s*([A-Z][a-z]+(?: [A-Z][a-z]+)+)', text)
    for name in contacts:
        people.append({"name": name.strip()})
    
    return people


def extract_search_queries(text: str) -> list[dict]:
    """Extract search queries from text."""
    queries = []
    
    # "Searched for X" pattern
    searched = re.findall(r'Searched for\s+(.+?)(?:\n|$)', text, re.IGNORECASE)
    for q in searched:
        clean = q.strip().rstrip(".")
        if len(clean) > 3 and len(clean) < 200:
            queries.append({"query": clean})
    
    # Standalone search-like content
    if "[Source: google_search]" in text:
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("[") and len(line) > 5 and len(line) < 200:
                queries.append({"query": line})
    
    return queries


def extract_locations(text: str) -> list[dict]:
    """Extract location references from text."""
    locations = []
    
    # "Visited place: X" pattern
    places = re.findall(r'Visited place:\s*(.+?)(?:\n|$)', text)
    for place in places:
        if place.strip():
            locations.append({"name": place.strip()})
    
    # "Address: X" pattern
    addresses = re.findall(r'Address:\s*(.+?)(?:\n|$)', text)
    for addr in addresses:
        if addr.strip():
            locations.append({"address": addr.strip()})
    
    # Known locations
    known_locations = ["vacaville", "sacramento", "san francisco", "davis", "california"]
    text_lower = text.lower()
    for loc in known_locations:
        if loc in text_lower:
            locations.append({"name": loc.title()})
    
    return locations


# ── Cypher Generation ────────────────────────────────────────────────

def generate_cypher(chunk_text: str, chunk_id: str) -> str:
    """Generate SELF-CONTAINED Cypher MERGE statements from a text chunk.
    
    Each statement is fully self-contained — no variable references across
    statements. This allows each one to execute independently.
    """
    statements = []
    
    # ── Always create the identity node ──
    statements.append(
        f"MERGE (user:Person {{name: '{IDENTITY_NODE}'}}) "
        f"SET user.email = 'andrewdarcy530@gmail.com';"
    )
    
    # ── Extract and process URLs ──
    urls = extract_urls(chunk_text)
    for url_info in urls:
        safe_url = escape_cypher(url_info["url"])
        safe_domain = escape_cypher(url_info["domain"])
        safe_org = escape_cypher(url_info["org_name"])
        safe_page = escape_cypher(url_info["page_title"][:100])
        
        # Create website/org node
        statements.append(
            f"MERGE (:{url_info['org_label']} {{name: '{safe_org}'}});"
        )
        
        # Create artifact + relationships as self-contained statement
        if safe_page and safe_page != safe_domain:
            safe_artifact_name = escape_cypher(f"{url_info['org_name']}: {url_info['page_title'][:80]}")
            statements.append(
                f"MERGE (page:DigitalArtifact {{url: '{safe_url}'}}) "
                f"SET page.name = '{safe_artifact_name}', "
                f"page.domain = '{safe_domain}';"
            )
            # Self-contained relationship: VISITED
            statements.append(
                f"MATCH (user:Person {{name: '{IDENTITY_NODE}'}}) "
                f"MERGE (page:DigitalArtifact {{url: '{safe_url}'}}) "
                f"MERGE (user)-[:VISITED]->(page);"
            )
            # Self-contained relationship: EXTRACTED_FROM
            statements.append(
                f"MATCH (page:DigitalArtifact {{url: '{safe_url}'}}) "
                f"MERGE (site:{url_info['org_label']} {{name: '{safe_org}'}}) "
                f"MERGE (page)-[:EXTRACTED_FROM]->(site);"
            )
    
    # ── Extract tech entities ──
    tech_entities = extract_tech_entities(chunk_text)
    for entity in tech_entities:
        safe_name = escape_cypher(entity["name"])
        category = entity["category"]
        rel = "USES" if category in ("Hardware", "Software") else "RESEARCHES"
        statements.append(
            f"MATCH (user:Person {{name: '{IDENTITY_NODE}'}}) "
            f"MERGE (e:{category} {{name: '{safe_name}'}}) "
            f"MERGE (user)-[:{rel}]->(e);"
        )
    
    # ── Extract projects ──
    projects = extract_projects(chunk_text)
    for proj in projects:
        safe_name = escape_cypher(proj["name"])
        statements.append(
            f"MATCH (user:Person {{name: '{IDENTITY_NODE}'}}) "
            f"MERGE (p:Project {{name: '{safe_name}'}}) "
            f"MERGE (user)-[:CONTRIBUTES_TO]->(p);"
        )
    
    # ── Extract people ──
    people = extract_people(chunk_text)
    for person in people:
        safe_name = escape_cypher(person["name"])
        if len(safe_name) < 2:
            continue
        email_set = ""
        if person.get("email"):
            safe_email = escape_cypher(person["email"])
            email_set = f" SET p.email = '{safe_email}'"
        statements.append(
            f"MATCH (user:Person {{name: '{IDENTITY_NODE}'}}) "
            f"MERGE (p:Person {{name: '{safe_name}'}}){email_set} "
            f"MERGE (user)-[:KNOWS]->(p);"
        )
    
    # ── Extract search queries ──
    queries = extract_search_queries(chunk_text)
    for q in queries:
        safe_query = escape_cypher(q["query"][:150])
        if len(safe_query) < 3:
            continue
        statements.append(
            f"MATCH (user:Person {{name: '{IDENTITY_NODE}'}}) "
            f"MERGE (sq:SearchQuery {{text: '{safe_query}'}}) "
            f"MERGE (user)-[:SEARCHED_FOR]->(sq);"
        )
    
    # ── Extract locations ──
    locations = extract_locations(chunk_text)
    seen_locations = set()
    for loc in locations:
        loc_name = loc.get("name", loc.get("address", ""))
        if not loc_name or loc_name in seen_locations:
            continue
        seen_locations.add(loc_name)
        safe_loc = escape_cypher(loc_name)
        addr_set = ""
        if loc.get("address"):
            safe_addr = escape_cypher(loc["address"])
            addr_set = f" SET loc.address = '{safe_addr}'"
        statements.append(
            f"MATCH (user:Person {{name: '{IDENTITY_NODE}'}}) "
            f"MERGE (loc:Location {{name: '{safe_loc}'}}){addr_set} "
            f"MERGE (user)-[:VISITED]->(loc);"
        )
    
    # ── Conversation-specific extraction (Gemini) ──
    if "gemini_activity" in chunk_text.lower() or "conversation" in chunk_text.lower():
        prompts = re.findall(r'User:\s*(.+?)(?:\n|$)', chunk_text)
        for prompt in prompts:
            clean = prompt.strip()[:150]
            if len(clean) > 10:
                for keyword, category in TECH_KEYWORDS.items():
                    if keyword in clean.lower():
                        safe_name = escape_cypher(keyword.title() if len(keyword) > 3 else keyword.upper())
                        statements.append(
                            f"MATCH (user:Person {{name: '{IDENTITY_NODE}'}}) "
                            f"MERGE (topic:{category} {{name: '{safe_name}'}}) "
                            f"MERGE (user)-[:INTERESTED_IN]->(topic);"
                        )
    
    # Add chunk provenance comment
    statements.append(f"// Extracted from chunk: {chunk_id}")
    
    return "\n".join(statements)


# ── Neo4j Execution ─────────────────────────────────────────────────

BATCH_TX_SIZE = 50  # Statements per transaction
MAX_RETRIES = 3


def execute_cypher_batched(cypher_text: str, driver) -> tuple[int, int]:
    """Execute self-contained Cypher statements against Neo4j using batched
    transactions with deadlock retry.
    """
    import random
    raw_stmts = [s.strip() for s in cypher_text.split(";")
                 if s.strip() and not s.strip().startswith("//")]
    
    success = 0
    errors = 0
    
    for batch_start in range(0, len(raw_stmts), BATCH_TX_SIZE):
        batch = raw_stmts[batch_start:batch_start + BATCH_TX_SIZE]
        
        for attempt in range(MAX_RETRIES):
            try:
                with driver.session() as session:
                    with session.begin_transaction() as tx:
                        for stmt in batch:
                            tx.run(stmt)
                        tx.commit()
                success += len(batch)
                break  # Success — exit retry loop
            except Exception as e:
                err_str = str(e)
                if "Forseti" in err_str or "DeadlockDetected" in err_str or "deadlock" in err_str.lower():
                    # Deadlock — retry with jittered backoff
                    wait = (0.1 * (2 ** attempt)) + random.uniform(0, 0.1)
                    time.sleep(wait)
                    continue
                else:
                    # Non-deadlock error — fall back to individual
                    for stmt in batch:
                        try:
                            with driver.session() as s2:
                                s2.run(stmt).consume()
                            success += 1
                        except Exception:
                            errors += 1
                    break
        else:
            # All retries exhausted — individual fallback
            for stmt in batch:
                try:
                    with driver.session() as s2:
                        s2.run(stmt).consume()
                    success += 1
                except Exception:
                    errors += 1
    
    return success, errors


def process_single_chunk(
    chunk_path: Path,
    driver,
    dry_run: bool,
) -> tuple[str, int, int, int]:
    """Process a single chunk: read → extract → generate Cypher → execute.
    
    Returns (chunk_id, merge_count, success, errors).
    Thread-safe — each call uses its own Neo4j session.
    """
    chunk_id = chunk_path.stem
    
    try:
        chunk_text = chunk_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return chunk_id, 0, 0, 1
    
    # Generate Cypher
    cypher = generate_cypher(chunk_text, chunk_id)
    merge_count = cypher.upper().count("MERGE")
    
    # Save Cypher to file
    cypher_path = CYPHER_DIR / f"{chunk_id}.cypher"
    cypher_path.write_text(cypher, encoding='utf-8')
    
    # Execute against Neo4j
    success, errors = 0, 0
    if driver and not dry_run:
        success, errors = execute_cypher_batched(cypher, driver)
    
    return chunk_id, merge_count, success, errors


# ── Main Pipeline ───────────────────────────────────────────────────

def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    parser = argparse.ArgumentParser(description="Knowledge Graph Builder")
    parser.add_argument("--test", action="store_true", help="Process only 1 chunk")
    parser.add_argument("--dry-run", action="store_true", help="Generate Cypher only, no Neo4j")
    parser.add_argument("--neo4j-uri", default=NEO4J_URI, help="Neo4j bolt URI")
    parser.add_argument("--neo4j-user", default=NEO4J_USER)
    parser.add_argument("--neo4j-pass", default=NEO4J_PASS)
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CASCADES Knowledge Graph Builder (PARALLEL)")
    print("=" * 60)
    print(f"  Chunks dir: {CHUNKS_DIR}")
    print(f"  Cypher dir: {CYPHER_DIR}")
    print(f"  Neo4j:      {args.neo4j_uri}")
    print(f"  Workers:    {args.workers}")
    print(f"  Dry run:    {args.dry_run}")
    
    # ── Find chunks ──
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.txt"))
    if not chunk_files:
        print(f"\nERROR: No chunk files found in {CHUNKS_DIR}")
        print("Run extract_takeout.py first.")
        sys.exit(1)
    
    # Filter already-processed chunks
    CYPHER_DIR.mkdir(parents=True, exist_ok=True)
    existing = {f.stem for f in CYPHER_DIR.glob("*.cypher")}
    remaining = [f for f in chunk_files if f.stem not in existing]
    
    print(f"\n  Total chunks:     {len(chunk_files)}")
    print(f"  Already processed: {len(existing)}")
    print(f"  Remaining:        {len(remaining)}")
    
    if not remaining:
        print("\nAll chunks already processed!")
        return
    
    if args.test:
        remaining = remaining[:1]
        print("  TEST MODE: Processing 1 chunk only")
    
    # ── Connect to Neo4j (unless dry run) ──
    driver = None
    if not args.dry_run:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                args.neo4j_uri,
                auth=(args.neo4j_user, args.neo4j_pass),
                max_connection_pool_size=args.workers + 4,
            )
            with driver.session() as session:
                session.run("RETURN 1")
            print(f"\n  ✓ Connected to Neo4j at {args.neo4j_uri}")
        except ImportError:
            print("\n  WARNING: neo4j Python driver not installed.")
            print("  Falling back to dry-run mode.")
            args.dry_run = True
        except Exception as e:
            print(f"\n  WARNING: Cannot connect to Neo4j: {e}")
            print("  Falling back to dry-run mode.")
            args.dry_run = True
    
    # ── Process chunks in parallel ──
    print(f"\n{'=' * 60}")
    print(f"PROCESSING {len(remaining)} CHUNKS ({args.workers} workers)")
    print(f"{'=' * 60}")
    
    total_merges = 0
    total_success = 0
    total_errors = 0
    completed = 0
    start_time = time.time()
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_chunk, cp, driver, args.dry_run): cp
            for cp in remaining
        }
        
        for future in as_completed(futures):
            chunk_id, merges, success, errors = future.result()
            
            with lock:
                completed += 1
                total_merges += merges
                total_success += success
                total_errors += errors
                
                # Progress every 50 chunks
                if completed % 50 == 0 or completed == len(remaining):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(remaining) - completed) / rate if rate > 0 else 0
                    print(
                        f"  [{completed}/{len(remaining)}] "
                        f"{rate:.1f} chunks/s | "
                        f"Neo4j: {total_success:,} OK, {total_errors} err | "
                        f"ETA: {eta/60:.1f}m"
                    )
    
    # ── Summary ──
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Chunks processed: {completed}")
    print(f"  Total MERGE stmts: {total_merges:,}")
    if not args.dry_run:
        print(f"  Neo4j executed:   {total_success:,} OK, {total_errors} errors")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Rate: {completed/elapsed:.1f} chunks/s")
    print(f"  Cypher files: {CYPHER_DIR}")
    
    if driver:
        driver.close()


if __name__ == "__main__":
    main()
