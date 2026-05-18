#!/usr/bin/env python3
"""
Converts Google Takeout chunks into CASCADES training data (JSONL ChatML format).

This script reads the raw text chunks extracted from Google Takeout and converts
them into the format required by CASCADES' data.py for continual learning on the
abliterated Qwen3 4B model.

Training Format (ChatML JSONL):
  {"prompt": "...", "response": "<think>...</think>...", "category": "...", "source": "..."}

The training data covers:
  - Search interests → Q&A about what you research
  - Browsing patterns → Q&A about sites you use
  - Conversations → Q&A about topics you discuss
  - Projects → Q&A about what you build
  - Tools → Q&A about what software/hardware you use
  - Geography → Q&A about where you've been
  - Relationships → Q&A about who you interact with
  - Preferences → Q&A about music/video/content you consume

Usage:
    python build_training_data.py                     # Process all chunks
    python build_training_data.py --max-chunks 100    # Process first 100 chunks
    python build_training_data.py --output custom.jsonl
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

# ── Configuration ────────────────────────────────────────────────────
CHUNKS_DIR = Path(r"E:\digital-twin\takeout_chunks")
OUTPUT_DIR = Path(r"E:\digital-twin\training_data")
DEFAULT_OUTPUT = OUTPUT_DIR / "digital_twin_cascades.jsonl"

IDENTITY_NAME = "Bender1011001"
REAL_NAME = "Andrew Darcy"


# ── URL to Interest Mapper ──────────────────────────────────────────

DOMAIN_INTERESTS = {
    "github.com": ("coding", "software development"),
    "stackoverflow.com": ("coding", "technical problem-solving"),
    "reddit.com": ("social media", "community discussion"),
    "youtube.com": ("video content", "entertainment/education"),
    "arxiv.org": ("research papers", "AI/ML research"),
    "huggingface.co": ("AI models", "machine learning"),
    "polymarket.com": ("prediction markets", "event trading"),
    "kalshi.com": ("prediction markets", "event derivatives"),
    "immunefi.com": ("bug bounties", "Web3 security"),
    "claude.ai": ("AI assistants", "coding"),
    "gemini.google.com": ("AI assistants", "research"),
    "grok.com": ("AI assistants", "research"),
    "openrouter.ai": ("AI APIs", "model routing"),
    "lmstudio.ai": ("local AI", "model inference"),
    "notebooklm.google.com": ("research tools", "note-taking"),
    "neo4j.com": ("graph databases", "knowledge graphs"),
    "stripe.com": ("payments", "e-commerce"),
    "remotedesktop.google.com": ("remote access", "system administration"),
    "chromewebstore.google.com": ("browser extensions", "tooling"),
}


def extract_browsing_qa(chunk_text: str) -> list[dict]:
    """Extract Q&A pairs from Chrome browsing history chunks."""
    pairs = []
    
    # Extract page titles and URLs
    titles = re.findall(r'"title":\s*"([^"]+)"', chunk_text)
    urls = re.findall(r'"url":\s*"([^"]+)"', chunk_text)
    
    # Group by domain
    domain_pages = defaultdict(list)
    for i, url in enumerate(urls):
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().lstrip("www.")
        except Exception:
            continue
        
        title = titles[i] if i < len(titles) else ""
        if title and title not in ("", "YouTube", "Google", "New Tab"):
            domain_pages[domain].append(title)
    
    # Generate Q&A pairs for frequently visited domains
    for domain, pages in domain_pages.items():
        unique_pages = list(set(pages))[:20]
        if not unique_pages:
            continue
        
        interest_info = DOMAIN_INTERESTS.get(domain)
        
        if interest_info:
            category, description = interest_info
            # Generate interest-based Q&A
            pairs.append({
                "prompt": f"Does {IDENTITY_NAME} use {domain}?",
                "response": f"<think>Checking browsing history for {domain} activity.</think>"
                            f"Yes, {IDENTITY_NAME} regularly uses {domain} for {description}. "
                            f"Some recent activities include: {', '.join(unique_pages[:5])}.",
                "category": "browsing_habit",
                "source": f"chrome_history_{domain}",
            })
        
        # YouTube specific — extract video interests
        if "youtube.com" in domain:
            video_titles = [t for t in unique_pages if "YouTube" not in t]
            if video_titles:
                pairs.append({
                    "prompt": f"What kind of videos does {IDENTITY_NAME} watch?",
                    "response": f"<think>Analyzing YouTube watch history.</think>"
                                f"{IDENTITY_NAME} watches a variety of content. "
                                f"Recent videos include: {', '.join(video_titles[:10])}.",
                    "category": "media_consumption",
                    "source": "youtube_history",
                })
        
        # Grok conversations
        if "grok.com" in domain:
            conversation_titles = [t for t in unique_pages if "New conversation" not in t and t != "Grok"]
            if conversation_titles:
                pairs.append({
                    "prompt": f"What does {IDENTITY_NAME} discuss with Grok?",
                    "response": f"<think>Reviewing Grok conversation history.</think>"
                                f"{IDENTITY_NAME} has discussed various topics with Grok, including: "
                                f"{', '.join(conversation_titles[:8])}.",
                    "category": "ai_conversations",
                    "source": "grok_history",
                })
        
        # Gemini conversations
        if "gemini.google.com" in domain:
            pairs.append({
                "prompt": f"Does {IDENTITY_NAME} use Google Gemini?",
                "response": f"<think>Checking Gemini usage patterns.</think>"
                            f"Yes, {IDENTITY_NAME} frequently uses Google Gemini for research and coding assistance.",
                "category": "tool_usage",
                "source": "gemini_history",
            })
    
    return pairs


def extract_search_qa(chunk_text: str) -> list[dict]:
    """Extract Q&A pairs from Google Search activity."""
    pairs = []
    
    # Pattern: "Searched for X"
    searches = re.findall(r'Searched for\s+(.+?)(?:\n|$)', chunk_text, re.IGNORECASE)
    
    if searches:
        unique_searches = list(set([s.strip() for s in searches if len(s.strip()) > 3]))[:20]
        if unique_searches:
            pairs.append({
                "prompt": f"What does {IDENTITY_NAME} search for on Google?",
                "response": f"<think>Reviewing search history patterns.</think>"
                            f"{IDENTITY_NAME} has searched for various topics including: "
                            f"{', '.join(unique_searches[:10])}.",
                "category": "search_interests",
                "source": "google_search",
            })
            
            # Individual search Q&As for specific topics
            for search in unique_searches[:5]:
                pairs.append({
                    "prompt": f"Has {IDENTITY_NAME} researched '{search}'?",
                    "response": f"<think>Checking search records for '{search}'.</think>"
                                f"Yes, {IDENTITY_NAME} has searched for '{search}', "
                                f"indicating interest in this topic.",
                    "category": "search_specific",
                    "source": "google_search",
                })
    
    return pairs


def extract_gemini_qa(chunk_text: str) -> list[dict]:
    """Extract Q&A from Gemini/AI conversation chunks."""
    pairs = []
    
    # Look for conversation patterns
    prompts = re.findall(r'User:\s*(.+?)(?:\n|$)', chunk_text)
    responses = re.findall(r'(?:AI|Assistant):\s*(.+?)(?:\n|$)', chunk_text)
    
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()
        if len(prompt) < 10:
            continue
        response = responses[i].strip() if i < len(responses) else ""
        
        if response:
            pairs.append({
                "prompt": f"What did {IDENTITY_NAME} ask about: {prompt[:100]}?",
                "response": f"<think>Recalling conversation about this topic.</think>"
                            f"{IDENTITY_NAME} asked: \"{prompt[:200]}\" and received: \"{response[:300]}\".",
                "category": "conversation",
                "source": "gemini_activity",
            })
    
    return pairs


def extract_project_qa(chunk_text: str) -> list[dict]:
    """Extract project-related Q&A from any chunk type."""
    pairs = []
    text_lower = chunk_text.lower()
    
    projects = {
        "cascades": "CASCADES - a continual PEFT framework for local LLMs using Riemannian optimization on the Stiefel manifold",
        "m-a-k-e-r": "M-A-K-E-R - a multi-persona smart contract auditing framework for Web3 security",
        "astroforge": "AstroForge - an astrological computation and reading platform",
        "digital twin": "Digital Twin - a knowledge graph and model of personal data for AI-native identity",
        "obliteratus": "OBLITERATUS - a project related to model abliteration",
        "tail risk": "Tail Risk Engine - an automated quant bot targeting mispriced event derivative contracts",
    }
    
    for keyword, description in projects.items():
        if keyword in text_lower:
            project_name = description.split(" - ")[0]
            pairs.append({
                "prompt": f"What is {project_name}?",
                "response": f"<think>Recalling project details about {project_name}.</think>"
                            f"{project_name} is one of {IDENTITY_NAME}'s projects. "
                            f"It's {description}.",
                "category": "project",
                "source": "project_reference",
            })
    
    return pairs


def extract_location_qa(chunk_text: str) -> list[dict]:
    """Extract location-related Q&A."""
    pairs = []
    
    # Visited place patterns
    places = re.findall(r'Visited place:\s*(.+?)(?:\n|$)', chunk_text)
    if places:
        unique_places = list(set([p.strip() for p in places if len(p.strip()) > 2]))[:10]
        if unique_places:
            pairs.append({
                "prompt": f"Where has {IDENTITY_NAME} been recently?",
                "response": f"<think>Checking location history.</think>"
                            f"{IDENTITY_NAME} has visited: {', '.join(unique_places)}.",
                "category": "location",
                "source": "location_history",
            })
    
    return pairs


def extract_contact_qa(chunk_text: str) -> list[dict]:
    """Extract contact/relationship Q&A."""
    pairs = []
    
    contacts = re.findall(r'Contact:\s*(.+?)(?:\n|$)', chunk_text)
    if contacts:
        unique_contacts = list(set([c.strip() for c in contacts if len(c.strip()) > 2]))[:10]
        if unique_contacts:
            pairs.append({
                "prompt": f"Who are some of {IDENTITY_NAME}'s contacts?",
                "response": f"<think>Reviewing contact information.</think>"
                            f"Some of {IDENTITY_NAME}'s contacts include: {', '.join(unique_contacts[:10])}.",
                "category": "relationships",
                "source": "contacts",
            })
    
    return pairs


def extract_identity_qa() -> list[dict]:
    """Generate core identity Q&A pairs."""
    return [
        {
            "prompt": f"Who is {IDENTITY_NAME}?",
            "response": f"<think>Recalling core identity information.</think>"
                        f"{IDENTITY_NAME} (real name {REAL_NAME}) is a software developer, "
                        f"AI researcher, and entrepreneur working on projects like CASCADES "
                        f"(continual PEFT for local LLMs), M-A-K-E-R (smart contract auditing), "
                        f"AstroForge (astrological computation), and the Tail Risk Engine "
                        f"(prediction market trading). Uses email andrewdarcy530@gmail.com.",
            "category": "identity",
            "source": "core_identity",
        },
        {
            "prompt": f"What does {IDENTITY_NAME} work on?",
            "response": f"<think>Reviewing project portfolio.</think>"
                        f"{IDENTITY_NAME} works on several technical projects: "
                        f"CASCADES (continual learning for local LLMs with Riemannian optimization), "
                        f"M-A-K-E-R (autonomous smart contract security auditing), "
                        f"AstroForge (astrology platform), Tail Risk Engine (event derivative trading), "
                        f"and Digital Twin (personal knowledge graph). "
                        f"Primary GPU is an RTX 4060 Ti 8GB.",
            "category": "identity",
            "source": "core_identity",
        },
        {
            "prompt": f"What tech stack does {IDENTITY_NAME} use?",
            "response": f"<think>Checking tool preferences.</think>"
                        f"{IDENTITY_NAME} uses: Python (primary language), PyTorch (ML framework), "
                        f"Transformers/HuggingFace (model loading), FastAPI (web servers), "
                        f"Neo4j (knowledge graph), React/Next.js (frontends), "
                        f"Docker, Git, Kali Linux (security testing). "
                        f"For AI: Qwen3-4B (abliterated, local), Google Gemini, Claude, Grok. "
                        f"Hardware: RTX 4060 Ti 8GB GPU, Windows.",
            "category": "identity",
            "source": "core_identity",
        },
    ]


# ── Main Pipeline ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build CASCADES training data from Takeout chunks")
    parser.add_argument("--max-chunks", type=int, default=0, help="Max chunks to process (0=all)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CASCADES Training Data Builder")
    print("=" * 60)
    print(f"  Chunks dir:  {CHUNKS_DIR}")
    print(f"  Output:      {output_path}")
    
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.txt"))
    if args.max_chunks > 0:
        chunk_files = chunk_files[:args.max_chunks]
    
    print(f"  Chunks:      {len(chunk_files)}")
    
    all_pairs = []
    
    # Core identity pairs (always included)
    identity_pairs = extract_identity_qa()
    all_pairs.extend(identity_pairs)
    print(f"\n  Identity pairs: {len(identity_pairs)}")
    
    # Process each chunk
    category_counts = defaultdict(int)
    
    for i, chunk_path in enumerate(chunk_files, 1):
        if i % 200 == 0:
            print(f"  Processing chunk {i}/{len(chunk_files)}...")
        
        try:
            chunk_text = chunk_path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        
        # Extract various types of Q&A
        pairs = []
        pairs.extend(extract_browsing_qa(chunk_text))
        pairs.extend(extract_search_qa(chunk_text))
        pairs.extend(extract_gemini_qa(chunk_text))
        pairs.extend(extract_project_qa(chunk_text))
        pairs.extend(extract_location_qa(chunk_text))
        pairs.extend(extract_contact_qa(chunk_text))
        
        for p in pairs:
            category_counts[p["category"]] += 1
        
        all_pairs.extend(pairs)
    
    # Deduplicate by prompt
    seen_prompts = set()
    unique_pairs = []
    for pair in all_pairs:
        key = pair["prompt"][:100]
        if key not in seen_prompts:
            seen_prompts.add(key)
            unique_pairs.append(pair)
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING DATA COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total pairs extracted: {len(all_pairs)}")
    print(f"  Unique pairs (deduped): {len(unique_pairs)}")
    print(f"  Written to: {output_path}")
    print(f"\nBreakdown by category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    
    # Calculate approximate token count
    total_chars = sum(len(json.dumps(p)) for p in unique_pairs)
    print(f"\n  Total chars: {total_chars:,}")
    print(f"  ~Tokens:     {total_chars // 4:,}")
    print(f"\nReady for CASCADES training with: python train.py --data {output_path}")


if __name__ == "__main__":
    main()
