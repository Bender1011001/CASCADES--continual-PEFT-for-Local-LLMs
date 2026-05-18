#!/usr/bin/env python3
"""
CASCADES Digital Twin Extraction Pipeline.
Processes Google Takeout chunk files into Cypher MERGE statements using
the local CASCADES Qwen3-4B server, while simultaneously feeding the
extracted facts into Hemisphere B for parametric memory consolidation.

Usage:
    python local_extract_cascades.py --test          # Process 1 chunk for validation
    python local_extract_cascades.py                 # Process all remaining chunks
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import requests

# --- Configuration ---
CHUNKS_DIR = r"E:\digital-twin\takeout_chunks"
CYPHER_DIR = r"D:\digital-twin\cypher_output"
CASCADES_URL = "http://127.0.0.1:8000"

# CASCADES server handles 32k context, so we don't need heavy sub-chunking
# We set this to 12000 (~3000 tokens) to leave plenty of room in 8192 context window
MAX_SUBCHUNK_CHARS = 12000
GENERATION_TIMEOUT = 1200  # 20 minutes per sub-chunk (quantized models can be slower)

EXTRACTION_PROMPT = """You are a knowledge graph extraction engine. Read the following data and extract ALL meaningful entities and relationships as Neo4j Cypher MERGE statements.

Identity Mapping:
- andrewdarcy530@gmail.com → Person "Bender1011001"
- andrewdarcy707@live.com → Person "Bender1011001"
- Any activity by the user → Person "Bender1011001"

Node Labels: Person, Project, Concept, Hardware, Software, Location, DigitalArtifact

Relationship Types: RESEARCHES, USES, PURCHASING, CONTRIBUTES_TO, TRACKING, INTERESTED_IN, CONFIGURES, DEBUGS, INSTALLS, EXTRACTED_FROM

Rules:
1. Use ONLY MERGE statements (never CREATE)
2. Extract specific, meaningful entities — not generic ones
3. Set properties: name, url, description, timestamp where available
4. Connect everything back to Bender1011001
5. If data is repetitive telemetry, write ONE representative MERGE block
6. Output ONLY valid Cypher — no explanations, no markdown fences

Data:
{chunk_content}"""


def check_cascades():
    """Verify CASCADES server is running and Hemisphere B is active."""
    print("Waiting for CASCADES server to boot (may take 60s for 8GB model)...")
    for attempt in range(15):
        try:
            resp = requests.get(f"{CASCADES_URL}/v1/memory/stats", timeout=5)
            resp.raise_for_status()
            stats = resp.json()
            print("    CASCADES stats:")
            for k, v in stats.items():
                print(f"      - {k}: {v}")
            return True
        except (requests.ConnectionError, requests.Timeout):
            print(f"  Attempt {attempt+1}/15: Server not ready, waiting 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"ERROR checking CASCADES: {e}")
            sys.exit(1)
            
    print("ERROR: Cannot connect to CASCADES server at", CASCADES_URL)
    print("Make sure it is running: python -m app.server --model_id ./abliteratedqwen3")
    sys.exit(1)


def get_remaining_chunks():
    """Find chunk files that don't have matching .cypher output."""
    chunk_files = glob.glob(os.path.join(CHUNKS_DIR, "chunk_*.txt"))
    
    existing_cypher = set()
    if os.path.isdir(CYPHER_DIR):
        for f in os.listdir(CYPHER_DIR):
            if f.endswith(".cypher"):
                stem = os.path.splitext(f)[0]
                existing_cypher.add(stem)
    
    remaining = []
    for cf in chunk_files:
        stem = os.path.splitext(os.path.basename(cf))[0]
        if stem not in existing_cypher:
            remaining.append(cf)
    
    def sort_key(path):
        name = os.path.basename(path)
        nums = re.findall(r'\d+', name)
        return int(nums[0]) if nums else 0
    
    remaining.sort(key=sort_key)
    return remaining


def split_into_subchunks(text, max_chars=MAX_SUBCHUNK_CHARS):
    """Split text into sub-chunks that fit within context window."""
    if len(text) <= max_chars:
        return [text]
    
    subchunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            subchunks.append(text[start:])
            break
        
        split_point = text.rfind('\n', start, end)
        if split_point <= start:
            split_point = end
        else:
            split_point += 1
        
        subchunks.append(text[start:split_point])
        start = split_point
    
    return subchunks


def clean_cypher_output(text):
    """Strip non-Cypher content from model output, keeping only valid statements."""
    text = re.sub(r'^```(?:cypher)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    
    cleaned_lines = []
    in_statement = False
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            if in_statement:
                cleaned_lines.append('')
            in_statement = False
            continue
        if stripped.upper().startswith('MERGE'):
            cleaned_lines.append(line)
            in_statement = True
            continue
        if stripped.startswith('//'):
            cleaned_lines.append(line)
            in_statement = False
            continue
        if stripped.upper().startswith(('SET ', 'ON CREATE', 'ON MATCH')):
            cleaned_lines.append(line)
            in_statement = True
            continue
        if stripped.upper().startswith(('WITH ', 'MATCH ')):
            cleaned_lines.append(line)
            in_statement = True
            continue
        in_statement = False
    
    return '\n'.join(cleaned_lines).strip()


def extract_cypher(text):
    """Send text to CASCADES server.
    This does TWO things automatically:
    1. Returns the generated Cypher
    2. The CASCADES server's Self-Synthesizer automatically reads the prompt/response, 
       extracts any 'Bender1011001' facts, and feeds Hemisphere B.
    """
    prompt = EXTRACTION_PROMPT.format(chunk_content=text)
    
    import uuid
    # Create a unique conversation_id for this chunk so the server runs the Self-Synthesizer
    chunk_conv_id = str(uuid.uuid4())
    
    payload = {
        "model": "cascades",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": 1536,  # 1536 tokens is plenty for cypher output
        "temperature": 0.1,
        "conversation_id": chunk_conv_id
    }
    
    try:
        start_time = time.time()
        resp = requests.post(
            f"{CASCADES_URL}/v1/chat/completions",
            json=payload,
            timeout=GENERATION_TIMEOUT
        )
        resp.raise_for_status()
        duration = time.time() - start_time
        
        result = resp.json()
        response_text = result["choices"][0]["message"]["content"]
        response_text = clean_cypher_output(response_text)
        
        return response_text, duration
    
    except requests.Timeout:
        print(f"    TIMEOUT after {GENERATION_TIMEOUT}s")
        return None, 0
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, 0


def process_chunk(chunk_path, chunk_num, total_chunks):
    """Process a single chunk file."""
    stem = os.path.splitext(os.path.basename(chunk_path))[0]
    cypher_path = os.path.join(CYPHER_DIR, f"{stem}.cypher")
    
    print(f"\n[{chunk_num}/{total_chunks}] Processing {os.path.basename(chunk_path)}...")
    
    try:
        with open(chunk_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"    ERROR reading file: {e}")
        return False, 0
    
    file_size = len(content)
    print(f"    File size: {file_size:,} chars ({file_size // 4:,} est. tokens)")
    
    subchunks = split_into_subchunks(content)
    num_subchunks = len(subchunks)
    
    if num_subchunks > 1:
        print(f"    Split into {num_subchunks} sub-chunks")
    
    all_cypher = []
    total_time_s = 0
    
    for i, subchunk in enumerate(subchunks):
        if num_subchunks > 1:
            print(f"    Sub-chunk {i + 1}/{num_subchunks} ({len(subchunk):,} chars)...", end=" ", flush=True)
        else:
            print(f"    Generating Cypher...", end=" ", flush=True)
        
        cypher_text, duration_s = extract_cypher(subchunk)
        
        if cypher_text is None:
            print("FAILED")
            continue
        
        total_time_s += duration_s
        
        merge_count = cypher_text.upper().count("MERGE")
        print(f"OK ({duration_s:.1f}s, {merge_count} MERGE statements)")
        
        if merge_count > 0:
            all_cypher.append(f"// --- Sub-chunk {i + 1}/{num_subchunks} from {stem} ---")
            all_cypher.append(cypher_text)
    
    if all_cypher:
        os.makedirs(CYPHER_DIR, exist_ok=True)
        combined = "\n\n".join(all_cypher)
        with open(cypher_path, 'w', encoding='utf-8') as f:
            f.write(combined)
        
        total_merges = combined.upper().count("MERGE")
        print(f"    [OK] Wrote {cypher_path} ({total_merges} MERGE statements, {total_time_s:.1f}s model time)")
        return True, time.time()
    else:
        print(f"    [SKIP] No valid Cypher extracted, skipping")
        return False, 0


def main():
    parser = argparse.ArgumentParser(description="CASCADES Digital Twin Extraction")
    parser.add_argument("--test", action="store_true", help="Process only 1 chunk for validation")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CASCADES Digital Twin Pipeline")
    print("Extracts Cypher & trains Hemisphere B simultaneously")
    print("=" * 60)
    
    print("Checking CASCADES Server...")
    check_cascades()
    
    remaining = get_remaining_chunks()
    print(f"Remaining chunks: {len(remaining)}")
    
    if not remaining:
        print("All chunks already processed!")
        return
    
    if args.test:
        remaining = remaining[:1]
        print(f"TEST MODE: Processing only 1 chunk")
    
    print()
    
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for i, chunk_path in enumerate(remaining, 1):
        success, _ = process_chunk(chunk_path, i, len(remaining))
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            
        # Print CASCADES memory stats after each chunk to show learning progress
        try:
            resp = requests.get(f"{CASCADES_URL}/v1/memory/stats", timeout=5)
            if resp.ok:
                stats = resp.json()
                facts = stats.get('known_facts', 0)
                dream_state = stats.get('dream', 'unknown')
                print(f"    [MEMORY] Facts Known: {facts} | System: {dream_state}")
        except:
            pass
            
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed:  {success_count + fail_count} chunks")
    print(f"Succeeded:  {success_count}")
    print(f"Failed:     {fail_count}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 3600:.2f} hours)")

if __name__ == "__main__":
    main()
