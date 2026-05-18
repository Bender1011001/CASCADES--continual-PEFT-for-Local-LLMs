#!/usr/bin/env python3
"""
Google Takeout Extractor & Chunker for CASCADES Digital Twin Pipeline.

Phase 1: Extracts Google Takeout ZIPs and parses all relevant data sources
into structured text chunks suitable for entity extraction and LLM training.

Data Sources Parsed:
  - Gemini/Google AI Activity (MyActivity.html) → Q&A conversation pairs
  - Google Search History (MyActivity.html) → search queries with timestamps
  - YouTube History (MyActivity.html) → watched/searched videos
  - Chrome Browser History (BrowserHistory.json) → visited pages
  - Google Maps / Location History → places visited
  - Gmail metadata (if present) → contact patterns
  - Google Contacts (if present) → relationship data

Output: Text chunks (~32k chars each) written to CHUNKS_DIR

Usage:
    python extract_takeout.py                    # Full pipeline
    python extract_takeout.py --skip-extract     # Skip ZIP extraction, just chunk
    python extract_takeout.py --only-extract     # Only extract ZIPs, don't chunk
"""

import argparse
import glob
import json
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from html.parser import HTMLParser

# ── Configuration ────────────────────────────────────────────────────
DATA_DIR = Path(r"e:\code.projects\CASCADES--continual-PEFT-for-Local-LLMs\data")
RAW_DIR = Path(r"E:\digital-twin\takeout_raw")
CHUNKS_DIR = Path(r"E:\digital-twin\takeout_chunks")
MAX_CHUNK_CHARS = 12000  # ~3000 tokens — leaves room for prompt + response within 8192 ctx


# ── HTML Text Extractor ─────────────────────────────────────────────

class HTMLTextExtractor(HTMLParser):
    """Simple HTML to text converter that preserves structure."""

    def __init__(self):
        super().__init__()
        self._text_parts = []
        self._skip = False
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "li", "tr", "td", "th"):
            self._text_parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._text_parts.append(data)

    def get_text(self):
        text = "".join(self._text_parts)
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def html_to_text(html_content: str) -> str:
    """Convert HTML to readable text."""
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    return parser.get_text()


# ── Phase 1: ZIP Extraction ─────────────────────────────────────────

def extract_zips():
    """Extract all Google Takeout ZIP files to RAW_DIR."""
    zip_files = list(DATA_DIR.glob("*.zip")) + list(DATA_DIR.glob("Takeout*.zip"))
    if not zip_files:
        print("ERROR: No ZIP files found in", DATA_DIR)
        sys.exit(1)

    print(f"Found {len(zip_files)} ZIP files:")
    for zf in zip_files:
        size_gb = zf.stat().st_size / (1024**3)
        print(f"  {zf.name} ({size_gb:.2f} GB)")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for zf_path in zip_files:
        print(f"\nExtracting {zf_path.name}...")
        try:
            with zipfile.ZipFile(zf_path, 'r') as zf:
                member_count = len(zf.namelist())
                print(f"  {member_count} files in archive")
                zf.extractall(RAW_DIR)
                print(f"  ✓ Extracted to {RAW_DIR}")
        except zipfile.BadZipFile:
            print(f"  ✗ BAD ZIP FILE — skipping {zf_path.name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Show what was extracted
    top_dirs = set()
    for item in RAW_DIR.iterdir():
        top_dirs.add(item.name)
    print(f"\nExtracted top-level directories: {sorted(top_dirs)}")


# ── Phase 2: Data Parsers ───────────────────────────────────────────

def parse_gemini_activity(html_path: Path) -> list[dict]:
    """Parse Gemini/Google AI Activity HTML into conversation chunks.
    
    Extracts Q&A pairs: user prompts and AI responses with timestamps.
    """
    print(f"  Parsing Gemini Activity: {html_path.name}")
    try:
        content = html_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"    Error reading: {e}")
        return []

    chunks = []

    # Pattern 1: outer-cell structure (Google Takeout format)
    # Each activity card has class="outer-cell" or similar
    cells = re.findall(
        r'<div class="outer-cell[^"]*">(.*?)</div>\s*</div>\s*</div>',
        content, re.DOTALL
    )

    if not cells:
        # Pattern 2: content-cell structure
        cells = re.findall(
            r'<div class="content-cell[^"]*">(.*?)</div>',
            content, re.DOTALL
        )

    if not cells:
        # Fallback: just convert the whole thing to text and chunk it
        text = html_to_text(content)
        if text:
            chunks.append({
                "source": "gemini_activity",
                "category": "conversation",
                "content": text,
                "file": str(html_path.name),
            })
        return chunks

    # Process each activity cell
    current_prompt = None
    current_timestamp = None

    for cell in cells:
        cell_text = html_to_text(cell)
        if not cell_text or len(cell_text.strip()) < 5:
            continue

        # Extract timestamp if present
        ts_match = re.search(r'(\w+ \d+, \d{4},? \d+:\d+:\d+ [APap][Mm])', cell_text)
        if ts_match:
            current_timestamp = ts_match.group(1)

        # Clean boilerplate
        clean_text = cell_text
        for boilerplate in ["Prompted Gemini Apps", "Said to Gemini Apps",
                           "Gemini Apps", "Google AI", "Prompted", "Said"]:
            clean_text = clean_text.replace(boilerplate, "").strip()

        if not clean_text or len(clean_text) < 3:
            continue

        # Alternate between prompts and responses
        if current_prompt is None:
            current_prompt = clean_text
        else:
            chunks.append({
                "source": "gemini_activity",
                "category": "conversation",
                "timestamp": current_timestamp or "",
                "prompt": current_prompt,
                "response": clean_text,
            })
            current_prompt = None
            current_timestamp = None

    # If we didn't find paired conversations, just chunk the whole text
    if not chunks:
        text = html_to_text(content)
        if text and len(text) > 50:
            chunks.append({
                "source": "gemini_activity",
                "category": "conversation",
                "content": text,
                "file": str(html_path.name),
            })

    print(f"    Found {len(chunks)} conversation entries")
    return chunks


def parse_search_activity(html_path: Path, source: str = "google_search") -> list[dict]:
    """Parse Google Search / YouTube activity HTML."""
    print(f"  Parsing {source}: {html_path.name}")
    try:
        content = html_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"    Error reading: {e}")
        return []

    chunks = []
    text = html_to_text(content)
    
    # Split by activity entries (typically separated by timestamps)
    entries = re.split(r'\n(?=\w+ \d+, \d{4})', text)
    
    for entry in entries:
        entry = entry.strip()
        if not entry or len(entry) < 10:
            continue
        
        # Extract timestamp
        ts_match = re.search(r'(\w+ \d+, \d{4},? \d+:\d+:\d+ [APap][Mm])', entry)
        timestamp = ts_match.group(1) if ts_match else ""
        
        # Clean common prefixes
        clean = entry
        for prefix in ["Searched for", "Visited", "Watched", "Used"]:
            clean = re.sub(rf'^{prefix}\s+', '', clean, flags=re.IGNORECASE)
        
        clean = clean.strip()
        if clean and len(clean) > 5:
            chunks.append({
                "source": source,
                "category": "activity",
                "timestamp": timestamp,
                "content": clean,
            })

    print(f"    Found {len(chunks)} entries")
    return chunks


def parse_chrome_history(json_path: Path) -> list[dict]:
    """Parse Chrome BrowserHistory.json."""
    print(f"  Parsing Chrome History: {json_path.name}")
    try:
        with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except Exception as e:
        print(f"    Error: {e}")
        return []

    chunks = []
    browser_history = data if isinstance(data, list) else data.get("Browser History", [])
    
    for entry in browser_history:
        title = entry.get("title", "").strip()
        url = entry.get("url", "")
        time_usec = entry.get("time_usec", 0)
        
        if not title or title.lower() in {"new tab", "untitled", ""}:
            continue
        
        # Convert microseconds to readable timestamp
        timestamp = ""
        if time_usec:
            try:
                # Windows epoch: microseconds since 1601-01-01
                # But Google Takeout uses Unix epoch microseconds
                ts = datetime.fromtimestamp(time_usec / 1_000_000)
                timestamp = ts.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OSError, OverflowError):
                pass

        chunks.append({
            "source": "chrome_history",
            "category": "browsing",
            "timestamp": timestamp,
            "content": f"Visited: {title}\nURL: {url}",
        })

    print(f"    Found {len(chunks)} browsing entries")
    return chunks


def parse_location_history(json_path: Path) -> list[dict]:
    """Parse Google Maps / Location History."""
    print(f"  Parsing Location History: {json_path.name}")
    try:
        with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except Exception as e:
        print(f"    Error: {e}")
        return []

    chunks = []
    locations = data.get("locations", data.get("timelineObjects", []))
    
    if isinstance(locations, list):
        for loc in locations[:5000]:  # Cap to avoid massive files
            if isinstance(loc, dict):
                # Semantic location visits
                place_visit = loc.get("placeVisit", {})
                if place_visit:
                    location = place_visit.get("location", {})
                    name = location.get("name", "")
                    address = location.get("address", "")
                    if name or address:
                        chunks.append({
                            "source": "location_history",
                            "category": "location",
                            "content": f"Visited place: {name}\nAddress: {address}",
                        })

    print(f"    Found {len(chunks)} location entries")
    return chunks


def parse_contacts(json_path: Path) -> list[dict]:
    """Parse Google Contacts."""
    print(f"  Parsing Contacts: {json_path.name}")
    try:
        with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except Exception as e:
        print(f"    Error: {e}")
        return []

    chunks = []
    contacts = data if isinstance(data, list) else data.get("connections", data.get("contacts", []))
    
    for contact in contacts:
        if isinstance(contact, dict):
            names = contact.get("names", [{}])
            name = names[0].get("displayName", "") if names else ""
            emails = [e.get("value", "") for e in contact.get("emailAddresses", [])]
            phones = [p.get("value", "") for p in contact.get("phoneNumbers", [])]
            orgs = [o.get("name", "") for o in contact.get("organizations", [])]
            
            if name:
                parts = [f"Contact: {name}"]
                if emails:
                    parts.append(f"Email: {', '.join(emails)}")
                if phones:
                    parts.append(f"Phone: {', '.join(phones)}")
                if orgs:
                    parts.append(f"Organization: {', '.join(orgs)}")
                
                chunks.append({
                    "source": "contacts",
                    "category": "relationship",
                    "content": "\n".join(parts),
                })

    print(f"    Found {len(chunks)} contacts")
    return chunks


def parse_generic_html(html_path: Path, source: str) -> list[dict]:
    """Generic HTML parser for any MyActivity file."""
    try:
        content = html_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return []
    
    text = html_to_text(content)
    if text and len(text) > 50:
        return [{
            "source": source,
            "category": "activity",
            "content": text,
            "file": str(html_path.name),
        }]
    return []


def parse_generic_json(json_path: Path, source: str) -> list[dict]:
    """Generic JSON parser for any data file."""
    try:
        with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except Exception:
        return []
    
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if text and len(text) > 50:
        return [{
            "source": source,
            "category": "data",
            "content": text[:100000],  # Cap at 100k chars
            "file": str(json_path.name),
        }]
    return []


# ── Phase 3: Discovery & Routing ────────────────────────────────────

def discover_and_parse(raw_dir: Path) -> list[dict]:
    """Walk the extracted takeout directory and route files to appropriate parsers."""
    all_chunks = []
    
    print("\n" + "=" * 60)
    print("DISCOVERING DATA SOURCES")
    print("=" * 60)
    
    # Walk the directory tree looking for known data sources
    for root, dirs, files in os.walk(raw_dir):
        root_path = Path(root)
        rel_path = root_path.relative_to(raw_dir)
        rel_str = str(rel_path).lower()
        
        for fname in files:
            fpath = root_path / fname
            fname_lower = fname.lower()
            
            # Skip tiny files
            try:
                if fpath.stat().st_size < 100:
                    continue
            except OSError:
                continue
            
            # ── Gemini / Google AI Activity ──
            if "gemini" in rel_str or "google ai" in rel_str:
                if fname_lower == "myactivity.html":
                    chunks = parse_gemini_activity(fpath)
                    all_chunks.extend(chunks)
                    continue
            
            # ── Google Search ──
            if "search" in rel_str and "my activity" in rel_str:
                if fname_lower == "myactivity.html":
                    chunks = parse_search_activity(fpath, "google_search")
                    all_chunks.extend(chunks)
                    continue
            
            # ── YouTube ──
            if "youtube" in rel_str:
                if fname_lower == "myactivity.html":
                    chunks = parse_search_activity(fpath, "youtube")
                    all_chunks.extend(chunks)
                    continue
                if fname_lower.endswith(".html"):
                    chunks = parse_generic_html(fpath, "youtube")
                    all_chunks.extend(chunks)
                    continue
            
            # ── Chrome Browser History ──
            if fname_lower == "browserhistory.json":
                chunks = parse_chrome_history(fpath)
                all_chunks.extend(chunks)
                continue
            
            # ── Location History / Semantic Location ──
            if "location" in rel_str or "maps" in rel_str:
                if fname_lower.endswith(".json"):
                    chunks = parse_location_history(fpath)
                    all_chunks.extend(chunks)
                    continue
            
            # ── Contacts ──
            if "contacts" in rel_str:
                if fname_lower.endswith(".json"):
                    chunks = parse_contacts(fpath)
                    all_chunks.extend(chunks)
                    continue
            
            # ── Other MyActivity files ──
            if fname_lower == "myactivity.html":
                source = str(rel_path).replace("\\", "/").replace("My Activity/", "")
                chunks = parse_search_activity(fpath, source)
                all_chunks.extend(chunks)
                continue
            
            # ── Other JSON data files ──
            if fname_lower.endswith(".json") and fpath.stat().st_size < 50_000_000:
                source = str(rel_path).replace("\\", "/")
                chunks = parse_generic_json(fpath, source)
                all_chunks.extend(chunks)
                continue
            
            # ── Other HTML files ──
            if fname_lower.endswith(".html") and fpath.stat().st_size < 50_000_000:
                source = str(rel_path).replace("\\", "/")
                chunks = parse_generic_html(fpath, source)
                all_chunks.extend(chunks)
                continue

    return all_chunks


# ── Phase 4: Chunking ───────────────────────────────────────────────

def chunk_entries(entries: list[dict], max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Combine individual entries into chunks of ~max_chars each."""
    chunks = []
    current_chunk_parts = []
    current_size = 0
    
    for entry in entries:
        # Format entry as readable text
        parts = []
        parts.append(f"[Source: {entry.get('source', 'unknown')}]")
        if entry.get('timestamp'):
            parts.append(f"[Timestamp: {entry['timestamp']}]")
        if entry.get('category'):
            parts.append(f"[Category: {entry['category']}]")
        
        if 'prompt' in entry and 'response' in entry:
            parts.append(f"User: {entry['prompt']}")
            parts.append(f"AI: {entry['response']}")
        elif 'content' in entry:
            parts.append(entry['content'])
        
        entry_text = "\n".join(parts) + "\n---\n"
        entry_size = len(entry_text)
        
        if current_size + entry_size > max_chars and current_chunk_parts:
            # Flush current chunk
            chunks.append("\n".join(current_chunk_parts))
            current_chunk_parts = []
            current_size = 0
        
        # If a single entry is larger than max_chars, split it
        if entry_size > max_chars:
            # Split large entries at line boundaries
            lines = entry_text.split('\n')
            sub_parts = []
            sub_size = 0
            for line in lines:
                if sub_size + len(line) + 1 > max_chars and sub_parts:
                    chunks.append("\n".join(sub_parts))
                    sub_parts = []
                    sub_size = 0
                sub_parts.append(line)
                sub_size += len(line) + 1
            if sub_parts:
                current_chunk_parts.extend(sub_parts)
                current_size += sub_size
        else:
            current_chunk_parts.append(entry_text)
            current_size += entry_size
    
    # Flush remaining
    if current_chunk_parts:
        chunks.append("\n".join(current_chunk_parts))
    
    return chunks


def write_chunks(chunks: list[str], output_dir: Path):
    """Write chunks to individual text files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        chunk_path = output_dir / f"chunk_{i:04d}.txt"
        chunk_path.write_text(chunk, encoding='utf-8')
    
    print(f"  Wrote {len(chunks)} chunks to {output_dir}")


# ── Main Pipeline ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Google Takeout Extractor & Chunker")
    parser.add_argument("--skip-extract", action="store_true", help="Skip ZIP extraction")
    parser.add_argument("--only-extract", action="store_true", help="Only extract ZIPs")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CASCADES Digital Twin — Takeout Extraction Pipeline")
    print("=" * 60)
    print(f"  Data dir:   {DATA_DIR}")
    print(f"  Raw dir:    {RAW_DIR}")
    print(f"  Chunks dir: {CHUNKS_DIR}")
    print(f"  Max chunk:  {MAX_CHUNK_CHARS:,} chars")
    print()
    
    # ── Phase 1: Extract ZIPs ──
    if not args.skip_extract:
        print("Phase 1: Extracting ZIP files...")
        extract_zips()
    else:
        print("Phase 1: Skipping ZIP extraction (--skip-extract)")
    
    if args.only_extract:
        print("\n--only-extract flag set, stopping here.")
        return
    
    # ── Phase 2: Parse all data sources ──
    print("\nPhase 2: Parsing data sources...")
    
    # Find all Takeout directories (there might be nested Takeout/ folders)
    takeout_dirs = []
    for item in RAW_DIR.rglob("Takeout"):
        if item.is_dir():
            takeout_dirs.append(item)
    
    if not takeout_dirs:
        # Maybe the files are directly in RAW_DIR
        takeout_dirs = [RAW_DIR]
    
    all_entries = []
    for td in takeout_dirs:
        print(f"\nProcessing: {td}")
        entries = discover_and_parse(td)
        all_entries.extend(entries)
    
    print(f"\n{'=' * 60}")
    print(f"Total entries extracted: {len(all_entries)}")
    
    # Break down by source
    sources = {}
    for entry in all_entries:
        src = entry.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    
    print("\nBreakdown by source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")
    
    # ── Phase 3: Chunk ──
    print(f"\nPhase 3: Creating chunks (max {MAX_CHUNK_CHARS:,} chars each)...")
    text_chunks = chunk_entries(all_entries)
    write_chunks(text_chunks, CHUNKS_DIR)
    
    # ── Summary ──
    total_chars = sum(len(c) for c in text_chunks)
    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total entries: {len(all_entries)}")
    print(f"  Total chunks:  {len(text_chunks)}")
    print(f"  Total chars:   {total_chars:,} (~{total_chars // 4:,} tokens)")
    print(f"  Chunks at:     {CHUNKS_DIR}")
    print(f"\nReady for entity extraction pipeline.")


if __name__ == "__main__":
    main()
