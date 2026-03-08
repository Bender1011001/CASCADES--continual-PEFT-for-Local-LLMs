"""
Self-Synthesizer — Extracts hard facts from conversations and synthesizes
dense declarative Q&A pairs for CASCADES parametric memory.

Prevents "Conversational Collapse" by training on factual Q&A instead of raw chat logs.

Pipeline:
    Raw chat → Fact Extraction → Q&A Synthesis → CASCADES Training Data
"""

import json
import re
from pathlib import Path
from typing import Optional


# ── Fact Extraction Patterns ───────────────────────────────────────
# These regex patterns catch common ways users share personal info.
# The Self-Synthesizer uses these to extract hard facts without needing
# an external LLM call (works offline, zero latency).

FACT_PATTERNS = [
    # Identity
    (r"my name is (\w+[\w\s]*)", "name", "The user's name is {0}."),
    (r"i'?m (\w+[\w\s]*)", "possible_name", None),  # Too ambiguous alone, needs context
    (r"call me (\w+[\w\s]*)", "name", "The user prefers to be called {0}."),

    # Profession / Work
    (r"i (?:work|am working) (?:as |in )?(?:a |an )?(.+?)(?:\.|,|$)", "job", "The user works as {0}."),
    (r"i'?m (?:a |an )?(\w+ ?\w*(?:engineer|developer|designer|manager|analyst|scientist|writer|teacher|student|doctor|nurse|lawyer|consultant|architect|artist))", "job", "The user is {0}."),
    (r"my (?:job|profession|career|work) is (.+?)(?:\.|,|$)", "job", "The user's profession is {0}."),

    # Location
    (r"i live in (.+?)(?:\.|,|$)", "location", "The user lives in {0}."),
    (r"i'?m from (.+?)(?:\.|,|$)", "origin", "The user is from {0}."),
    (r"i'?m (?:currently |now )?(?:based |located )?in (.+?)(?:\.|,|$)", "location", "The user is based in {0}."),

    # Preferences
    (r"my favorite (\w+) is (.+?)(?:\.|,|$)", "preference", "The user's favorite {0} is {1}."),
    (r"i (?:really )?(?:like|love|enjoy|prefer) (.+?)(?:\.|,|$)", "interest", "The user enjoys {0}."),
    (r"i (?:hate|dislike|don'?t like) (.+?)(?:\.|,|$)", "dislike", "The user dislikes {0}."),

    # Family / Relationships
    (r"my (\w+(?:'s)?) name is (\w+[\w\s]*)", "family", "The user's {0} name is {1}."),
    (r"i have (?:a |an )?(\w+ ?\w*(?:dog|cat|pet|child|kid|son|daughter|brother|sister|wife|husband|partner))", "family", "The user has {0}."),

    # Projects / Goals
    (r"i'?m (?:working on|building|creating|developing) (.+?)(?:\.|,|$)", "project", "The user is building {0}."),
    (r"my (?:goal|dream|plan) is (?:to )?(.+?)(?:\.|,|$)", "goal", "The user's goal is to {0}."),

    # Explicit memory requests
    (r"remember (?:that |this:? )?(.+?)(?:\.|$)", "explicit", "The user wants me to remember: {0}."),
    (r"don'?t forget (?:that |this:? )?(.+?)(?:\.|$)", "explicit", "The user wants me to remember: {0}."),
]


class Fact:
    """A single extracted fact about the user."""

    def __init__(self, category: str, value: str, template: Optional[str], source_text: str):
        self.category = category
        self.value = value.strip().rstrip(".,!?")
        self.template = template
        self.source_text = source_text

    def to_declarative(self) -> str:
        if self.template:
            groups = [g.strip().rstrip(".,!?") for g in self.value.split("|||")]
            return self.template.format(*groups)
        return f"Fact about the user: {self.value}"

    def to_qa_pairs(self) -> list[dict]:
        """Generate multiple Q&A pairs for a single fact (data augmentation)."""
        decl = self.to_declarative()
        pairs = []

        if self.category == "name":
            pairs.extend([
                {"q": "What is the user's name?", "a": decl},
                {"q": "What should I call the user?", "a": decl},
                {"q": "Who am I talking to?", "a": decl},
            ])
        elif self.category == "job":
            pairs.extend([
                {"q": "What does the user do for work?", "a": decl},
                {"q": "What is the user's profession?", "a": decl},
                {"q": "Tell me about the user's career.", "a": decl},
            ])
        elif self.category == "location":
            pairs.extend([
                {"q": "Where does the user live?", "a": decl},
                {"q": "Where is the user located?", "a": decl},
            ])
        elif self.category == "interest":
            pairs.extend([
                {"q": "What does the user enjoy?", "a": decl},
                {"q": "What are the user's interests?", "a": decl},
            ])
        elif self.category == "project":
            pairs.extend([
                {"q": "What is the user working on?", "a": decl},
                {"q": "Tell me about the user's current project.", "a": decl},
            ])
        elif self.category == "explicit":
            pairs.extend([
                {"q": "What did the user ask me to remember?", "a": decl},
                {"q": "What important things has the user told me?", "a": decl},
            ])
        else:
            pairs.append({"q": f"What do you know about the user's {self.category}?", "a": decl})

        return pairs

    def __repr__(self) -> str:
        return f"Fact({self.category}: {self.value!r})"


class SelfSynthesizer:
    """Extracts facts from conversations and synthesizes Q&A training data."""

    def __init__(self):
        self.known_facts: dict[str, Fact] = {}  # category → latest fact (dedup)

    def extract_facts(self, text: str) -> list[Fact]:
        """Extract facts from a single user message."""
        facts = []
        text_lower = text.lower().strip()

        for pattern, category, template in FACT_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                if len(match.groups()) == 1:
                    value = match.group(1)
                elif len(match.groups()) >= 2:
                    value = "|||".join(match.groups())
                else:
                    continue

                # Skip very short or likely false positives
                if len(value.strip()) < 2:
                    continue
                if category == "possible_name":
                    # "I'm tired" shouldn't be treated as a name
                    common_words = {"tired", "hungry", "happy", "sad", "fine", "good",
                                    "great", "okay", "sorry", "sure", "not", "just",
                                    "really", "very", "so", "too", "also", "going",
                                    "trying", "looking", "thinking", "wondering",
                                    "interested", "curious", "confused", "lost"}
                    if value.strip().lower().split()[0] in common_words:
                        continue
                    category = "name"
                    template = "The user's name is {0}."

                fact = Fact(category, value, template, text)
                facts.append(fact)

                # Update known facts (latest wins for same category)
                self.known_facts[category] = fact

        return facts

    def process_conversation(self, messages: list[dict]) -> list[Fact]:
        """Extract all facts from a conversation's user messages."""
        all_facts = []
        for msg in messages:
            if msg.get("role") == "user":
                facts = self.extract_facts(msg["content"])
                all_facts.extend(facts)
        return all_facts

    def synthesize_training_data(
        self,
        conversations_jsonl: Path,
        output_path: Path,
    ) -> int:
        """Read exported conversations, extract facts, synthesize Q&A pairs.

        Returns the number of training examples written.
        """
        # Read all conversations
        all_facts: list[Fact] = []

        with open(conversations_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                facts = self.extract_facts(example.get("prompt", ""))
                all_facts.extend(facts)

                # Also check if user explicitly flagged this as important
                if example.get("flagged"):
                    # Create a generic memory fact from the flagged exchange
                    fact = Fact(
                        "explicit",
                        example["prompt"],
                        None,
                        example["prompt"],
                    )
                    all_facts.append(fact)

        # Deduplicate: keep latest per category
        unique: dict[str, Fact] = {}
        for fact in all_facts:
            key = f"{fact.category}:{fact.value[:30]}"
            unique[key] = fact

        # Synthesize Q&A pairs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for fact in unique.values():
                for qa in fact.to_qa_pairs():
                    training_example = {
                        "prompt": qa["q"],
                        "response": f"<think>Recalling what I know about the user.</think>{qa['a']}",
                        "category": fact.category,
                        "source": fact.source_text[:200],
                    }
                    f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                    count += 1

        print(f"Self-Synthesizer: {len(unique)} unique facts → {count} Q&A training pairs")
        return count

    def get_identity_summary(self) -> str:
        """Return a natural language summary of everything known about the user."""
        if not self.known_facts:
            return "I don't know much about you yet."

        parts = []
        if "name" in self.known_facts:
            parts.append(self.known_facts["name"].to_declarative())
        if "job" in self.known_facts:
            parts.append(self.known_facts["job"].to_declarative())
        if "location" in self.known_facts:
            parts.append(self.known_facts["location"].to_declarative())
        for cat, fact in self.known_facts.items():
            if cat not in ("name", "job", "location", "possible_name"):
                parts.append(fact.to_declarative())

        return " ".join(parts)
