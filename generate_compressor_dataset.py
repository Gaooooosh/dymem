#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate pretraining dataset for a compressor attached to an LLM (e.g., Qwen2.5-3B) using Evicted/Noise/Query zones.

Output format: JSON Lines (train.jsonl and optionally valid.jsonl), each line has fields:
  - type: "A" | "B" | "C" | "D"
  - evicted_text: string
  - noise_text: string
  - query_text: string (e.g., "Question: ...\nAnswer:")
  - full_text: string (evicted + two newlines + noise + two newlines + query)
  - answer_text: string or null

Dependencies:
  - Python 3 standard library
  - datasets (pip install datasets) for optional natural text sampling from HuggingFace Hub dataset vllg/loong_c4
  - transformers (pip install transformers) for optional precise token budgeting via AutoTokenizer

Notes about sink tokens:
  - Training code will automatically prepend [sink_token] * SINK_N to each sample before feeding the model.
  - This script does NOT include sink tokens in full_text. Only generate the text segments (evicted/noise/query).

HuggingFace datasets usage example is provided at the bottom of the file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# -----------------------------
# Config constants and budgets
# -----------------------------

DEFAULT_NUM_A = 100
DEFAULT_NUM_B = 75
DEFAULT_NUM_C = 50
DEFAULT_NUM_D = 25

# Whether to use real natural text from vllg/loong_c4.
USE_LOONG_C4_DEFAULT = True

# Sink token count (training side will prepend [sink_token] * SINK_N automatically).
SINK_N_DEFAULT = 128

# Tokenizer controls (optional precise token budgeting)
USE_TOKENIZER_DEFAULT = True
TOKENIZER_NAME_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"

# Token-length budgets (precise, preferred when tokenizer is available)
EVICTED_MIN_TOKENS = 512
EVICTED_MAX_TOKENS = 3000
NOISE_MIN_TOKENS = 1000
NOISE_MAX_TOKENS = 2000
QUERY_MIN_TOKENS = 50
QUERY_MAX_TOKENS = 200
MAX_FULL_TEXT_TOKENS = 1024*4

# Char-length budgets (rough approximations to token lengths)
EVICTED_MIN_CHARS = 4000   # ~800 tokens eq.
EVICTED_MAX_CHARS = 10000  # ~4000 tokens eq.

NOISE_MIN_CHARS = 6000     # ~2000 tokens eq.
NOISE_MAX_CHARS = 14000    # ~6000 tokens eq.

QUERY_MIN_CHARS = 300      # ~50 tokens eq.
QUERY_MAX_CHARS = 1200     # ~200 tokens eq.

# Global safety cap to keep full_text within a conservative bound
MAX_FULL_TEXT_CHARS = 16000

# Validation split ratio default (0 means no validation file)
VALID_RATIO_DEFAULT = 0.0


# -----------------------------
# Utilities
# -----------------------------

def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None

_TQDM = _get_tqdm()

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clamp_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cutoff = max_chars
    # Try to end at a sentence or newline boundary for nicer truncation
    boundary = text.rfind(". ", 0, max_chars)
    if boundary == -1:
        boundary = text.rfind("\n", 0, max_chars)
    if boundary == -1:
        boundary = text.rfind(".\n", 0, max_chars)
    if boundary > 0 and boundary >= max_chars // 2:
        cutoff = boundary + 1
    return text[:cutoff].rstrip()


def safe_join(parts: List[str]) -> str:
    return normalize_whitespace("\n\n".join([p.strip() for p in parts if p is not None]))


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


# -----------------------------
# Optional: tokenizer helper
# -----------------------------

class TokenizerHelper:
    def __init__(self, tokenizer_name: str):
        self._available = False
        self._tok = None
        try:
            from transformers import AutoTokenizer  # type: ignore
            self._tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            self._available = True
        except Exception:
            self._available = False

    def available(self) -> bool:
        return self._available and self._tok is not None

    def count_tokens(self, text: str) -> int:
        if not self.available():
            return len(text)
        tokens = self._tok.encode(text, add_special_tokens=False)
        return int(len(tokens))

    def clamp_tokens(self, text: str, max_tokens: int) -> str:
        if not self.available():
            return clamp_text(text, max_tokens)
        tokens = self._tok.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        tokens = tokens[:max_tokens]
        return self._tok.decode(tokens, skip_special_tokens=True)


# -----------------------------
# Optional: vllg/loong_c4 loader
# -----------------------------

class LoongC4Sampler:
    def __init__(self, use_loong_c4: bool, tokenizer: Optional[TokenizerHelper] = None):
        self.use_loong_c4 = use_loong_c4
        self.ds = None
        self._loaded = False
        self._failed = False
        self.tokenizer = tokenizer

        if self.use_loong_c4:
            try:
                from datasets import load_dataset  # type: ignore
            except Exception:
                self._failed = True
                self.use_loong_c4 = False
                return
            try:
                self.ds = load_dataset("vllg/loong_c4", split="train", streaming=False)
                self._loaded = True
            except Exception:
                # Fallback silently to pseudo-natural text
                self._failed = True
                self.use_loong_c4 = False

    def _record_to_text(self, rec: Dict[str, Any]) -> str:
        for key in ("text", "content", "raw_content"):
            if key in rec and isinstance(rec[key], str):
                return rec[key]
        # If unknown schema, concatenate string-like fields
        parts = []
        for k, v in rec.items():
            if isinstance(v, str) and k not in {"id"}:
                parts.append(v)
        return "\n\n".join(parts) if parts else ""

    def sample_paragraph(self, min_chars: int, max_chars: int) -> str:
        if not self.use_loong_c4 or self.ds is None:
            return generate_natural_text(min_chars, max_chars)

        # The dataset supports indexing when not streaming.
        size = len(self.ds)
        accum: List[str] = []
        total = 0
        # Iterate random indices until reaching min_chars
        attempts = 0
        while total < min_chars and attempts < max(1000, min_chars // 100):
            idx = random.randint(0, size - 1)
            rec = self.ds[idx]
            txt = self._record_to_text(rec)
            if not txt:
                attempts += 1
                continue
            txt = normalize_whitespace(txt)
            # Avoid extremely short fragments
            if len(txt) < 100:
                attempts += 1
                continue
            accum.append(txt)
            total += len(txt) + 2
            attempts += 1

        paragraph = safe_join(accum)
        paragraph = clamp_text(paragraph, max_chars)
        return paragraph

    def sample_paragraph_tokens(self, min_tokens: int, max_tokens: int) -> str:
        tok = self.tokenizer
        if tok is None or not tok.available():
            # Fallback: approximate via chars using a multiplier
            approx_min_chars = max(512, int(min_tokens * 2))
            approx_max_chars = max(1024, int(max_tokens * 2))
            return self.sample_paragraph(approx_min_chars, approx_max_chars)

        if not self.use_loong_c4 or self.ds is None:
            # Use pseudo-natural and clamp by tokens
            base = generate_natural_text(int(min_tokens * 2), int(max_tokens * 2))
            return tok.clamp_tokens(base, max_tokens)

        size = len(self.ds)
        accum: List[str] = []
        total_tokens = 0
        attempts = 0
        while total_tokens < min_tokens and attempts < max(1000, min_tokens // 2):
            idx = random.randint(0, size - 1)
            rec = self.ds[idx]
            txt = self._record_to_text(rec)
            if not txt:
                attempts += 1
                continue
            txt = normalize_whitespace(txt)
            if len(txt) < 100:
                attempts += 1
                continue
            accum.append(txt)
            total_tokens += tok.count_tokens(txt) + 1
            attempts += 1

        paragraph = safe_join(accum)
        paragraph = tok.clamp_tokens(paragraph, max_tokens)
        return paragraph


# -----------------------------
# Pseudo-natural text fallback
# -----------------------------

_WORDS_NOUN = (
    "system", "network", "dataset", "model", "algorithm", "paper", "memory", "sequence",
    "window", "token", "compressor", "analysis", "experiment", "result", "architecture",
    "framework", "service", "document", "language", "feature", "performance", "quality",
)
_WORDS_VERB = (
    "runs", "builds", "tests", "learns", "measures", "compresses", "retrieves", "predicts",
    "generates", "improves", "ensures", "maintains", "monitors", "summarizes", "scans",
)
_WORDS_ADJ = (
    "robust", "efficient", "scalable", "modular", "lightweight", "flexible", "accurate",
    "reliable", "comprehensive", "consistent", "portable", "interactive", "balanced",
)
_WORDS_MISC = (
    "however", "therefore", "meanwhile", "notably", "in practice", "for example",
    "in addition", "moreover", "specifically", "as a result",
)


def random_word() -> str:
    pool = random.choice([_WORDS_NOUN, _WORDS_VERB, _WORDS_ADJ])
    return random.choice(pool)


def random_sentence(min_w: int = 8, max_w: int = 20) -> str:
    n = random.randint(min_w, max_w)
    words = []
    for _ in range(n):
        w = random_word()
        if random.random() < 0.15:
            w = random.choice(_WORDS_MISC)
        words.append(w)
    s = " ".join(words)
    s = s.capitalize()
    punctuation = random.choice([".", ".", ".", "?", "!"])
    return s + punctuation


def random_paragraph(min_s: int = 4, max_s: int = 12) -> str:
    n = random.randint(min_s, max_s)
    lines = []
    for i in range(n):
        sent = random_sentence()
        if i > 0 and random.random() < 0.2:
            # Insert a quoted phrase or list-like item to resemble natural text
            sent += " " + random.choice([
                '"in context learning"', '"sliding window"', '"long-range retrieval"',
                'note: performance varies', 'cf. baseline methods'
            ])
        lines.append(sent)
    return " ".join(lines)


def generate_natural_text(min_chars: int, max_chars: int) -> str:
    pieces: List[str] = []
    total = 0
    while total < min_chars:
        para = random_paragraph()
        pieces.append(para)
        total += len(para) + 2
    text = safe_join(pieces)
    return clamp_text(text, max_chars)


# -----------------------------
# Builders
# -----------------------------

def build_full_text(evicted: str, noise: str, query: str, max_total_chars: int = MAX_FULL_TEXT_CHARS) -> Tuple[str, str, str]:
    evicted = normalize_whitespace(evicted)
    noise = normalize_whitespace(noise)
    query = normalize_whitespace(query)

    # Clamp in order to keep structure: evicted first, then noise, then query
    if len(evicted) > max_total_chars:
        evicted = clamp_text(evicted, max_total_chars)
        noise = ""
        query = ""
        full_text = safe_join([evicted, noise, query])
        return full_text, evicted, noise

    remaining = max_total_chars - len(evicted) - 2  # two newlines
    if remaining < 0:
        remaining = 0

    # Assign budget to noise and query (70% noise, 30% query as a heuristic)
    noise_budget = int(remaining * 0.7)
    query_budget = remaining - noise_budget

    if len(noise) > noise_budget:
        noise = clamp_text(noise, max(0, noise_budget))

    # After clamping noise, recompute remaining for query precisely
    remaining_for_query = max_total_chars - (len(evicted) + 2 + len(noise) + 2)
    if remaining_for_query < 0:
        remaining_for_query = 0
    if len(query) > remaining_for_query:
        query = clamp_text(query, remaining_for_query)

    full_text = safe_join([evicted, noise, query])
    # As a final guard
    if len(full_text) > max_total_chars:
        full_text = clamp_text(full_text, max_total_chars)
    return full_text, evicted, noise


def build_full_text_tokens(evicted: str, noise: str, query: str, tokenizer: TokenizerHelper, max_total_tokens: int = MAX_FULL_TEXT_TOKENS) -> Tuple[str, str, str]:
    evicted = normalize_whitespace(evicted)
    noise = normalize_whitespace(noise)
    query = normalize_whitespace(query)

    ev_tokens = tokenizer.count_tokens(evicted)
    if ev_tokens > max_total_tokens:
        evicted = tokenizer.clamp_tokens(evicted, max_total_tokens)
        full_text = safe_join([evicted, "", ""]) 
        return full_text, evicted, ""

    # Heuristic: allocate remaining tokens mostly to noise, then query
    remaining = max_total_tokens - ev_tokens
    noise_budget = int(remaining * 0.7)
    query_budget = remaining - noise_budget

    noise = tokenizer.clamp_tokens(noise, max(0, noise_budget))
    query = tokenizer.clamp_tokens(query, max(0, query_budget))

    full_text = safe_join([evicted, noise, query])
    total = tokenizer.count_tokens(full_text)
    if total > max_total_tokens:
        full_text = tokenizer.clamp_tokens(full_text, max_total_tokens)
    return full_text, evicted, noise


def write_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def write_jsonl_line(fh, sample: Dict[str, Any]) -> None:
    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


# -----------------------------
# Sample type generators
# -----------------------------

COLORS = [
    "RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE", "CYAN", "MAGENTA", "BLACK", "WHITE",
    "PINK", "BROWN", "GRAY", "SILVER", "GOLD", "BEIGE", "MAROON", "NAVY", "TEAL", "OLIVE",
    "LIME", "INDIGO", "VIOLET", "TURQUOISE", "AQUA", "AMBER", "BRONZE", "COPPER", "CRIMSON", "SCARLET",
    "CHARCOAL", "KHAKI", "TAUPE", "PEACH", "APRICOT", "CORAL", "SALMON", "MINT", "EMERALD", "JADE",
    "SAPPHIRE", "RUBY", "TOPAZ", "AZURE", "CERULEAN", "COBALT", "IVORY", "LAVENDER", "PLUM", "MUSTARD",
    "FUCHSIA", "SEPIA", "SLATE", "SKY BLUE", "SEAFOAM", "FOREST GREEN", "MIDNIGHT BLUE", "STEEL BLUE", "SAND",
    "CHOCOLATE", "CHESTNUT", "BURGUNDY", "MAUVE", "ROSE", "PEARL", "SMOKE", "SUNSET", "SUNRISE"
]
CITIES = [
    "LONDON", "PARIS", "TOKYO", "BERLIN", "MADRID", "ROME", "DUBLIN", "SYDNEY", "TORONTO", "SINGAPORE",
    "NEW YORK", "LOS ANGELES", "SAN FRANCISCO", "CHICAGO", "HOUSTON", "MIAMI", "SEATTLE", "BOSTON", "WASHINGTON", "PHILADELPHIA",
    "ATLANTA", "DETROIT", "BALTIMORE", "PHOENIX", "LAS VEGAS", "DALLAS", "AUSTIN", "SAN DIEGO", "SAN JOSE", "MEXICO CITY",
    "BUENOS AIRES", "SANTIAGO", "RIO DE JANEIRO", "SAO PAULO", "LIMA", "BOGOTA", "CARACAS", "QUITO", "MEDELLIN", "CAPE TOWN",
    "JOHANNESBURG", "NAIROBI", "CAIRO", "ALGIERS", "CASABLANCA", "ACCRA", "LAGOS", "ABUJA", "ADDIS ABABA", "DOHA",
    "DUBAI", "ABU DHABI", "RIYADH", "JEDDAH", "KUWAIT CITY", "MANAMA", "MUSCAT", "TEHRAN", "ANKARA", "ISTANBUL",
    "MOSCOW", "SAINT PETERSBURG", "WARSAW", "PRAGUE", "BUDAPEST", "VIENNA", "ZURICH", "GENEVA", "MUNICH", "FRANKFURT",
    "HAMBURG", "COLOGNE", "AMSTERDAM", "ROTTERDAM", "BRUSSELS", "OSLO", "STOCKHOLM", "HELSINKI", "COPENHAGEN", "REYKJAVIK",
    "ATHENS", "BARCELONA", "VALENCIA", "SEVILLE", "LISBON", "PORTO", "BELGRADE", "SOFIA", "BUCHAREST", "TIRANA",
    "TBILISI", "YEREVAN", "BAKU", "ALMATY", "ASTANA", "BANGKOK", "HANOI", "HO CHI MINH CITY", "PHNOM PENH", "VIENTIANE",
    "YANGON", "KUALA LUMPUR", "JAKARTA", "MANILA", "TAIPEI", "SEOUL", "BUSAN", "BEIJING", "SHANGHAI", "GUANGZHOU",
    "SHENZHEN", "CHENGDU", "HANGZHOU", "WUHAN", "NANJING", "XIAN", "CHONGQING", "SUZHOU", "NINGBO", "TIANJIN",
    "ZHENGZHOU", "SHIJIAZHUANG", "XIAMEN", "FUZHOU", "NANCHANG", "CHANGSHA", "HAIKOU", "HARBIN", "CHANGCHUN", "DALIAN",
    "QINGDAO", "JINAN", "URUMQI", "LANZHOU"
]
COUNTRIES = [
    "UK", "FRANCE", "JAPAN", "GERMANY", "SPAIN", "ITALY", "IRELAND", "AUSTRALIA", "CANADA", "SINGAPORE",
    "UNITED STATES", "CHINA", "INDIA", "BRAZIL", "RUSSIA", "MEXICO", "ARGENTINA", "CHILE", "PERU", "COLOMBIA",
    "SOUTH AFRICA", "EGYPT", "NIGERIA", "KENYA", "TURKEY", "SAUDI ARABIA", "UNITED ARAB EMIRATES", "QATAR", "KUWAIT", "BAHRAIN",
    "OMAN", "IRAN", "IRAQ", "ISRAEL", "JORDAN", "LEBANON", "SYRIA", "GREECE", "PORTUGAL", "NETHERLANDS",
    "BELGIUM", "SWITZERLAND", "AUSTRIA", "POLAND", "CZECHIA", "HUNGARY", "ROMANIA", "BULGARIA", "SERBIA", "CROATIA",
    "SLOVENIA", "SLOVAKIA", "UKRAINE", "BELARUS", "LITHUANIA", "LATVIA", "ESTONIA", "NORWAY", "SWEDEN", "FINLAND",
    "DENMARK", "ICELAND", "MALTA", "CYPRUS", "LUXEMBOURG", "MONACO", "ANDORRA", "SAN MARINO", "VATICAN", "MOROCCO",
    "ALGERIA", "TUNISIA", "LIBYA", "ETHIOPIA", "GHANA", "IVORY COAST", "SENEGAL", "TANZANIA", "UGANDA", "ZAMBIA",
    "ZIMBABWE", "ANGOLA", "MOZAMBIQUE", "NAMIBIA", "BOTSWANA", "CAMEROON", "CONGO", "DR CONGO", "SUDAN", "SOUTH SUDAN",
    "PAKISTAN", "BANGLADESH", "SRI LANKA", "NEPAL", "BHUTAN", "MONGOLIA", "LAOS", "CAMBODIA", "VIETNAM", "THAILAND",
    "MALAYSIA", "INDONESIA", "PHILIPPINES", "TAIWAN", "SOUTH KOREA", "NORTH KOREA", "NEW ZEALAND", "UNITED KINGDOM"
]
NAMES = [
    "alice", "bob", "charlie", "diana", "edgar", "fiona", "george", "helen", "ivan", "julia", "marcus", "lee",
    "aaron", "abby", "adam", "adrian", "albert", "alexa", "alex", "alfred", "alina", "allen", "amber", "amy",
    "andrew", "angel", "angela", "anna", "anne", "anthony", "arthur", "ava", "barbara", "ben", "benjamin", "beth",
    "betty", "blake", "brad", "brandon", "brenda", "brian", "brittany", "brooke", "bruce", "cameron", "carla",
    "carlos", "carmen", "caroline", "carter", "catherine", "charles", "chelsea", "chris", "christopher", "cindy",
    "claire", "clark", "claudia", "cole", "colin", "connie", "craig", "dan", "daniel", "danny", "david", "dawn",
    "dean", "deborah", "dennis", "derek", "diane", "donald", "donna", "doris", "doug", "douglas", "dylan", "edmund",
    "edward", "elaine", "elena", "eleanor", "eli", "eliza", "elizabeth", "ellen", "emily", "emma", "eric", "erika",
    "ethan", "eva", "evan", "frank", "fred", "gabriel", "gary", "gerald", "gina", "gloria", "grant", "greg",
    "gregory", "hannah", "harold", "harry", "henry", "holly", "ian", "irene", "isaac", "isabel", "jack", "jacob",
    "jade", "james", "jamie", "janet", "jane", "jason", "jeff", "jeffrey", "jennifer", "jenny", "jeremy", "jerry",
    "jesse", "jessica", "jim", "joan", "joanna", "joe", "joel", "john", "johnny", "jon", "jonathan", "jose",
    "joseph", "josh", "joshua", "joy", "juan", "judy", "julian", "julie", "justin", "karen", "karl", "kate",
    "katherine", "kathy", "keith", "kelly", "kevin", "kim", "kristin", "kylie", "laura", "lauren", "leo", "leon",
    "leonard", "leslie", "liam", "linda", "lisa", "liz", "logan", "lois", "louis", "lucas", "lucy", "luke",
    "lydia", "madison", "maria", "marie", "mark", "martha", "martin", "marvin", "mary", "mason", "matt", "matthew",
    "megan", "melanie", "melissa", "micah", "michael", "michelle", "miguel", "mike", "morgan", "nancy", "natasha",
    "nathan", "neil", "nicholas", "nick", "nicole", "noah", "olivia", "oscar", "owen", "paige", "pam", "pamela",
    "patricia", "patrick", "paul", "paula", "peter", "phil", "philip", "phoebe", "rachel", "rafael", "ralph",
    "ray", "raymond", "rebecca", "richard", "riley", "rita", "robert", "robin", "roger", "roland", "ron", "ronald",
    "rose", "roy", "ruby", "russell", "ryan", "sam", "samantha", "samuel", "sara", "sarah", "scott", "sean",
    "selena", "serena", "shane", "sharon", "sheila", "sophia", "stacey", "stanley", "stephanie", "stephen", "steve",
    "steven", "sue", "susan", "suzanne", "sydney", "taylor", "terry", "thomas", "tim", "timothy", "tina", "todd",
    "tom", "tony", "tracy", "travis", "tyler", "valerie", "vanessa", "victor", "victoria", "vincent", "virginia",
    "wesley", "william", "wills", "wilson", "winston", "xavier", "yolanda", "zach", "zachary"
]


def _random_phrase(min_words: int = 1, max_words: int = 8) -> str:
    n = random.randint(min_words, max_words)
    words = []
    for _ in range(n):
        w = random.choice(_WORDS_ADJ + _WORDS_NOUN)
        words.append(w)
    return " ".join(words).upper()


def _kv_block(num_keys: int) -> Tuple[str, Dict[str, str]]:
    lines = []
    mapping: Dict[str, str] = {}
    for _ in range(num_keys):
        kind = random.choice(["ID", "KEY", "User", "Code"])
        if kind in {"ID", "KEY", "Code"}:
            suffix = str(random.randint(1, 999999))
            key = f"{kind}_{suffix}"
        else:
            key = f"User_{random.choice(NAMES)}"

        vtype = random.choice(["color", "city", "country", "phrase"])
        if vtype == "color":
            val = random.choice(COLORS)
        elif vtype == "city":
            val = random.choice(CITIES)
        elif vtype == "country":
            val = random.choice(COUNTRIES)
        else:
            val = _random_phrase()

        mapping[key] = val
        lines.append(f"{key} = {val}")
    block = "\n".join(lines)
    return block, mapping


def generate_sample_A(sampler: LoongC4Sampler, tokenizer: Optional[TokenizerHelper] = None) -> Dict[str, Any]:
    num_keys = random.randint(5, 50)
    block, mapping = _kv_block(num_keys)

    # Optionally add some natural padding to reach evicted length target
    if tokenizer and tokenizer.available():
        # Build evicted by tokens
        padding = sampler.sample_paragraph_tokens(max(0, EVICTED_MIN_TOKENS - tokenizer.count_tokens(block)), EVICTED_MAX_TOKENS)
        evicted_text = safe_join([block, padding]) if padding else block
    else:
        evicted_padding = sampler.sample_paragraph(min_chars=max(0, EVICTED_MIN_CHARS - len(block)), max_chars=EVICTED_MAX_CHARS)
        evicted_text = safe_join([block, evicted_padding]) if evicted_padding else block

    # Noise zone
    noise_text = sampler.sample_paragraph_tokens(NOISE_MIN_TOKENS, NOISE_MAX_TOKENS) if (tokenizer and tokenizer.available()) else sampler.sample_paragraph(NOISE_MIN_CHARS, NOISE_MAX_CHARS)

    # Pick a key to query
    qkey = random.choice(list(mapping.keys()))
    qval = mapping[qkey]

    templates = [
        f"Question: What is the value of {qkey}?\nAnswer:",
        f"Question: Provide the value associated with {qkey}.\nAnswer:",
        f"Question: What is the entry for {qkey}?\nAnswer:",
    ]
    query_text = random.choice(templates)

    # Clamp query to budget range
    if tokenizer and tokenizer.available():
        query_text = tokenizer.clamp_tokens(query_text, QUERY_MAX_TOKENS)
        if tokenizer.count_tokens(query_text) < QUERY_MIN_TOKENS:
            pad = generate_natural_text(128, 512)
            query_text = tokenizer.clamp_tokens(query_text + "\n" + pad, QUERY_MAX_TOKENS)
    else:
        query_text = clamp_text(query_text, QUERY_MAX_CHARS)
        if len(query_text) < QUERY_MIN_CHARS:
            query_text = query_text + "\n" + generate_natural_text(QUERY_MIN_CHARS - len(query_text), QUERY_MAX_CHARS)

    if tokenizer and tokenizer.available():
        full_text, evicted_text, noise_text = build_full_text_tokens(evicted_text, noise_text, query_text, tokenizer, MAX_FULL_TEXT_TOKENS)
    else:
        full_text, evicted_text, noise_text = build_full_text(evicted_text, noise_text, query_text)

    return {
        "type": "A",
        "evicted_text": evicted_text,
        "noise_text": noise_text,
        "query_text": query_text,
        "full_text": full_text,
        "answer_text": qval,
    }


def _random_secret() -> Tuple[str, str, str]:
    kind = random.choice(["code", "agent", "access"])
    if kind == "code":
        value = str(random.randint(100000, 999999))
        sentence = f"The secret code is {value}."
        query = "What is the secret code mentioned earlier?\nAnswer:"
    elif kind == "agent":
        value = random.choice(["Marcus Lee", "Alice Chen", "David Brown", "Sophia Wang"])  # name unlikely in base
        sentence = f"The hidden agent's name is {value}."
        query = "What is the hidden agent's name?\nAnswer:"
    else:
        value = str(random.randint(100000, 999999))
        sentence = f"The special access key is {value}."
        query = "Please provide the special access key from the earlier text.\nAnswer:"
    return sentence, value, query


def insert_at_random(text: str, insert: str) -> str:
    if not text:
        return insert
    pos = random.randint(0, len(text))
    return text[:pos] + ("\n" if pos > 0 else "") + insert + "\n" + text[pos:]


def generate_sample_B(sampler: LoongC4Sampler, tokenizer: Optional[TokenizerHelper] = None) -> Dict[str, Any]:
    base = sampler.sample_paragraph_tokens(EVICTED_MIN_TOKENS, EVICTED_MAX_TOKENS) if (tokenizer and tokenizer.available()) else sampler.sample_paragraph(EVICTED_MIN_CHARS, EVICTED_MAX_CHARS)
    sentence, value, query_text = _random_secret()
    # Insert secret sentence at random
    evicted_text = insert_at_random(base, sentence)

    # Noise zone
    if tokenizer and tokenizer.available():
        noise_text = sampler.sample_paragraph_tokens(1500, 4000)
    else:
        noise_text = sampler.sample_paragraph(1500, 4000)

    # Ensure query length bounds
    if tokenizer and tokenizer.available():
        query_text = tokenizer.clamp_tokens(query_text, QUERY_MAX_TOKENS)
        if tokenizer.count_tokens(query_text) < QUERY_MIN_TOKENS:
            pad = generate_natural_text(128, 512)
            query_text = tokenizer.clamp_tokens(query_text + "\n" + pad, QUERY_MAX_TOKENS)
    else:
        query_text = clamp_text(query_text, QUERY_MAX_CHARS)
        if len(query_text) < QUERY_MIN_CHARS:
            query_text = query_text + "\n" + generate_natural_text(QUERY_MIN_CHARS - len(query_text), QUERY_MAX_CHARS)

    if tokenizer and tokenizer.available():
        full_text, evicted_text, noise_text = build_full_text_tokens(evicted_text, noise_text, query_text, tokenizer, MAX_FULL_TEXT_TOKENS)
    else:
        full_text, evicted_text, noise_text = build_full_text(evicted_text, noise_text, query_text)
    return {
        "type": "B",
        "evicted_text": evicted_text,
        "noise_text": noise_text,
        "query_text": query_text,
        "full_text": full_text,
        "answer_text": value,
    }


def _unique_token_not_in(text: str) -> str:
    # Prefer a 6-digit number or uppercase code unlikely to occur
    for _ in range(10):
        if random.random() < 0.5:
            token = str(random.randint(100000, 999999))
        else:
            token = "".join(random.choice(string.ascii_uppercase) for _ in range(6))
        if token not in text:
            return token
    return "".join(random.choice(string.ascii_uppercase) for _ in range(8))


def generate_sample_C(sampler: LoongC4Sampler, tokenizer: Optional[TokenizerHelper] = None) -> Dict[str, Any]:
    base = sampler.sample_paragraph_tokens(EVICTED_MIN_TOKENS, EVICTED_MAX_TOKENS) if (tokenizer and tokenizer.available()) else sampler.sample_paragraph(EVICTED_MIN_CHARS, EVICTED_MAX_CHARS)
    value = _unique_token_not_in(base)
    needle_line = random.choice([f"needle: {value}", f"needle = {value}"])
    evicted_text = insert_at_random(base, needle_line)

    # Ensure the needle appears exactly once
    # If accidental duplication, rebuild by replacing extra occurrences
    occurrences = evicted_text.count(value)
    if occurrences > 1:
        evicted_text = re.sub(re.escape(value), lambda m, count=[0]: (count.__setitem__(0, count[0]+1) or (value if count[0]==1 else "")), evicted_text)

    noise_text = sampler.sample_paragraph_tokens(NOISE_MIN_TOKENS, NOISE_MAX_TOKENS) if (tokenizer and tokenizer.available()) else sampler.sample_paragraph(NOISE_MIN_CHARS, NOISE_MAX_CHARS)

    query_templates = [
        "What is the needle?\nAnswer:",
        "Please repeat the value associated with \"needle\".\nAnswer:",
    ]
    query_text = random.choice(query_templates)
    if tokenizer and tokenizer.available():
        query_text = tokenizer.clamp_tokens(query_text, QUERY_MAX_TOKENS)
        if tokenizer.count_tokens(query_text) < QUERY_MIN_TOKENS:
            pad = generate_natural_text(128, 512)
            query_text = tokenizer.clamp_tokens(query_text + "\n" + pad, QUERY_MAX_TOKENS)
    else:
        query_text = clamp_text(query_text, QUERY_MAX_CHARS)
        if len(query_text) < QUERY_MIN_CHARS:
            query_text = query_text + "\n" + generate_natural_text(QUERY_MIN_CHARS - len(query_text), QUERY_MAX_CHARS)

    if tokenizer and tokenizer.available():
        full_text, evicted_text, noise_text = build_full_text_tokens(evicted_text, noise_text, query_text, tokenizer, MAX_FULL_TEXT_TOKENS)
    else:
        full_text, evicted_text, noise_text = build_full_text(evicted_text, noise_text, query_text)
    return {
        "type": "C",
        "evicted_text": evicted_text,
        "noise_text": noise_text,
        "query_text": query_text,
        "full_text": full_text,
        "answer_text": value,
    }


def generate_sample_D(sampler: LoongC4Sampler, tokenizer: Optional[TokenizerHelper] = None) -> Dict[str, Any]:
    natural = sampler.sample_paragraph_tokens(NOISE_MIN_TOKENS, NOISE_MAX_TOKENS) if (tokenizer and tokenizer.available()) else sampler.sample_paragraph(NOISE_MIN_CHARS, NOISE_MAX_CHARS)
    evicted_text = ""
    query_text = ""
    noise_text = natural
    if tokenizer and tokenizer.available():
        full_text = tokenizer.clamp_tokens(normalize_whitespace(natural), MAX_FULL_TEXT_TOKENS)
    else:
        full_text = clamp_text(normalize_whitespace(natural), MAX_FULL_TEXT_CHARS)
    return {
        "type": "D",
        "evicted_text": evicted_text,
        "noise_text": noise_text,
        "query_text": query_text,
        "full_text": full_text,
        "answer_text": None,
    }


# -----------------------------
# Validation
# -----------------------------

def validate_sample(sample: Dict[str, Any], tokenizer: Optional[TokenizerHelper] = None) -> None:
    for key in ("type", "evicted_text", "noise_text", "query_text", "full_text", "answer_text"):
        if key not in sample:
            raise ValueError(f"Missing field: {key}")
    t = sample["type"]
    if t not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid type: {t}")
    if not isinstance(sample["evicted_text"], str) or not isinstance(sample["noise_text"], str) or not isinstance(sample["query_text"], str) or not isinstance(sample["full_text"], str):
        raise ValueError("Text fields must be strings")
    if t == "D":
        if sample["answer_text"] is not None:
            raise ValueError("Type D must have answer_text = null")
    else:
        if not isinstance(sample["answer_text"], str) or not sample["answer_text"].strip():
            raise ValueError("Types A/B/C must have non-empty string answer_text")
    if tokenizer and tokenizer.available():
        total_tokens = tokenizer.count_tokens(sample["full_text"])
        if total_tokens > MAX_FULL_TEXT_TOKENS:
            raise ValueError("full_text exceeds MAX_FULL_TEXT_TOKENS")
    else:
        if len(sample["full_text"]) > MAX_FULL_TEXT_CHARS:
            raise ValueError("full_text exceeds MAX_FULL_TEXT_CHARS")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    global EVICTED_MIN_CHARS, EVICTED_MAX_CHARS
    global NOISE_MIN_CHARS, NOISE_MAX_CHARS
    global QUERY_MIN_CHARS, QUERY_MAX_CHARS
    global MAX_FULL_TEXT_CHARS
    parser = argparse.ArgumentParser(description="Generate compressor pretraining dataset JSONL")
    parser.add_argument("--num_A", type=int, default=DEFAULT_NUM_A)
    parser.add_argument("--num_B", type=int, default=DEFAULT_NUM_B)
    parser.add_argument("--num_C", type=int, default=DEFAULT_NUM_C)
    parser.add_argument("--num_D", type=int, default=DEFAULT_NUM_D)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--use_loong_c4", type=str, default=str(USE_LOONG_C4_DEFAULT))
    parser.add_argument("--sink_n", type=int, default=SINK_N_DEFAULT)
    parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    # Tokenizer precise control
    parser.add_argument("--use_tokenizer", type=str, default=str(USE_TOKENIZER_DEFAULT))
    parser.add_argument("--tokenizer_name", type=str, default=TOKENIZER_NAME_DEFAULT)
    # Length budget controls (characters)
    parser.add_argument("--evicted_min_chars", type=int, default=EVICTED_MIN_CHARS)
    parser.add_argument("--evicted_max_chars", type=int, default=EVICTED_MAX_CHARS)
    parser.add_argument("--noise_min_chars", type=int, default=NOISE_MIN_CHARS)
    parser.add_argument("--noise_max_chars", type=int, default=NOISE_MAX_CHARS)
    parser.add_argument("--query_min_chars", type=int, default=QUERY_MIN_CHARS)
    parser.add_argument("--query_max_chars", type=int, default=QUERY_MAX_CHARS)
    parser.add_argument("--max_full_text_chars", type=int, default=MAX_FULL_TEXT_CHARS)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))

    args = parser.parse_args()
    random.seed(args.seed)

    use_loong_c4 = str2bool(args.use_loong_c4)
    sink_n = int(args.sink_n)
    use_tokenizer = str2bool(args.use_tokenizer)
    tokenizer_name = str(args.tokenizer_name)

    EVICTED_MIN_CHARS = int(args.evicted_min_chars)
    EVICTED_MAX_CHARS = int(args.evicted_max_chars)
    NOISE_MIN_CHARS = int(args.noise_min_chars)
    NOISE_MAX_CHARS = int(args.noise_max_chars)
    QUERY_MIN_CHARS = int(args.query_min_chars)
    QUERY_MAX_CHARS = int(args.query_max_chars)
    MAX_FULL_TEXT_CHARS = int(args.max_full_text_chars)

    tokenizer = TokenizerHelper(tokenizer_name) if use_tokenizer else None
    sampler = LoongC4Sampler(use_loong_c4, tokenizer=tokenizer)
    if use_loong_c4 and (sampler._failed or not sampler._loaded):
        # Fallback to pseudo-natural text
        use_loong_c4 = False

    total_counts = {
        "A": int(args.num_A),
        "B": int(args.num_B),
        "C": int(args.num_C),
        "D": int(args.num_D),
    }

    # Generate samples for each type with streaming writes
    def _iterate(n: int, desc: str):
        if _TQDM is not None:
            return _TQDM(range(n), desc=desc)
        return range(n)

    # Streaming split setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_ratio = max(0.0, min(1.0, float(args.valid_ratio)))
    n_total = total_counts["A"] + total_counts["B"] + total_counts["C"] + total_counts["D"]
    n_valid = int(n_total * valid_ratio)
    valid_indices: set = set(random.sample(range(n_total), n_valid)) if n_valid > 0 else set()

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    train_fh = train_path.open("w", encoding="utf-8")
    valid_fh = valid_path.open("w", encoding="utf-8") if n_valid > 0 else None

    workers = max(1, int(args.workers))

    idx = 0
    n_train_written = 0
    n_valid_written = 0

    executor = ThreadPoolExecutor(max_workers=workers)

    try:
        def _dispatch(kind: str, count: int):
            nonlocal idx, n_train_written, n_valid_written
            if count <= 0:
                return
            if kind == "A":
                gen_fn = generate_sample_A
            elif kind == "B":
                gen_fn = generate_sample_B
            elif kind == "C":
                gen_fn = generate_sample_C
            else:
                gen_fn = generate_sample_D

            pbar = _TQDM(total=count, desc=f"Generating {kind}") if _TQDM is not None else None
            for s in executor.map(lambda _: gen_fn(sampler, tokenizer), range(count), chunksize=max(1, workers*2)):
                validate_sample(s, tokenizer=tokenizer)
                if idx in valid_indices and valid_fh is not None:
                    write_jsonl_line(valid_fh, s)
                    n_valid_written += 1
                else:
                    write_jsonl_line(train_fh, s)
                    n_train_written += 1
                idx += 1
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

        _dispatch("A", total_counts["A"])
        _dispatch("B", total_counts["B"])
        _dispatch("C", total_counts["C"])
        _dispatch("D", total_counts["D"])
    finally:
        try:
            executor.shutdown(wait=True)
        except Exception:
            pass
        try:
            train_fh.close()
        except Exception:
            pass
        if valid_fh is not None:
            try:
                valid_fh.close()
            except Exception:
                pass

    # Final console messages
    print(f"Generated {n_train_written} train samples and {n_valid_written} validation samples in {output_dir}.")
    note = f"Note: training pipeline will prepend [sink_token] * {sink_n} automatically; do not include sink tokens in full_text."
    if tokenizer and tokenizer.available():
        note += f" Using tokenizer: {tokenizer_name}."
    print(note)


if __name__ == "__main__":
    main()


# ---------------------------------
# HuggingFace datasets usage example
# ---------------------------------
# from datasets import load_dataset
#
# dataset = load_dataset(
#     "json",
#     data_files={
#         "train": "train.jsonl",
#         "validation": "valid.jsonl"
#     }
# )
#
# print(dataset)
# print(dataset["train"][0]["full_text"])  # Inspect the combined text


# ---------------------------------
# CLI invocation example
# ---------------------------------
# python generate_compressor_dataset.py \
#   --num_A 40000 --num_B 30000 --num_C 20000 --num_D 10000 \
#   --output_dir ./data \
#   --use_loong_c4 true \
#   --sink_n 32 \
#   --valid_ratio 0.1


# ---------------------------------
# Example JSON lines (illustrative only)
# ---------------------------------
# {"type":"A","evicted_text":"ID_123 = BLUE\nKEY_812 = LONDON\n...","noise_text":"<natural text>","query_text":"Question: What is the value of KEY_812?\nAnswer:","full_text":"<evicted>\n\n<noise>\n\n<query>","answer_text":"LONDON"}
# {"type":"B","evicted_text":"... The secret code is 529341. ...","noise_text":"<natural text>","query_text":"What is the secret code mentioned earlier?\nAnswer:","full_text":"...","answer_text":"529341"}
# {"type":"D","evicted_text":"","noise_text":"<natural text>","query_text":"","full_text":"<same as noise_text>","answer_text":null}
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None
