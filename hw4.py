"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

import math
from collections import Counter

from bs4 import BeautifulSoup

# Filled while chunk_document runs, so parse_query can see the corpus.
CORPUS_COMPANIES: dict[str, str] = {}   # company key -> company name as printed
CORPUS_BROKERS: set[str] = set()
CORPUS_LABELS: set[str] = set()         # table row labels: region and segment names
QUESTION_WORDS = {"what", "which", "how", "when", "where", "who", "why", "whom", "whose"}

DOC_MARKERS = {
    "annual_report": "annual report",
    "earnings_call": "earnings conference call",
    "research_report": "results review",
}
BROKER_SUFFIXES = ("securities", "capital", "research", "bank", "partners", "equity")
NOT_A_COMPANY = {
    "annual report", "earnings call", "earnings conference call", "conference call",
    "fourth quarter", "full year", "chief financial officer", "chief executive officer",
    "chief operating officer", "general counsel", "chair of the board", "board of directors",
    "audit committee", "independent non-executive director", "second quarter", "third quarter",
    "first quarter", "united states", "hong kong", "new york",
}
STOPWORDS = set("""a an the of and or in on at to for from by with as is are was were be been
this that these those it its their they them we our us you your i he she his her not no
than then so such which who whom whose what when where why how much many did do does done
have has had will would should could can may might must also more most other others same
between both each any all some if but about into over under during end year years fy
company company's group please give tell according report reports question answer""".split())


def collapse(text: str) -> str:
    """One line, single spaces."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_legal(name: str) -> str:
    """'Northgate Robotics Inc.' -> 'northgate robotics' (case, punctuation and legal suffix removed)."""
    text = unicodedata.normalize("NFKC", name).lower().replace("&", " and ")
    words = re.sub(r"[^a-z0-9]+", " ", text).split()
    return " ".join(w for w in words if w not in {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"})


def tokens(text: str) -> list[str]:
    """Lower-case word tokens, numbers without thousands separators, light plural stripping."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\bfy\s?(\d{4})\b", r"\1", text)      # FY2024 and 2024 are the same year
    out = []
    for raw in re.split(r"[^a-z0-9.]+", text):
        raw = raw.strip(".")
        if len(raw) < 2 or raw in STOPWORDS:
            continue
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        out.append(raw)
    return out


def cell_text(cell) -> str:
    return collapse(cell.get_text(" ", strip=True))


def table_to_text(table_html: str) -> str:
    """Task 1.1: one line per table row: row label, then 'column header: value' pairs, then the caption."""
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table") or soup
    lines = []
    caption = table.find("caption")
    if caption is not None:
        lines.append(cell_text(caption))
    heads = [cell_text(th) for th in table.select("thead th")]
    row_head = heads[0].lower() if heads else ""
    headers = heads[1:] if row_head in ("", "name", "company", "segment", "region", "function") else heads
    for tr in table.select("tbody tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        if not cells:
            continue
        label = cell_text(cells[0])
        values = [cell_text(c) for c in cells[1:]]
        if not label:
            continue
        if not values:                      # a section row such as "Current assets"
            lines.append(label)
            continue
        pairs = [f"{h}: {v}" for h, v in zip(headers, values) if h and v]
        lines.append(f"{label} | " + " | ".join(pairs) if pairs else label)
    # A row can get long; keep every line inside the 400-character limit by splitting on the pairs.
    out = []
    for line in lines:
        while len(line) > 390:
            cut = line.rfind(" | ", 0, 390)
            if cut <= 0:
                break
            out.append(line[:cut])
            head, _, rest = line.partition(" | ")
            line = f"{head} | {rest[cut + 3:]}"
        out.append(line)
    return "\n".join(out)


def document_meta(file_name: str, html: str) -> dict:
    """doc_type, company and fiscal year, all read from the first lines of the document."""
    soup = BeautifulSoup(html, "lxml")
    h1 = collapse(soup.find("h1").get_text(" ", strip=True)) if soup.find("h1") else ""
    text = collapse(soup.get_text(" ", strip=True))
    company = re.sub(r"\s*\([^()]*\)\s*$", "", h1).strip()
    head = collapse(text[:400]).lower()
    if "earnings conference call" in head or "earnings call" in head:
        doc_type = "earnings_call"
        match = re.search(r"full year (\d{4})", text, re.I)
    elif "annual report" in head:
        doc_type = "annual_report"
        match = re.search(r"annual report (\d{4})", text, re.I) or re.search(r"year ended 31 december (\d{4})", text, re.I)
    else:
        doc_type = "research_report"
        match = re.search(r"FY(\d{4}) results review", text)
    fiscal_year = int(match.group(1)) if match else 2025
    return {"source": file_name, "company": company, "fiscal_year": fiscal_year, "doc_type": doc_type}


def document_blocks(html: str) -> list[tuple[str, str]]:
    """(section title, text) blocks in document order; tables go through table_to_text."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body") or soup
    blocks: list[tuple[str, str]] = []
    section = ""
    for el in body.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"]):
        if any(a.name in {"h1", "h2", "h3", "p", "ul", "ol", "table"} for a in el.parents if a is not body):
            continue
        if el.name in ("h1", "h2", "h3"):
            title = cell_text(el)
            if el.name == "h1":
                continue
            section = title
            continue
        if el.name == "table":
            text = table_to_text(str(el))
            for tr in el.select("tbody tr"):          # remember row labels: regions, segments, line items
                cells = tr.find_all(["td", "th"], recursive=False)
                if cells:
                    CORPUS_LABELS.add(strip_legal(cell_text(cells[0])))
        elif el.name in ("ul", "ol"):
            text = "\n".join("- " + cell_text(li) for li in el.find_all("li"))
        else:
            text = cell_text(el)
        text = text.strip()
        if text:
            blocks.append((section, text))
    return blocks


def split_text(text: str, room: int) -> list[str]:
    """Split into pieces of at most *room* characters without breaking a line (a table row)."""
    if len(text) <= room:
        return [text]
    lines = text.split("\n")
    if len(lines) == 1:
        lines = text.split(" ")
        joiner = " "
    else:
        joiner = "\n"
    pieces: list[str] = []
    current = ""
    for line in lines:
        if current and len(current) + len(line) + 1 > room:
            pieces.append(current.rstrip())
            current = ""
        current += line + joiner
    if current.strip():
        pieces.append(current.rstrip())
    return pieces


def pack_chunks(blocks: list[tuple[str, str]], meta: dict) -> list[Document]:
    """Group blocks into chunks under MAX_CHUNK_CHARS, each with a metadata header line."""
    company, year, doc_type = meta["company"], meta["fiscal_year"], meta["doc_type"]
    chunks: list[Document] = []
    buf: list[str] = []
    used = 0
    section = ""

    def flush():
        nonlocal buf, used
        if buf:
            header = f"{company} | {doc_type.replace('_', ' ')} FY{year} | {section or 'document'}"
            body = "\n".join(buf)[:MAX_CHUNK_CHARS - len(header) - 1]
            chunks.append(Document(page_content=header + "\n" + body, metadata=dict(meta)))
            buf, used = [], 0

    for block_section, text in blocks:
        if block_section and block_section != section:
            flush()                      # the buffered blocks belong to the previous section
            section = block_section
        room = max(MAX_CHUNK_CHARS - (len(company) + len(section) + 40), 300)
        for piece in split_text(text, room):
            if used + len(piece) + 1 > room and buf:
                flush()
            buf.append(piece)
            used += len(piece) + 1
    flush()
    return chunks


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    ### YOUR CODE HERE
    meta = document_meta(file_name, html)
    chunks = pack_chunks(document_blocks(html), meta)
    if meta["company"]:
        CORPUS_COMPANIES.setdefault(strip_legal(meta["company"]), meta["company"])
    if meta["doc_type"] == "research_report":
        first = BeautifulSoup(html, "lxml").get_text(" ", strip=True)[:200]
        broker = first.split("|")[0].strip()
        if broker:
            CORPUS_BROKERS.add(strip_legal(broker))
    return chunks


CAPITAL_RUN = re.compile(r"\b([A-Z][\w&'\-]*(?:\s+(?:of|and|&|the|[A-Z][\w&'\-]*)){1,4})")


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    ### YOUR CODE HERE
    text = collapse(question)
    flat = strip_legal(text)
    companies: list[str] = []
    for key, printed in sorted(CORPUS_COMPANIES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", flat):
            companies.append(printed)
    taken = strip_legal(" ".join(companies))
    for match in CAPITAL_RUN.finditer(text):
        run = match.group(1).replace("'s", "").strip()
        words = run.split()
        while words and words[-1].lower() in ("of", "and", "the", "&"):
            words.pop()
        run = " ".join(words)
        key = strip_legal(run)
        if len(words) < 2 or not key or key in taken or key in CORPUS_BROKERS:
            continue
        if key in NOT_A_COMPANY or key in CORPUS_LABELS or words[-1].lower() in BROKER_SUFFIXES:
            continue
        if any(key in label for label in CORPUS_LABELS):      # "Middle East and Africa" is a table row
            continue
        if words[0].lower() in QUESTION_WORDS:
            continue
        if any(key in strip_legal(c) or strip_legal(c) in key for c in companies):
            continue
        if any(word.lower() in ("fy", "fy20", "q1", "q2", "q3", "q4") for word in words):
            continue
        companies.append(run)
        taken += " " + key

    year = None
    for pattern in (r"FY\s?(\d{4})", r"\b(20\d{2})\b"):
        found = re.search(pattern, text)
        if found:
            year = int(found.group(1))
            break

    lower = text.lower()
    if "earnings call" in lower or "conference call" in lower or "call transcript" in lower:
        doc_type = "earnings_call"
    elif re.search(r"\b(rating|target price|price target|forecast|estimate|estimates|"
                   r"results review|broker|recommendation|analyst)\b", lower):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    return {"companies": companies, "fiscal_year": year, "doc_type": doc_type}


class BM25:
    """Plain BM25 over token lists."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = documents
        self.k1, self.b = k1, b
        self.n = len(documents)
        self.avgdl = sum(len(d) for d in documents) / max(self.n, 1)
        self.freqs = [Counter(d) for d in documents]
        df: Counter = Counter()
        for freq in self.freqs:
            df.update(freq.keys())
        self.idf = {t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.freqs):
                tf = freq.get(term, 0)
                if not tf:
                    continue
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / max(self.avgdl, 1e-9)))
                out[i] += idf * norm
        return out


# A document of type T with fiscal year Y covers the years [Y - back, Y + ahead].
# The FY2025 annual report also shows FY2024, and a FY2025 results review carries FY2024A to FY2027E.
YEAR_WINDOW = {"annual_report": (1, 0), "earnings_call": (0, 1), "research_report": (1, 2)}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    ### YOUR CODE HERE
    chunks = [c for c in chunks if isinstance(c, Document) and c.page_content]
    texts = [tokens(c.page_content) for c in chunks]
    groups: dict[tuple[str, str, int], list[int]] = {}
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        key = (strip_legal(meta.get("company", "")), str(meta.get("doc_type", "")),
               int(meta.get("fiscal_year", 0) or 0))
        groups.setdefault(key, []).append(i)
    return {"chunks": chunks, "bm25": BM25(texts), "groups": groups,
            "company_text": _company_text(chunks)}


def _company_text(chunks: list[Document]) -> dict[str, str]:
    """All text of each company, lower-cased, for the 'is this figure disclosed' check."""
    out: dict[str, list[str]] = {}
    for chunk in chunks:
        key = strip_legal((chunk.metadata or {}).get("company", ""))
        out.setdefault(key, []).append(chunk.page_content)
    return {key: " ".join(parts).lower() for key, parts in out.items()}


ASKED_BEFORE = re.compile(r"\b(?:what|which|how much|how many)\b.*?\b(?:did|does|do|was|were|is|are|had|has|have)?\b", re.I)
GENERIC = {"company", "reported", "report", "reports", "fy", "year", "years", "end", "hold", "held"}
# Words that only phrase the question; they are not the name of a figure.
PHRASING = {"value", "city", "generate", "earn", "spend", "hold", "held", "announce", "guide",
            "large", "largest", "much", "many", "firm", "business", "audit", "audited"}


def _asked_terms(question: str) -> list[str]:
    """The normalised words that name the figure being asked for, e.g. 'lobbying expense'."""
    text = collapse(question)
    text = re.sub(r"\b(?:at|on|in|for|during)\s+(?:the\s+)?(?:end\s+of\s+)?(?:31\s+december\s+|fy\s?)?\d{4}\b", " ", text, flags=re.I)
    text = re.sub(r"\bin (?:usd|percent|us dollars?).*$", " ", text, flags=re.I)
    return tokens(text)


def _disclosed(question: str, company: str, company_tokens: str) -> bool:
    """False when the asked figure is named by words the company's documents never use together.

    'lobbying expenses' or 'customer churn rate' never appear in the corpus, so it cannot answer
    them; 'accounts receivable' does appear, so it can.
    """
    name_words = set(tokens(company))
    terms = [w for w in _asked_terms(question) if w not in GENERIC and w not in PHRASING and w not in name_words]
    pairs = [f"{a} {b}" for a, b in zip(terms, terms[1:])]
    if not pairs:
        return True
    haystack = " " + " ".join(tokens(company_tokens)) + " "
    return any(f" {pair} " in haystack for pair in pairs)


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    parsed = parse_query(question)
    chunks = index["chunks"]
    groups = index["groups"]
    named = [strip_legal(c) for c in parsed["companies"]]
    known = [c for c in named if any(key[0] == c for key in groups)]
    if named and not known:
        return []                                   # the company is not in the corpus
    companies = known or sorted({key[0] for key in groups})
    year, doc_type = parsed["fiscal_year"], parsed["doc_type"]

    candidates: list[int] = []
    back, ahead = YEAR_WINDOW[doc_type]
    for company in companies:
        for (group_company, group_type, doc_year), indices in groups.items():
            if group_company != company or group_type != doc_type:
                continue
            if year is not None and not (doc_year - back <= year <= doc_year + ahead):
                continue
            candidates.extend(indices)
    if not candidates:
        return []                                   # the year is not covered by the corpus
    if len(companies) == 1 and not _disclosed(question, companies[0], index["company_text"].get(companies[0], "")):
        return []                                   # the figure is not disclosed

    scores = index["bm25"].scores(tokens(question))
    name_words: set[str] = set()
    for name in companies:
        name_words.update(tokens(name))
    metric = [w for w in _asked_terms(question)
              if w not in PHRASING and w not in GENERIC and w not in name_words][:4]
    for i in candidates:                   # a table row that names the figure and the year wins
        if not metric:
            break
        for line in chunks[i].page_content.lower().splitlines():
            if all(word in line for word in metric) and (year is None or str(year) in line):
                scores[i] += 3.0
                break
    ranked: dict[str, list[int]] = {}
    for company in companies:
        own = [i for i in candidates if strip_legal((chunks[i].metadata or {}).get("company", "")) == company]
        ranked[company] = sorted(own, key=lambda i: -scores[i])
    chosen: list[int] = []
    for round_number in range(MAX_RETRIEVED):                 # round-robin, so a comparison gets both sides
        for company in companies:
            if round_number < len(ranked[company]) and len(chosen) < MAX_RETRIEVED:
                chosen.append(ranked[company][round_number])
    return [chunks[i] for i in chosen[:MAX_RETRIEVED]]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
