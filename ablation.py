"""Task 5: the ablation study behind the "Homework 4 solution" table.

Each configuration changes one component relative to the row above it and is
measured on public_test with the same fixed generator and scoring as hw4.py.

    python3 ablation.py --data-folder public_test

Writes ablation.csv (one row per configuration) and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup

import hw4
from hw4 import Document, MAX_RETRIEVED


# --------------------------------------------------------------------------- configuration
CONFIGS = [
    ("B0", dict(table=False, sections=False, parse=False, year=False, roundrobin=False,
                abstain=False, bonus=False)),
    ("V1", dict(table=True, sections=False, parse=False, year=False, roundrobin=False,
                abstain=False, bonus=False)),
    ("V2", dict(table=True, sections=True, parse=False, year=False, roundrobin=False,
                abstain=False, bonus=False)),
    ("V3", dict(table=True, sections=True, parse=True, year=False, roundrobin=False,
                abstain=False, bonus=False)),
    ("V4", dict(table=True, sections=True, parse=True, year=True, roundrobin=False,
                abstain=False, bonus=False)),
    ("V5", dict(table=True, sections=True, parse=True, year=True, roundrobin=True,
                abstain=False, bonus=False)),
    ("V6", dict(table=True, sections=True, parse=True, year=True, roundrobin=True,
                abstain=True, bonus=False)),
    ("V7", dict(table=True, sections=True, parse=True, year=True, roundrobin=True,
                abstain=True, bonus=True)),
]
CHANGE = {
    "B0": "naive baseline: BeautifulSoup get_text(), fixed 1000-character splits, "
          "required metadata only, BM25 top-5 over the whole corpus, no query parsing, never abstains",
    "V1": "table_to_text() renders every table as 'label | column: value' lines",
    "V2": "section-aware chunking, each chunk carries a company/doctype/FY/section header",
    "V3": "parse_query() metadata filter: only chunks of the named company and document type",
    "V4": "fiscal-year window: an annual report also covers the previous year",
    "V5": "comparison questions retrieve per company in round-robin instead of one ranking",
    "V6": "abstention: [] when the company is absent, the year is not covered, or the figure is never named",
    "V7": "table-row bonus: a row naming the asked figure and the year is boosted",
}


def fixed_split(text: str, size: int = 1000, overlap: int = 100) -> list[str]:
    """A plain fixed-size splitter with overlap (stands in for RecursiveCharacterTextSplitter)."""
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return [p for p in pieces if p.strip()]


def build_chunks(corpus: dict[str, str], cfg: dict) -> list[Document]:
    """Chunks for one configuration: naive splitting, or hw4.chunk_document when sections are on."""
    chunks: list[Document] = []
    for file_name, html in corpus.items():
        meta = hw4.document_meta(file_name, html)
        if cfg["sections"]:
            chunks.extend(hw4.chunk_document(file_name, html))
            continue
        soup = BeautifulSoup(html, "lxml")
        if cfg["table"]:
            for table in soup.find_all("table"):
                rendered = hw4.table_to_text(str(table))
                table.replace_with(BeautifulSoup(f"<p>{rendered}</p>", "lxml").get_text())
        text = soup.get_text("\n", strip=True)
        for piece in fixed_split(text):
            chunks.append(Document(page_content=piece, metadata=dict(meta)))
    return chunks


def retrieve_cfg(index, question: str, cfg: dict) -> list[Document]:
    """Retrieval for one configuration, mirroring hw4.retrieve when every flag is on."""
    chunks = index["chunks"]
    scores = index["bm25"].scores(hw4.tokens(question))
    if not cfg["parse"]:
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        return [chunks[i] for i in order[:MAX_RETRIEVED]]

    parsed = hw4.parse_query(question)
    named = [hw4.strip_legal(c) for c in parsed["companies"]]
    known = [c for c in named if any(key[0] == c for key in index["groups"])]
    if cfg["abstain"] and named and not known:
        return []
    companies = known or sorted({key[0] for key in index["groups"]})
    doc_type = parsed["doc_type"]
    year = parsed["fiscal_year"] if cfg["year"] else None

    candidates: list[int] = []
    back, ahead = hw4.YEAR_WINDOW[doc_type]
    for company in companies:
        for (group_company, group_type, doc_year), indices in index["groups"].items():
            if group_company != company or group_type != doc_type:
                continue
            if year is not None and not (doc_year - back <= year <= doc_year + ahead):
                continue
            candidates.extend(indices)
    if not candidates:
        return []
    if cfg["abstain"] and len(companies) == 1 and \
            not hw4._disclosed(question, companies[0], index["company_text"].get(companies[0], "")):
        return []

    if cfg["bonus"]:
        name_words: set[str] = set()
        for name in companies:
            name_words.update(hw4.tokens(name))
        metric = [w for w in hw4._asked_terms(question)
                  if w not in hw4.PHRASING and w not in hw4.GENERIC and w not in name_words][:4]
        if metric:
            for i in candidates:
                for line in chunks[i].page_content.lower().splitlines():
                    if all(word in line for word in metric) and (year is None or str(year) in line):
                        scores[i] += 3.0
                        break

    if cfg["roundrobin"]:
        chosen: list[int] = []
        ranked = {}
        for company in companies:
            own = [i for i in candidates if hw4.strip_legal((chunks[i].metadata or {}).get("company", "")) == company]
            ranked[company] = sorted(own, key=lambda i: -scores[i])
        for round_number in range(MAX_RETRIEVED):
            for company in companies:
                if round_number < len(ranked[company]) and len(chosen) < MAX_RETRIEVED:
                    chosen.append(ranked[company][round_number])
        return [chunks[i] for i in chosen[:MAX_RETRIEVED]]

    order = sorted(candidates, key=lambda i: -scores[i])
    return [chunks[i] for i in order[:MAX_RETRIEVED]]


def measure(corpus, questions, gold, cfg) -> dict:
    chunks = build_chunks(corpus, cfg)
    index = hw4.build_index(chunks)
    hits, kept, rejected, correct = 0, 0, 0, 0
    answerable = [q for q in questions if gold[q]["evidence"]]
    unanswerable = [q for q in questions if not gold[q]["evidence"]]

    def one(qid):
        got = retrieve_cfg(index, questions[qid], cfg)
        answer = hw4.generate(questions[qid], got)
        return got, answer

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(questions, pool.map(one, questions)))
    for qid in questions:
        got, answer = results[qid]
        if gold[qid]["evidence"]:
            hits += hw4.retrieval_hit(got, gold[qid])
            kept += bool(got)
        else:
            rejected += not got
        correct += hw4.answer_correct(answer, gold[qid])
    recall = hits / max(len(answerable), 1)
    kept_rate = kept / max(len(answerable), 1)
    rejected_rate = rejected / max(len(unanswerable), 1)
    accuracy = correct / max(len(questions), 1)
    points = {
        "2.1": 5 * kept_rate,
        "2.3": 15 * max(0.0, min(1.0, (recall - 0.5) / 0.4)),
        "3": 10 * max(0.0, kept_rate + rejected_rate - 1),
        "4": 20 * accuracy,
    }
    return {"recall": recall, "kept": kept_rate, "rejected": rejected_rate,
            "accuracy": accuracy, "points": points, "total": sum(points.values()),
            "chunks": len(chunks)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output", default="ablation.csv")
    args = parser.parse_args()
    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    gold = hw4.load_json(folder / "ground_truth.json")

    rows = []
    for name, cfg in CONFIGS:
        m = measure(corpus, questions, gold, cfg)
        rows.append({"config": name, "change": CHANGE[name], "chunks": m["chunks"],
                     "recall": round(m["recall"], 3), "kept": round(m["kept"], 3),
                     "rejected": round(m["rejected"], 3), "accuracy": round(m["accuracy"], 3),
                     **{f"task_{k}": round(v, 2) for k, v in m["points"].items()},
                     "total_of_50": round(m["total"], 2)})
        print(f"{name}: recall {m['recall']:.1%} kept {m['kept']:.1%} rejected {m['rejected']:.1%} "
              f"acc {m['accuracy']:.1%} -> {m['total']:.2f}/50", flush=True)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
