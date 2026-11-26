"""Task 5: the ablation study behind the "Homework 4 solution" table.

Each configuration changes one component relative to the row above it and is
measured on public_test with the same fixed generator and the same scoring code
as hw4.py.

    python3 ablation.py --data-folder public_test

Writes ablation.csv (one row per configuration) and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4
from hw4 import MAX_RETRIEVED, Document

CONFIGS = [
    ("B0", dict(table=False, sections=False, tokens=False, parse=False, year=False,
                roundrobin=False, abstain=False, bonus=False)),
    ("V1", dict(table=True, sections=False, tokens=False, parse=False, year=False,
                roundrobin=False, abstain=False, bonus=False)),
    ("V2", dict(table=True, sections=True, tokens=False, parse=False, year=False,
                roundrobin=False, abstain=False, bonus=False)),
    ("V3", dict(table=True, sections=True, tokens=True, parse=False, year=False,
                roundrobin=False, abstain=False, bonus=False)),
    ("V4", dict(table=True, sections=True, tokens=True, parse=True, year=False,
                roundrobin=False, abstain=False, bonus=False)),
    ("V5", dict(table=True, sections=True, tokens=True, parse=True, year=True,
                roundrobin=False, abstain=False, bonus=False)),
    ("V6", dict(table=True, sections=True, tokens=True, parse=True, year=True,
                roundrobin=True, abstain=False, bonus=False)),
    ("V7", dict(table=True, sections=True, tokens=True, parse=True, year=True,
                roundrobin=True, abstain=True, bonus=False)),
    ("V8", dict(table=True, sections=True, tokens=True, parse=True, year=True,
                roundrobin=True, abstain=True, bonus=True)),
]
CHANGE = {
    "B0": "naive baseline: get_text(), RecursiveCharacterTextSplitter(1000/100), required "
          "metadata only, BM25 top-5 over the whole corpus, no query parsing, never abstains",
    "V1": "table_to_text() renders every table as 'label | column: value' lines",
    "V2": "section-aware chunking, each chunk carries a company/doctype/FY/section header",
    "V3": "tokeniser: numbers lose thousands separators, plurals stemmed, FY2024 -> 2024",
    "V4": "parse_query() metadata filter: only the named company and document type",
    "V5": "fiscal-year window: an annual report also covers the previous year",
    "V6": "comparison questions retrieve per company in round-robin",
    "V7": "abstention: [] when the company is absent, the year is uncovered, or the figure is never named",
    "V8": "table-row bonus: a row naming the asked figure and the year is boosted",
}

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def plain_tokens(text: str) -> list[str]:
    """The naive tokeniser: lower-cased whitespace tokens."""
    return str(text).lower().split()


def build_chunks(corpus: dict[str, str], cfg: dict) -> list[Document]:
    """Chunks for one configuration."""
    if cfg["sections"]:
        return [chunk for file_name, html in corpus.items() for chunk in hw4.chunk_document(file_name, html)]
    chunks: list[Document] = []
    for file_name, html in corpus.items():
        meta = hw4.document_meta(file_name, html)
        soup = BeautifulSoup(html, "lxml")
        if cfg["table"]:
            for table in soup.find_all("table"):
                rendered = hw4.table_to_text(str(table))
                table.replace_with(BeautifulSoup(f"<p>{rendered}</p>", "lxml").get_text())
        for piece in SPLITTER.split_text(soup.get_text("\n", strip=True)):
            chunks.append(Document(page_content=piece, metadata=dict(meta)))
    return chunks


def build_index(chunks: list[Document], tokenizer) -> dict:
    groups: dict[tuple[str, str, int], list[int]] = {}
    text_by_company: dict[str, list[str]] = {}
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        company = hw4.strip_legal(meta.get("company", ""))
        groups.setdefault((company, meta.get("doc_type"), int(meta.get("fiscal_year", 0) or 0)), []).append(i)
        text_by_company.setdefault(company, []).append(chunk.page_content)
    return {"chunks": chunks, "groups": groups,
            "bm25": BM25Okapi([tokenizer(c.page_content) for c in chunks]),
            "company_text": {k: " ".join(v) for k, v in text_by_company.items()}}


def retrieve_cfg(index: dict, question: str, cfg: dict):
    tokenizer = hw4.tokens if cfg["tokens"] else plain_tokens
    chunks, groups, company_text = index["chunks"], index["groups"], index["company_text"]
    scores = list(index["bm25"].get_scores(tokenizer(question)))

    if not cfg["parse"]:                            # B0..V3: one ranking over the whole corpus
        return [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:MAX_RETRIEVED]]

    parsed = hw4.parse_query(question)
    named = [hw4.strip_legal(c) for c in parsed["companies"]]
    known = [c for c in named if c in company_text]
    if cfg["abstain"] and named and not known:
        return []
    companies = known or list(company_text)
    doc_type = parsed["doc_type"]
    year = parsed["fiscal_year"] if cfg["year"] else None
    back, ahead = hw4.YEAR_WINDOW[doc_type]
    candidates = [i for (company, kind, doc_year), indices in groups.items()
                  if company in companies and kind == doc_type
                  and (year is None or doc_year - back <= year <= doc_year + ahead)
                  for i in indices]
    if not candidates:
        return []
    if cfg["abstain"] and len(companies) == 1 and \
            not hw4.disclosed(question, companies[0], company_text[companies[0]]):
        return []

    if cfg["bonus"]:
        name_words = {w for company in companies for w in tokenizer(company)}
        metric = [w for w in hw4.asked_terms(question)
                  if w not in hw4.PHRASING and w not in hw4.GENERIC and w not in name_words][:4]
        if metric:
            for i in candidates:
                for line in chunks[i].page_content.lower().splitlines():
                    if all(word in line for word in metric) and (year is None or str(year) in line):
                        scores[i] += 3.0
                        break

    if not cfg["roundrobin"]:
        return [chunks[i] for i in sorted(candidates, key=lambda i: -scores[i])[:MAX_RETRIEVED]]

    ranked = {company: sorted((i for i in candidates
                               if hw4.strip_legal((chunks[i].metadata or {}).get("company", "")) == company),
                              key=lambda i: -scores[i])
              for company in companies}
    chosen: list[int] = []
    for round_number in range(MAX_RETRIEVED):
        for company in companies:
            if round_number < len(ranked[company]) and len(chosen) < MAX_RETRIEVED:
                chosen.append(ranked[company][round_number])
    return [chunks[i] for i in chosen]


def measure(corpus, questions, gold, cfg) -> dict:
    chunks = build_chunks(corpus, cfg)
    index = build_index(chunks, hw4.tokens if cfg["tokens"] else plain_tokens)

    def one(qid: str):
        got = retrieve_cfg(index, questions[qid], cfg)
        return got, hw4.generate(questions[qid], got)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(questions, pool.map(one, questions)))
    answerable = [q for q in questions if gold[q]["evidence"]]
    unanswerable = [q for q in questions if not gold[q]["evidence"]]
    hits = sum(hw4.retrieval_hit(results[q][0], gold[q]) for q in answerable)
    kept = sum(bool(results[q][0]) for q in answerable)
    rejected = sum(not results[q][0] for q in unanswerable)
    correct = sum(hw4.answer_correct(results[q][1], gold[q]) for q in questions)
    recall = hits / max(len(answerable), 1)
    kept_rate = kept / max(len(answerable), 1)
    rejected_rate = rejected / max(len(unanswerable), 1)
    accuracy = correct / max(len(questions), 1)
    points = {"2.1": 5 * kept_rate,
              "2.3": 15 * max(0.0, min(1.0, (recall - 0.5) / 0.4)),
              "3": 10 * max(0.0, kept_rate + rejected_rate - 1),
              "4": 20 * accuracy}
    return {"recall": recall, "kept": kept_rate, "rejected": rejected_rate, "accuracy": accuracy,
            "points": points, "total": sum(points.values()), "chunks": len(chunks)}


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
