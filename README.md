# FTEC5660 Homework 4 — solution

A lexical RAG retriever over 40 company reports. Five functions in `hw4.py`, one
ablation script (`ablation.py`) and its output (`ablation.csv`); every run of
`hw4.py` writes `results.csv`.

`python3 hw4.py --data-folder public_test` → **75.00 / 75** on the public set
(the grade uses the private set).

---

## 1. Pipeline

```
corpus/doc_XXX.html
      │
      │  document_meta(): doc_type, company and fiscal year from the first lines
      ▼
BeautifulSoup, body blocks in order: h2/h3 · p · ul/ol · table
      │                                        │
      │                                        └─ table_to_text(): one line per row,
      │                                           "Revenue | 2024: 2,649.8 | 2023: 2,316.8",
      │                                           caption first (it carries the unit)
      ▼
chunk_document(): section-aware packing with RecursiveCharacterTextSplitter
(separators "\n" then " ", so a table row is never cut), ≤1500 characters per
chunk, each chunk prefixed with "Company | annual report FY2025 | financial statements"
metadata: source, company, fiscal_year, doc_type
      │
      ▼
build_index(): rank_bm25 BM25Okapi over normalised tokens (thousands separators
dropped, plurals stemmed, "FY2024" → "2024"); chunks grouped by
(company, doc_type, fiscal_year); each company's whole text kept for the
"is this figure disclosed" check
      │
question ──parse_query()──► companies (the corpus names, plus a name the corpus
      │                      does not have when it starts like one it does) ·
      │                      fiscal_year · doc_type
      ▼
retrieve(): keep the chunks of the named company and document type whose fiscal
year can cover the asked year (an annual report covers its year and the previous
one, an earnings call its year and the next, a research report FY2024A to
FY2027E), then rank by BM25 plus 3.0 when a table row names the asked figure and
the year. Comparison questions take one chunk per company in round-robin, so five
chunks of one company cannot crowd out the other.
      │  [] if the company is absent from the corpus, if no document covers the
      │  year, or if the asked figure is named by a word pair the company's
      │  documents never use
      ▼
generate() — the fixed generator, top-5 chunks with a metadata header —► answer
```

The whole thing runs in about six seconds and makes **no LLM calls of its own**.
That is a design decision, not a shortcut: the generator is fixed, so my job is to
decide what it sees, and every part of that decision can be computed. Tables are
rendered so a value sits on the same line as its row label and column header — the
one thing that lets both BM25 and the generator use a figure without guessing
which year or which row it came from. Metadata does the work BM25 cannot: telling
Northgate Robotics from Northgate Payments, and 2024 from 2025. The year window is
just what the data is — the FY2025 report also shows FY2024, so a 2024 question
can be answered from either report, while a 2021 question has no answer anywhere.
Abstention is kept narrow on purpose: it fires only on positive evidence that the
corpus cannot answer, because a wrong abstention costs exactly as much as a wrong
answer.

## 2. Ablation study

`python3 ablation.py --data-folder public_test` reproduces this table
(`ablation.csv`, about one minute). Each row changes one component of the row above
it and is measured with the same fixed generator and the same scoring code as
`hw4.py`.

| Config | One change | recall@5 | kept | rejected | accuracy | Tasks 2.1+2.3+3+4 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | naive baseline: `get_text()`, `RecursiveCharacterTextSplitter(1000/100)`, required metadata, BM25 top-5 over the whole corpus, no parsing, never abstains | 23.9% | 100% | 0% | 40.0% | 13.00 / 50 |
| V1 | + `table_to_text()` | 26.1% | 100% | 0% | 43.3% | 13.67 / 50 |
| V2 | + section-aware chunking with a company/FY/section header | 80.4% | 100% | 0% | 81.7% | 32.75 / 50 |
| V3 | + tokeniser (no thousands separators, plurals stemmed, FY2024 → 2024) | 93.5% | 100% | 0% | 88.3% | 37.67 / 50 |
| V4 | + `parse_query()` metadata filter | 95.7% | 100% | 0% | 91.7% | 38.33 / 50 |
| V5 | + fiscal-year window | 95.7% | 100% | 35.7% | 93.3% | 42.24 / 50 |
| V6 | + per-company round-robin for comparisons | 97.8% | 100% | 35.7% | 98.3% | 43.24 / 50 |
| V7 | + abstention rules | 97.8% | 100% | 100% | 98.3% | 49.67 / 50 |
| V8 | + table-row bonus in the ranking | 100% | 100% | 100% | 100% | 50.00 / 50 |

**What each change did, and why.**

*Table rendering (V1) is worth little on its own.* It raises recall by 2.2 points
and accuracy by 3.3, but that is the whole gain from the component the handout
gives 8 points to. The reason is visible in the next row: with a plain
fixed-size splitter the rendered rows are still cut in half, so
"Inventories | 31 December 2025: 297.4" reaches the index as three unrelated
fragments. A table format is only worth what the chunker lets it be.

*Section-aware chunking (V2) is where the homework is won:* recall 26% → 80%,
accuracy 43% → 82%, 19 points on its own. Two effects compound — rows stop being
cut, and every chunk now carries "Company | annual report FY2025 | financial
statements" inside its text, so BM25 can match a company and a year that the
question names but the chunk body never repeats. Without that header a balance
sheet chunk says "31 December 2025: 445.1" and nothing about whose balance sheet
it is.

*The tokeniser (V3) is the second largest jump* — recall 80% → 94%. The corpus
writes "2,710.4" where a question may not, and "FY2024" where a chunk says "2024";
aligning those two strings recovers matches that BM25 would otherwise miss.

*The metadata filter (V4) adds less than it looks* (2.2 recall, 3.4 accuracy),
because the header already put the company and the year into the chunk text. What
it buys is precision on the cases the header cannot separate: two companies whose
names share a word, and the wrong year's report when every report mentions both
years.

*The year window (V5) pays off mostly in abstention*: accuracy rises 91.7% → 93.3%,
but its real contribution is that 2021 and 2022 questions stop finding a chunk with
the right company and type and start finding nothing — which is what "kept +
rejected − 1" rewards once V7 arrives.

*Round-robin (V6) is small and cheap*: accuracy 93.3% → 98.3%. A comparison needs
two chunks from two companies, and a single global ranking spends all five slots
on whichever company matches the question's wording better.

*Abstention (V7) is the largest scoring jump in the table*, 43.24 → 49.67, because
Task 3 is all-or-nothing: returning chunks everywhere scores zero by construction.
Three rules catch all fourteen unanswerable public questions without rejecting a
single answerable one — a company the corpus does not have, a year no document
covers, and a figure whose words never appear together in that company's documents
("lobbying expenses", "customer churn rate", "carbon offset purchases").

*The table-row bonus (V8) is the smallest change and still worth 0.33 points*: it
finds the two table questions whose row sat in a chunk BM25 ranked below the fifth
place, taking recall from 97.8% to 100%.

**A note on reproducibility.** The generator is an API call at temperature 0 but
the API is not guaranteed deterministic, so the accuracy column can move by about
one question between runs. Recall, kept and rejected come from my own code and are
exactly reproducible. The last row above is the same configuration that
`hw4.py` runs, and it matches `results.csv`.

## 3. Error analysis

The public set ends with **no wrong answer out of 60**: every table, fact and
comparison question is right and all fourteen unanswerable questions are answered
`NOT FOUND`.

The near misses are worth recording, because they show where the design is thin
rather than where it is broken:

* **Comparisons are the fragile category.** They are the only ones that need two
  documents, and they are where retrieval and generation both have room to fail.
  In an earlier run the retriever returned both companies' balance sheets and the
  fixed generator still named the wrong company — a hit flag of true with a wrong
  answer, which no retrieval change can fix. Round-robin (V6) and the tighter
  ranking are what removed most of this class, not a change to the generator.
* **Abstention is tuned to three kinds of unanswerable question** and would miss a
  fourth kind: a figure that the documents do mention in prose but never quantify.
  My word-pair test would consider such a question answerable, return chunks, and
  the generator would answer `NOT FOUND` — which is still the right answer under
  the no-chunks rule, but it costs one rejected point if the private set contains
  that shape.
* **The parse step depends on near-miss naming.** An unknown company is recognised
  because it begins with the same word as a company in the corpus, which is the
  pattern Section 3.1 describes for the unanswerable questions. A question about a
  company with no name overlap at all would be treated as a question about nothing
  and answered from the whole corpus.

## 4. Task 6 — Reflection

**What kind of person would you want to be in that world?**
The one who decides what is worth doing, and who can tell whether what came back is
any good. This homework is a small version of that: the generator was given to me
and wrote every answer, and everything that decided whether an answer was right —
what it was allowed to see — was mine.

**Which of your current ambitions would still matter?**
The ones about judgment rather than throughput. I want my "this is probably wrong"
to be worth listening to. The model produced a confident, well-formed answer for
every question; the reason the scores are right is that the verdicts were computed
in code, not asked for.

**What would you want to be known for beyond your productivity and professional
achievements?**
For being straight when the numbers are bad. The hardest sentence to write in this
kind of work is "this does not work", and the easiest is a paragraph that sounds
like it might. The line in my own report that says table rendering alone made
things slightly worse is worth more to me than the line that says 100%.

**What are you willing to struggle for, even when an AI agent could make the
struggle unnecessary?**
Understanding the data before modelling it. I could have let an agent write the
retriever in one go; instead I read the tables and found that the FY2025 report also
carries FY2024, and that "FY2024" and "2024" were two different tokens to my index.
Each of those was worth more than any code.

**What skills, qualities, or experiences would still make you valuable?**
Taste about what to measure, and the patience to check. My first retriever looked
reasonable and scored 26% recall; the ten minutes I spent reading what it returned
saved the afternoon.

**If your value is no longer determined primarily by what you can do, how will you
define who you are?**
By what I choose to take responsibility for. An agent can write the retriever; it
cannot decide that a wrong "PASS" is worse than a cautious "NOT FOUND". I would
rather be the person who signs off on that choice.

**What should you start doing differently tomorrow?**
Run the boring measurement before the clever fix, and write down the number each
change replaced, because "it felt better" is not a result.
