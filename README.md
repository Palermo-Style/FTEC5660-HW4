# FTEC5660 Homework 4 — solution

A lexical RAG retriever over 40 company reports. Five functions in `hw4.py`, one
ablation script (`ablation.py`) and its output (`ablation.csv`); every run of
`hw4.py` writes `results.csv`.

`python3 hw4.py --data-folder public_test` → **74.67 / 75** on the public set
(the grade uses the private set).

---

## 1. Pipeline

```
corpus/doc_XXX.html
      │
      │  document_meta(): doc_type, company, fiscal year read from the first lines
      ▼
 BeautifulSoup, body blocks in order: h2/h3 · p · ul/ol · table
      │                                        │
      │                                        └─ table_to_text(): one line per row,
      │                                           "Revenue | 2024: 2,649.8 | 2023: 2,316.8",
      │                                           caption first (it carries the unit)
      ▼
 chunk_document(): section-aware packing, ≤1500 chars per chunk, never splitting a
 table row, each chunk prefixed with "Company | doc type FY2025 | section"
 metadata: source, company, fiscal_year, doc_type   (extra: none needed)
      │
      ▼
 build_index(): Okapi BM25 (k1=1.5, b=0.75) over lower-cased tokens; numbers lose
 thousands separators, plurals are stemmed, "FY2024" → "2024"; chunks grouped by
 (company, doc_type, fiscal_year) and the whole text of each company kept for the
 "is this figure disclosed" check
      │
 question ──parse_query()──► companies (corpus names + near-miss names that are not
      │                      brokers or table labels) · fiscal_year · doc_type
      ▼
 retrieve(): keep the chunks of the named company/companies and document type whose
 fiscal year can cover the asked year (an annual report covers its year and the
 previous one; an earnings call its year and the next; a research report FY2024A to
 FY2027E), then rank by BM25 + 3.0 when a table row names the asked figure and the
 year. Comparison questions take one chunk per company in round-robin, so five
 chunks of one company can never crowd out the other.
      │  [] if the company is absent from the corpus, if no document covers the year,
      │  or if the asked figure is named by a word pair the company's documents never use
      ▼
 generate() — fixed generator, top-5 chunks with a metadata header —► answer
```

The design follows one rule: **the model never decides anything, because the
generator is fixed and my job is to make its context unambiguous.** Everything
that can be computed is computed. Tables are rendered so that a value sits on the
same line as its row label and its column header, which is what makes both BM25
and the generator able to use a figure without guessing which year or which row it
came from. Metadata does the work BM25 is bad at: telling Northgate Robotics from
Northgate Payments and 2024 from 2025. The year window encodes what the data
actually is — the FY2025 report also shows FY2024, so a question about 2024 can be
answered from either report, while a question about 2021 has no answer anywhere.
Abstention is deliberately conservative: it fires only on evidence that the corpus
cannot answer (unknown company, uncovered year, a figure whose words never appear
together), because every wrong abstention costs as much as a wrong answer.

## 2. Ablation study

`python3 ablation.py --data-folder public_test` reproduces this table
(`ablation.csv`). Each row changes exactly one component from the row above and
is measured with the same fixed generator and the same scoring code.

| Config | One change | recall@5 | kept | rejected | accuracy | Tasks 2.1+2.3+3+4 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | naive baseline: `get_text()`, fixed 1000-character splits, required metadata, BM25 top-5 over everything, no parsing, never abstains | 26.1% | 100% | 0% | 45.0% | 14.00 / 50 |
| V1 | + `table_to_text()` | 23.9% | 100% | 0% | 41.7% | 13.33 / 50 |
| V2 | + section-aware chunking with a company/FY/section header | 95.7% | 100% | 0% | 88.3% | 37.67 / 50 |
| V3 | + `parse_query()` metadata filter | 97.8% | 100% | 0% | 91.7% | 38.33 / 50 |
| V4 | + fiscal-year window | 97.8% | 100% | 35.7% | 96.7% | 42.90 / 50 |
| V5 | + per-company round-robin for comparisons | 97.8% | 100% | 35.7% | 98.3% | 43.24 / 50 |
| V6 | + abstention rules | 97.8% | 100% | 100% | 98.3% | 49.67 / 50 |
| V7 | + table-row bonus in the ranking | 100% | 100% | 100% | 98.3% | 49.67 / 50 |

**What each change did, and why.**

*Table rendering alone (V1) did nothing.* This was the surprise. Rendered rows
carry their year and their row label, but the chunker still cut the document every
1,000 characters, so rows were sliced in half and "Inventories | 31 December 2025:
297.4" reached the index as three unrelated fragments. Recall went down slightly
(26.1% → 23.9%): the rendered text is longer, so the fixed windows straddle more
tables. A good table format is only worth what the chunker lets it be.

*Section-aware chunking (V2) is where the homework was won.* Recall jumped from
23.9% to 95.7% and accuracy from 41.7% to 88.3%, worth 24 points on its own. Two
effects compound: chunks no longer split table rows, and every chunk carries
"Company | annual report FY2025 | financial statements" in its text, so BM25 can
match a company and a year that the question names but the body never repeats.
Without the header, a balance-sheet chunk says "31 December 2025: 445.1" and
nothing about whose balance sheet it is.

*The metadata filter (V3) is worth less than it looks* — 2.1 points of recall and
3.4 of accuracy — because the header already put the company and the year into the
chunk text. What the filter buys is precision on the hard cases: similar names
(Brightwater Energy vs Brightwater Software) and the wrong year's report, which
the header alone cannot separate when both documents say "2025" somewhere.

*The year window (V4) pays for itself in abstention, not accuracy.* It raises
accuracy by 5 points, but its real contribution is that 2021 and 2022 questions
stop finding a chunk with the right company and document type and start finding
nothing — which is what "kept + rejected − 1" rewards. Together with V6 it takes
Task 3 from 0 to 10.

*Round-robin (V5) is small and cheap*: accuracy 96.7% → 98.3%. A comparison needs
two chunks from two companies, and a single global ranking will spend all five
slots on whichever company matches the question's wording better.

*Abstention (V6) is the single largest scoring jump in the table*, 43.24 → 49.67,
because Task 3 is all-or-nothing: returning chunks everywhere scores zero by
construction. Three rules catch the fourteen unanswerable public questions without
rejecting a single answerable one: a company that is not in the corpus, a year no
document covers, and a figure whose words never appear together in that company's
documents ("lobbying expenses", "customer churn rate", "carbon offset purchases").

*The table-row bonus (V7) is the smallest change and still worth 0.33 points*: it
finds the one table question whose row was in a chunk that BM25 ranked tenth,
raising recall from 97.8% to 100%.

**A note on reproducibility.** The generator runs at temperature 0 but the API is
not guaranteed deterministic; the answer-accuracy column can move by about one
question between runs. Recall, kept and rejected are computed from my own code and
are exactly reproducible.

## 3. Error analysis

The public set ends with **one wrong answer out of 60**, and it is not a retrieval
error. For "Which company held more cash and cash equivalents at 31 December 2024:
Lumen Harbor Foods or Sablewood Retail?" both companies' balance-sheet chunks were
retrieved (the hit flag is true) and the fixed generator still named the wrong
company. That is the failure mode this homework cannot fix from the retrieval side:
two numbers that are correct on their own lines, in two different chunks, and a
generator that compares them in its head. The only lever left would be to put both
figures into one chunk, which the rules do not allow (a chunk's text must come from
its own document).

The other 59 answers are right, including all 22 table questions and all 14
unanswerable ones. Recall@5 is 100% on the public set.

## 4. Task 6 — Reflection

**What kind of person would you want to be in that world?**
The one who decides what is worth doing, and who can tell whether the thing that
came back is any good. This homework is a small version of that: the generator was
given to me and wrote every answer, and everything that determined whether the
answer was right — what it was allowed to see — was mine. I would want to keep
being the person who owns that decision.

**Which of your current ambitions would still matter?**
The ones about judgment, not throughput. I want to be someone whose "this is
probably wrong" is worth listening to. In this homework the model produced a
confident, well-formed summary for every company; the only reason the verdicts
were right is that I computed them in SQL and never asked the model. That instinct
— knowing which parts to take away from the agent — feels like the part of my
ambition that a better model does not erode.

**What would you want to be known for beyond your productivity and professional
achievements?**
For being straight with people when the numbers are bad. In the HW3-style work the
hardest sentence to write is "this company fails", and the easiest is to hedge.
Being known as someone who reports the failing case plainly is worth more to me
than being known as fast.

**What are you willing to struggle for, even when an AI agent could make the
struggle unnecessary?**
Understanding the data before modelling it. I could have let an agent write the
retriever in one shot; instead I spent the time looking at the tables, finding that
the FY2025 report also carries FY2024, and that "FY2024" and "2024" were two
different tokens to my index. Every one of those was worth more than the code. I
want to keep doing the part where I have to be confused for a while.

**What skills, qualities, or experiences would still make you valuable?**
Taste about what to measure, and the patience to check. My first chunker looked
fine and scored 26% recall; the run that told me why took ten minutes and saved the
afternoon. Also the willingness to write down what did *not* work — the ablation
table above says table rendering alone made things slightly worse, and that line is
more useful to the next person than the one where I got 100%.

**If your value is no longer determined primarily by what you can do, how will you
define who you are?**
By what I choose to take responsibility for. The agent can write the retriever; it
cannot decide that "kept + rejected − 1" is the right way to think about
abstention, or that a wrong "PASS" is worse than a cautious "NOT FOUND". I would
rather be the person who signs off on those choices than the person who produces
the most output.

**What should you start doing differently tomorrow?**
Two things. Run the boring measurement before the clever fix — the ablation table
was more informative than any single improvement I made. And when something works,
write down the number it replaced, because "it felt better" is not a result.
