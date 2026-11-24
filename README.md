# Recipe RAG – Retrieval Stage

Retrieval stage for a recipe RAG (retrieval-augmented generation) system, built as a personal project.
Given a Korean natural-language request (e.g. "다이어트 저칼로리 샐러드"), it embeds the query with
`intfloat/multilingual-e5-large`, ranks about 5,000 Korean recipes by cosine similarity, picks a
small, varied set of candidates and formats each one as a short passage that ends with the
recipe ID it came from (`근거 ID`, "evidence ID").

**There is no generation step yet.** The output is the evidence a language model would be given;
no LLM is called anywhere in this repository.

> **Status.** The code was refactored on 2026-01-13 and that refactor is unfinished.
> The batch-evaluation results below were produced on 2026-01-12 by the earlier version.
> In the current code, dense retrieval, cosine top-k, cooking-method diversity and the evidence-ID
> output work, but the query-triggered filters and the rule-based reasons/tags described in the
> config are **not wired into the current entry points** (see [Known issues](#known-issues)).

## Pipeline

```
recipes_rag_with_metadata.jsonl ──build_from_jsonl.py──► recipes_output_fixed.csv (doc_text + kind/situation/method …)
                                                              │
                                              embedding.ipynb │ "passage: " + doc_text
                                                              ▼  multilingual-e5-large, L2-normalised
                                   recipe_embeddings_e5_large.npy (5000 × 1024)  +  recipe_meta.csv
                                                              │
query ──"query: " prefix──► e5 embedding ──► cosine top-k ──► filters / method diversity ──► rule-based reasons & tags
                                                              │
                                                              ▼
                                     evidence-tagged passages  ──►  (next step: LLM generator)
```

1. **Corpus preparation** – `build_from_jsonl.py` turns the source JSONL into a CSV with one
   `doc_text` per recipe (dish name, category `kind / situation / method`, main ingredient group,
   ingredients, servings) and separate `kind`, `situation`, `method` columns.
2. **Document embeddings** – `embedding.ipynb` encodes `"passage: " + doc_text` with
   `intfloat/multilingual-e5-large` (normalised, batch size 64, GPU) and saves
   `recipe_embeddings_e5_large.npy` (5000 × 1024, float32) and `recipe_meta.csv`.
3. **Query embedding and cosine top-k** – the query is embedded with the same model and scored
   against the L2-normalised document matrix (`core/retrieval.py`, `core/engine.py`,
   `local_retrieval.py`). The engine takes a pool of 5·k candidates and drops those below `min_score`.
4. **Query-triggered filters** – `configs/menu_rag.json → auto_filters` maps trigger words to
   metadata constraints, e.g. a query containing "샐러드" keeps `kind = 샐러드`, and "다이어트 / 저칼로리 /
   저열량" keeps `situation = 다이어트`. The filter is relaxed if it leaves fewer than k candidates.
   *(Used in the 2026-01-12 runs; not connected in the current code.)*
5. **Cooking-method diversity** – at most `max_per_method` results per cooking method (`method`
   column, e.g. 비빔, 데치기, 찜); if that leaves fewer than k results, the rest are filled by score.
6. **Rule-based explanations and tags** – `core/rule_engine.py` evaluates the `rules.reasons` and
   `rules.tags` blocks of the config (`contains_any` / `equals` / `regex` conditions combined with
   `any` / `all`) against the title and key ingredients; the highest-priority matching reason
   becomes the "추천 이유" (reason) line, followed by a note on the item's cooking method.
   *(Used in the 2026-01-12 runs; the evaluator is here but not called by the current engine.)*
7. **Evidence-tagged passages** – each result is rendered with the template in
   `configs/menu_rag.json → passage.template`:
   ```
   [MENU] {title}
   - 종류={kind} | 조리={method} | 상황={situation}
   - 핵심 재료: {ingredients_compact}
   - 근거 ID: {doc_id}
   ```
   `make_passages.py` / `passage_utils.py` build the same passages for the whole corpus
   (`menu_passages.jsonl`), so a generator can quote and cite them by recipe ID.

## Repository layout

```
core/
  engine.py          retrieval pipeline (candidates → selection → reasons) used by the CLIs
  unified_engine.py  wrapper so the CLI and batch script call the same engine entry point
  retrieval.py       query embedding ("query: " prefix) and cosine top-k helpers
  select.py          score-ordered selection with method/tag diversity
  rule_engine.py     condition evaluator for reason/tag rules
  enrich.py          ingredient clean-up and key-ingredient extraction
  render.py          text output (menu, category, tags, ingredients, reason, evidence ID)
  io.py, types.py    config loading, RunConfig / RunResult dataclasses
configs/menu_rag.json  data paths, field names, passage template, auto-filters, reason/tag rules
local_menu_rag.py      single-query CLI (core engine)
local_retrieval.py     minimal stand-alone top-k retrieval script
batch_eval.py          runs a JSONL of test queries and writes runs/run_<timestamp>*.json[l]
build_from_jsonl.py    source JSONL → CSV with doc_text and category columns
make_passages.py       CSV → evidence-tagged passages (menu_passages.jsonl)
passage_utils.py       passage template helpers
embedding.ipynb        document embedding with multilingual-e5-large
test_queries.jsonl     18 test queries (Korean) with k, min_score and diversity settings
runs/*_summary.json    batch-evaluation summaries (per-query logs are not tracked)
```

## Data

The recipe data and everything derived from it are **not included** (see `.gitignore`):
`recipes_rag_with_metadata.jsonl`, `recipes_output_fixed.csv`, `recipe_meta.csv`,
`menu_passages.jsonl`, `recipe_embeddings_e5_large.npy`, `doc_norm.npy`.

The corpus is 5,000 Korean recipes with a recipe serial number (`RCP_SNO`), dish name
(`CKG_NM`), an ingredient string and category labels for dish kind, situation, cooking method and
main ingredient group. It comes from a third-party Korean recipe dataset whose redistribution terms
are not confirmed, so it is not redistributed here. To run the code you need a JSONL file in this
format, one recipe per line:

```json
{"id": 123, "text": "요리명: …\n카테고리: <kind> / <situation> / <method>\n주재료군: …\n재료: …\n인분: …",
 "metadata": {"dish": "…", "kind": "…", "situation": "…", "method": "…",
              "main_material_group": "…", "servings": "…", "ingredients_keywords": "…"}}
```

Rebuild the derived files from it (run everything from the repository root; paths are relative):

```bash
python build_from_jsonl.py --input recipes_rag_with_metadata.jsonl --out_csv recipes_output_fixed.csv
# run embedding.ipynb  → recipe_embeddings_e5_large.npy, recipe_meta.csv
python make_passages.py --meta recipe_meta.csv --out menu_passages.jsonl
```

## How to run

```bash
pip install -r requirements.txt

# single query, stand-alone retrieval
python local_retrieval.py --q "매콤한 닭요리 추천" --k 5

# single query through the core engine (method diversity, evidence IDs)
python local_menu_rag.py --q "다이어트 저칼로리 샐러드" --k 7 --diversify_method --max_per_method 1 --generate_rule

# batch evaluation
python batch_eval.py --queries test_queries.jsonl --generate_rule
```

The first call downloads `intfloat/multilingual-e5-large` (~2.2 GB) from Hugging Face.
Query embedding runs on CPU; the document embeddings were computed once on a GPU.

## Batch evaluation (2026-01-12)

`test_queries.jsonl` holds 18 Korean queries in eight groups (diet/salad, spicy chicken, quick meals,
soups, vegetarian/tofu, family, guests, late-night snacks), each with its own `k` (5 or 7),
`min_score` (0.80 or 0.82) and method-diversity setting. The tracked summary
`runs/run_20260112_214047_summary.json` reports:

| Metric | Value |
| --- | --- |
| Queries | 18 |
| Average results returned | 6.67 (every query returned its requested k: 15 × 7, 3 × 5) |
| Average cosine score (mean of per-query means) | 0.851 |
| Failed queries | 0 |

Three runs on that day (`run_20260112_011529`, `_012015`, `_214047`) returned identical results.
In that run the auto-filters fired for the four diet/salad queries (`kind = 샐러드`,
`situation = 다이어트`). Of the 15 queries with `max_per_method = 1`, 10 returned 7 different
cooking methods, the four salad queries returned 6, and "따뜻한 국/탕 추천" returned 1 (when too few
methods are available the limit is relaxed to fill k). There are no relevance labels, so these numbers describe coverage and
similarity scores, not retrieval accuracy.

## Known issues

These were found while preparing the repository and are left unfixed so that the code is
published as it was written.

- **Filters and rules not connected.** `core/engine.py` only loads a config named
  `config.json`/`settings.json`/… from the working directory, so `configs/menu_rag.json` is not
  read by the engine; `core/rule_engine.py` has no function with the names the engine looks up,
  so reasons fall back to a fixed sentence and no guardrail is applied; `core/enrich.py` is never
  called. The January runs were produced by an earlier version of the engine that did use them.
  The "선정 기준: (자동 가드레일) …" header printed by `core/render.py` is inferred from the first
  result's category, not from a filter that was applied.
- **`core/select.py`** uses `sorted_candidates` before it is assigned, so the call fails and the
  engine falls back to its own method-diversity selection (`_select_with_diversity`).
- **Query prefix.** The engine's fallback query embedding does not add the `"query: "` prefix that
  e5 expects (and `local_retrieval.py` and `core/retrieval.py` do add), and it reloads the model for
  every query.
- **`batch_eval.py`** does not skip the `#` comment lines in `test_queries.jsonl` and stops with a
  JSON error; remove those lines (or run on a copy without them) to use it. The current version also
  writes a simpler summary (`total`, `avg_returned`) than the January summaries.
- Scripts use paths relative to the working directory, so run them from the repository root.

## Limitations

- Ranking uses only embedding similarity; there is no BM25/hybrid retrieval and no re-ranker.
- No labelled relevance judgements, so retrieval quality is not measured (no recall@k or nDCG).
- Category labels (kind/situation/method) come from the source data and are noisy
  (e.g. a chicken skewer labelled as dessert), which affects filters and diversity.
- Reason and tag rules are hand-written keyword rules for a few ingredient families.
- 5,000 recipes, brute-force cosine search in NumPy; no approximate index.

## Next step

Connect the filters and reason/tag rules to the current engine again, then plug an LLM generator
onto the evidence-tagged passages: give it the top-k passages and ask for an answer that cites the
`근거 ID` of every recipe it recommends, so each claim can be traced to a retrieved document.
