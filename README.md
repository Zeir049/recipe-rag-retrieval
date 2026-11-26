# Recipe RAG – Retrieval Stage

Retrieval stage for a recipe RAG (retrieval-augmented generation) system, built as a personal project.
Given a Korean natural-language request (e.g. "다이어트 저칼로리 샐러드"), it embeds the query with
`intfloat/multilingual-e5-large`, ranks about 5,000 Korean recipes by cosine similarity, applies
filters triggered by words in the query, picks a set of candidates with varied cooking methods, adds
a rule-based reason to each, and formats each one as a short passage that ends with the recipe ID it
came from (`근거 ID`, "evidence ID").

**There is no generation step yet.** The output is the evidence a language model would be given;
no LLM is called anywhere in this repository.

## Pipeline

```
recipes_rag_with_metadata.jsonl ──build_from_jsonl.py──► recipes_output_fixed.csv (doc_text + kind/situation/method …)
                                                              │
                                              embedding.ipynb │ "passage: " + doc_text
                                                              ▼  multilingual-e5-large, L2-normalised
                                   recipe_embeddings_e5_large.npy (5000 × 1024)  +  recipe_meta.csv
                                                              │
query ──"query: " prefix──► e5 embedding ──► cosine top-(10·k) ──► filters / dedupe / method diversity ──► rule-based reasons & tags
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
3. **Query embedding and cosine top-k** – the query is embedded as `"query: " + query` with the
   same model (loaded once per process) and scored against the L2-normalised document matrix
   (`core/retrieval.py`). The engine keeps a pool of `pool_multiplier · k` = 10·k candidates.
4. **Ingredients and tags** – `core/enrich.py` strips quantities and units from the ingredient
   string and keeps the first 5 as key ingredients; `core/rule_engine.py` assigns tags from
   `rules.tags` in the config (e.g. 단백질, 채소위주, 매콤).
5. **Query-triggered filters** – `configs/menu_rag.json → auto_filters` maps trigger words to
   metadata constraints: a query containing "샐러드" requires `kind` to contain 샐러드, and
   "다이어트 / 저칼로리 / 저열량" requires `situation` to contain 다이어트.
6. **Selection** (`core/select.py`) – candidates below `min_score` are dropped, duplicate titles
   are removed, and at most `max_per_method` results per cooking method are taken. If the
   constraints leave fewer than k results, they are relaxed step by step (filter + method + tag →
   … → score only) to fill k.
7. **Rule-based explanations** – the highest-priority rule in `rules.reasons` whose conditions
   (`contains_any` / `equals` / `regex`, combined with `any` / `all`) match the title or key
   ingredients becomes the reason ("추천 이유"); if none matches, a fixed fallback sentence is used.
   When an item's cooking method appears only once among the results, a note on the method is added.
8. **Evidence-tagged passages** – each result is rendered with `passage.template` from the config:
   ```
   [MENU] {title}
   - 종류={kind} | 조리={method} | 상황={situation}
   - 핵심 재료: {ingredients_compact}
   - 근거 ID: {doc_id}
   ```
   `make_passages.py` / `passage_utils.py` build the same kind of passage for the whole corpus
   (`menu_passages.jsonl`), so a generator can quote and cite them by recipe ID.

## Repository layout

```
core/
  engine.py          the pipeline above (steps 3–8), configured by configs/menu_rag.json
  unified_engine.py  wrapper so the CLI and the batch script call the same engine entry point
  retrieval.py       cached model, query embedding ("query: " prefix), cosine top-k
  select.py          auto-filters, dedupe, method/tag diversity with step-wise relaxation
  rule_engine.py     condition evaluator for reason/tag rules
  enrich.py          ingredient clean-up and key-ingredient extraction
  render.py          text output (filters, menu, category, tags, ingredients, reason, evidence ID)
  io.py, types.py    config loading, RunConfig / RunResult dataclasses
configs/menu_rag.json  data paths, model, field names, passage template, auto-filters, reason/tag rules
local_menu_rag.py      single-query CLI
local_retrieval.py     minimal stand-alone top-k retrieval script (no filters or rules)
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
main ingredient group.

**Source: 만개의레시피 (10000recipe) recipe data, as distributed via KADX (농식품 빅데이터 거래소).**
This was identified from the schema rather than from a record kept in the project: the fields
`RCP_SNO` and `CKG_NM` and the four category axes (종류 / 상황 / 방법 / 재료, here `kind`,
`situation`, `method`, `main_material_group`) follow that dataset's naming, and the public
식품안전나라 recipe API uses different fields (`RCP_SEQ`, `RCP_NM`, …).

Licence not confirmed; the data is not included in this repo. The project starts from a
preprocessed file, `recipes_rag_with_metadata.jsonl`, with 5,000 recipes.

To run the code you need a JSONL file in this format, one recipe per line:

```json
{"id": 123, "text": "요리명: …\n카테고리: <kind> / <situation> / <method>\n주재료군: …\n재료: …\n인분: …",
 "metadata": {"dish": "…", "kind": "…", "situation": "…", "method": "…",
              "main_material_group": "…", "servings": "…", "ingredients_keywords": "…"}}
```

Rebuild the derived files from it (run from the repository root):

```bash
python build_from_jsonl.py --input recipes_rag_with_metadata.jsonl --out_csv recipes_output_fixed.csv
# run embedding.ipynb  → recipe_embeddings_e5_large.npy, recipe_meta.csv
python make_passages.py --meta recipe_meta.csv --out menu_passages.jsonl
```

## How to run

```bash
pip install -r requirements.txt

# one query: filters, method diversity, rule-based reasons
python local_menu_rag.py --q "다이어트 저칼로리 샐러드" --k 7 --min_score 0.82 --diversify_method --max_per_method 1 --generate_rule

# the same query as evidence-tagged passages (what a generator would receive)
python local_menu_rag.py --q "다이어트 저칼로리 샐러드" --k 7 --diversify_method --passages

# batch evaluation over test_queries.jsonl
python batch_eval.py --queries test_queries.jsonl

# minimal retrieval only
python local_retrieval.py --q "매콤한 닭요리 추천" --k 5
```

The first call downloads `intfloat/multilingual-e5-large` (~2.2 GB) from Hugging Face. Query
embedding runs on CPU; the 18-query batch took about 2 minutes on CPU (including the model load). The document
embeddings were computed once on a GPU.

## Batch evaluation

`test_queries.jsonl` holds 18 Korean queries in eight groups (diet/salad, spicy chicken, quick meals,
soups, vegetarian/tofu, family, guests, late-night snacks), each with its own `k` (5 or 7),
`min_score` (0.80 or 0.82) and method-diversity setting.

| Metric | 2026-01-12 (original code) | 2026-11-24 (restored code) |
| --- | --- | --- |
| Queries | 18 | 18 |
| Average results returned | 6.67 | 6.67 |
| Average cosine score (mean of per-query means) | 0.851 | 0.851 |
| Queries returning fewer than k | 0 | 0 |
| Queries where a filter fired | 4 | 4 |

Summaries: `runs/run_20260112_214047_summary.json` and `runs/run_20261124_182545_summary.json`.
In the 2026-11-24 run:

- Every query returned its requested k (15 × 7, 3 × 5); the selected recipes and their order are
  identical to the 2026-01-12 run for all 18 queries.
- The filters fired for the four diet/salad queries (`kind = 샐러드`, `situation = 다이어트`), and all
  7 results of each of those queries satisfy both constraints.
- Of the 15 queries with `max_per_method = 1`, 10 returned 7 different cooking methods, the four
  salad queries returned 6, and "따뜻한 국/탕 추천" returned 1 (almost every soup is 끓이기, so the
  limit was relaxed to fill k).
- 77 of the 120 results got a rule-based reason; the other 43 got the fallback sentence. 100 of
  120 got at least one tag.

There are no relevance labels, so these numbers describe coverage, constraint satisfaction and
similarity scores, not retrieval accuracy.

## What changed and why (2026-11-24)

The code was refactored on 2026-01-13 (a common entry point for the CLI and the batch script), and
the refactor was left unfinished: the 2026-01-12 runs show filters, reasons and tags, but the
refactored code no longer produced them. This update reconnects them.

| Problem in the 2026-01-13 code | Change |
| --- | --- |
| The engine looked for `config.json` in the working directory, so `configs/menu_rag.json` was never read. | `core/engine.py` loads `configs/menu_rag.json` (or `--config`); data paths in it are resolved from the repository root. |
| The engine looked up functions by guessed names (`make_rule_answer`, `extract_guardrail`, `enrich`, …) that do not exist, so filters, reasons, tags and ingredient clean-up were silently skipped. | `core/engine.py` calls the existing functions directly: `infer_auto_filters`, `attach_filter_ok`, `select_guarded`, `enrich_doc`, `infer_tags`, `pick_best_reason`. |
| `core/select.py` used a variable before assigning it, so every call failed and a simpler fallback was used. | Restored the 2026-01-12 `core/select.py` (auto-filters, dedupe by title, method/tag limits with step-wise relaxation). |
| The query was embedded without the `"query: "` prefix that E5 expects (documents use `"passage: "`), and the 2 GB model was reloaded for every query. | `core/retrieval.py` adds the prefix and caches the model (`functools.lru_cache`). |
| `batch_eval.py` stopped at the `#` comment lines in `test_queries.jsonl`. | Comment and blank lines are skipped; per-query `generate_rule` is honoured; the summary again reports average score and failures. |
| The method note ("조리 방법이 …이라 다른 후보와 스타일이 달라…") was added to every result, even when several results share the method. | Added only when the item's method is unique among the results. |
| The "(자동 가드레일)" header was guessed from the first result. | The header lists the filters that were actually applied ("(자동 필터)"). |
| `make_passages.py` built JSON lines by hand and left the passages' line breaks unescaped, so `menu_passages.jsonl` had 19,430 lines for 5,000 records and could not be parsed. | It writes each record with `json.dumps`; the file now has 5,000 lines that all parse, with the same content. Nothing else in the repository reads this file, so retrieval results are unaffected. |

The agreement with the 2026-01-12 run (same recipes, same order, same average score) is the check
that the restored pipeline behaves like the original one.

## Limitations

- Ranking uses only embedding similarity; there is no BM25/hybrid retrieval and no re-ranker.
- No labelled relevance judgements, so retrieval quality is not measured (no recall@k or nDCG).
- Only two filter rules exist (salad, diet), so filters fire for 4 of the 18 test queries.
- Category labels (kind/situation/method) come from the source data and are noisy
  (e.g. a chicken skewer labelled as dessert), which affects filters and diversity.
- Reason and tag rules are hand-written keyword rules for a few ingredient families; 43 of 120
  results in the batch run fall back to the generic sentence, and keyword matches can be loose
  (e.g. "고추" in 마른 홍고추 tags a pasta salad as 매콤).
- 5,000 recipes, brute-force cosine search in NumPy; no approximate index.
- `batch_eval.py` writes to `runs/` relative to the working directory, and the data-preparation
  scripts use relative default paths, so run them from the repository root.

## Next step

Plug an LLM generator onto the evidence-tagged passages: give it the top-k passages
(`--passages` output) and ask for an answer that cites the `근거 ID` of every recipe it recommends,
so each claim can be traced to a retrieved document.
