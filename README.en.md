# Omada API Assistant

A web-based AI assistant for the **TP-Link Omada Open API**. It indexes your controller's OpenAPI
specification into a local RAG and an OpenAI agent queries it to give you the exact endpoint, parameters,
permissions, error codes and code examples.

[Versión en español](README.md)

## Features

- **Grounded answers**: the model searches the index and reads each operation's detail before answering.
- **Hybrid retrieval**: vectors + BM25 keywords (exact identifiers like `createNewSite` or `-33101`), merged with Reciprocal Rank Fusion.
- **Web portal** (UI text is in Spanish): streaming chat, visible agent steps, endpoints panel, `Ctrl+K` search, interactive category map, light/dark theme, local history.
- **No extra servers**: SQLite and ChromaDB run in-process. Python only.

## Requirements

Python 3.10+, an OpenAI API key, and the OpenAPI spec (`.json` or `.zip`) from your controller.

## Quick start

```bash
git clone <your-repo-url> && cd omada-api-assistant
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env_template .env                                  # then set OPENAI_API_KEY

# put the spec in data/spec/ (Settings > Platform Integration > Open API > Online API Document)
python manage.py build-index
python manage.py serve                                 # http://localhost:8000
```

## Commands

`build-index` (`--full`, `--keyword-only`) · `import-guide FILE` · `serve` (`--reload`) · `search "query"` · `stats` · `usage` (spend report) · `doctor`

## Configuration

Everything lives in `.env` (see `.env_template`). Key settings: `OPENAI_API_KEY`, `OPENAI_MODEL`,
`OPENAI_REASONING_EFFORT` (reasoning models only; leave empty otherwise), `OPENAI_EMBEDDING_MODEL`
(rebuild with `build-index --full` if changed), `PORTAL_ACCESS_TOKEN`.

## Authentication guides

The spec does **not** contain the authentication flow. Copy the *Open API Access Guide* text into a file
and import it (no formatting needed):

```bash
python manage.py import-guide access_guide.txt --name organization_access_guide
python manage.py build-index
```

`import-guide` turns numbered titles into sections, makes tables readable and fences `curl`/JSON examples.
Without guides the assistant says it lacks that information instead of guessing.

## Production: spending limits, users, logs

Every answered question is written to a local SQLite ledger (`data/usage/usage.db`: who, tokens, estimated cost,
time; never the text). Before answering, four caps are checked and the user gets a clear message when one is reached:
`BUDGET_DAILY_USD` (all users, per day), `BUDGET_MONTHLY_USD`, `BUDGET_USER_DAILY_USD` (per person) and
`USER_DAILY_REQUESTS` (per person; works even when the model's price is unknown). `0` turns a cap off; users get a
heads-up at 80 % and see the approximate cost under each answer. Costs use the table in `usage/pricing.py`
(override with `PRICE_INPUT_PER_MTOK` / `PRICE_OUTPUT_PER_MTOK`). A single answer can overshoot a cap by its own cost.
`python manage.py usage` prints the report.

Behind a login proxy (Cloudflare Access, oauth2-proxy...) set `PORTAL_USER_HEADER=X-Forwarded-Email` and
`PORTAL_BEHIND_PROXY=true` so limits and usage are per person - **only when the proxy sets that header and strips
any client-sent value**. `PORTAL_ADMINS` restricts who can see everyone's usage (`/api/usage`).
`LOG_FORMAT=json` emits one JSON line per request/question with an `X-Request-ID`; `LOG_FILE` adds a rotating file.
Question text is not logged unless `LOG_QUESTIONS=true`. Not covered yet: the per-minute rate limiter is in-memory and
per process, the vector index is local, and there is a single spec version per index.

## Evaluations

```bash
python manage.py eval                                   # search + 56 curated questions
python manage.py eval --only retrieval --keyword-only   # free, no API key
python manage.py eval --check                           # validate the questions against your index
```

Writes a plain-language report to `evals/results/<date>/report.md` (UI/report text is in Spanish): an overall
0-100 grade, per-type/difficulty/topic tables, an "invented paths" reliability section, speed and cost, every failed
question with the reason, and recommendations. Search is measured on 300 sampled operations (top-1/3/8); answers are
graded with deterministic checks (right endpoint, key terms, no invented paths) plus an optional LLM judge
(`OPENAI_JUDGE_MODEL`; use a model stronger than the one answering - a same-model judge only raises warnings).
Search is also measured with everyday-wording questions (`eval --only retrieval --natural`, cached in
`evals/natural_queries.json`), because questions that repeat the docs' own words flatter the search. `--baseline latest` compares with the previous run. Questions live in
`evals/questions.json`; guide-dependent ones are skipped until you import the access guide.

## Architecture

One document per operation; full schemas are served on demand (`get_operation`) to keep prompts small.
The server is stateless (the browser sends the history), embeddings failures degrade to keyword search,
and the agent uses the OpenAI Responses API with streaming function calls.

## Tests

`pip install -r requirements-dev.txt && pytest` (no network or API key needed).

## Privacy & limits

Questions and retrieved documentation are sent to OpenAI. Answers are AI-generated: verify critical details.
For network use, set `PORTAL_ACCESS_TOKEN` and serve over HTTPS. Stop the server before rebuilding the index.

## License

MIT. Independent project, not affiliated with TP-Link. The spec is not included.
