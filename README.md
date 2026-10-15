# Redrob Candidate Ranker

**Team Ctrl+Alt+Defeat:** Ankit Kumar, Aaditya Rathi, Kshitiz Tyagi, Purabjeet Singh Bagga

Ranks a candidate pool against the Senior AI Engineer (Founding Team) job description and writes a top-100 CSV with columns `candidate_id`, `rank`, `score`, and `reasoning`.

The ranking step is CPU-only and does not use the network. On 100,000 candidates it takes about 45 seconds and peaks near 1.5 GB RAM.

`candidates.jsonl` is not in this repository. Put `candidates.jsonl` or `candidates.jsonl.gz` at the repo root, or pass another path with `--candidates`. The default job description is `data/job_description.txt`.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Once. Writes data/candidate_embeddings.npy and data/candidate_ids.json.
# About 4.3 minutes for 100,000 candidates. Offline after the model is cached.
python scripts/precompute_embeddings.py --candidates ./candidates.jsonl

# About 45 seconds. CPU-only, no network.
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

python validate_submission.py ./submission.csv
```

A `.gz` pool is read directly. On Windows, activate the venv with `.venv\Scripts\activate`.

`--semantic-backend` defaults to `auto`. That uses the precomputed embeddings when `data/candidate_embeddings.npy` exists, and TF-IDF otherwise. To rank without the precompute step:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --semantic-backend tfidf
```

Before upload, rename `submission.csv` to the registered participant id, for example `team_xxx.csv`.

## Sandbox

Both options run the same scoring code on a sample of at most 100 candidates, with the TF-IDF backend. They do not download a model or require precomputed embeddings.

Streamlit, for a local run or a Hugging Face / Streamlit Cloud app:

```bash
pip install -r requirements-sandbox.txt
streamlit run app.py
```

Point the hosted app at `app.py` and `requirements-sandbox.txt`. With no upload, it ranks the bundled 50-candidate file `data/sample_candidates.json`.

Docker:

```bash
docker build -t redrob-ranker .
docker run --rm redrob-ranker

docker run --rm -v "$PWD":/data redrob-ranker \
  --candidates /data/candidates.jsonl --out /data/submission.csv \
  --semantic-backend tfidf
```

The image installs `requirements-core.txt` only.

## Scoring

Each candidate gets a score in `[0, 1]`:

```
score = (weighted components) × disqualifier × availability
```

Profiles that fail the plausibility checks are multiplied by 0.02. Equal scores are ordered by `candidate_id` ascending. The CSV contains the top 100.

| Component | Weight | What is measured |
|---|---:|---|
| Title | 0.20 | Match to AI/ML, search, NLP, and recommendation titles. Generic engineering titles are scaled by other AI evidence. Junior, intern, and trainee titles are reduced. |
| Skills | 0.22 | Coverage of required skill families, weighted by proficiency, months of use, endorsements, and the platform assessment score. |
| Career evidence | 0.22 | Work-history mentions of retrieval, ranking, recommendation, or production ML. |
| Semantic | 0.14 | Cosine similarity between a short candidate document and the job description. |
| Experience | 0.12 | Closeness to a 5–9 year range, with 6–8 years preferred. |
| Location | 0.10 | Pune and Noida, then other Tier-1 Indian cities, then the rest of India, then outside India. `willing_to_relocate` lowers the penalty. |

The disqualifier multiplier is floored at 0.30. It covers mostly services or consulting careers, a research title with no production evidence, CV/speech/robotics work without NLP or information retrieval, framework-only profiles, and a run of short stints.

The availability multiplier stays in `[0.40, 1.05]`. It uses recruiter response rate, last activity, open-to-work, notice period, profile completeness, interview completion, GitHub activity, and recruiter saves plus search appearances relative to this pool’s median and 90th percentile.

Semantic backends:

- `embeddings` uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, about 80 MB). `scripts/precompute_embeddings.py` embeds the pool once. Ranking embeds the job description only and multiplies it against the saved matrix.
- `tfidf` builds a word-bigram TF-IDF index during the ranking run.

The `reasoning` column is filled from fields on that profile: current title, years of experience, skills present on the record, location, and the score breakdown.

## Layout

```
rank.py                              CLI for the ranking step
app.py                               Streamlit sandbox
Dockerfile
scripts/precompute_embeddings.py
validate_submission.py
requirements.txt                     ranking and embeddings
requirements-core.txt                TF-IDF ranking only
requirements-sandbox.txt             core plus Streamlit
submission_metadata.yaml
src/redrob_ranker/
  jd_spec.py                         job-description rubric
  features.py                        per-candidate features
  plausibility.py                    internal-consistency checks
  retrieval.py                       TF-IDF backend
  retrieval_dense.py                 embedding backend
  scoring.py
  reasoning.py
  pipeline.py
tests/test_pipeline.py
data/                                job description, schema, samples
outputs/                             generated submission.csv
```

`data/candidate_embeddings.npy` and `data/candidate_ids.json` are gitignored. The embedding matrix is about 153 MB and is rebuilt by the precompute script.

## Runtime

Measured on an Apple Silicon laptop, 8 CPU cores, 16 GB RAM. No GPU. No network during ranking.

| Step | Time |
|---|---|
| Precompute embeddings, once | ~4.3 min |
| Rank with embeddings | ~43 s |
| Rank with TF-IDF | ~46 s |

The same input produces the same CSV.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

`python tests/test_pipeline.py` runs the same checks without pytest.
