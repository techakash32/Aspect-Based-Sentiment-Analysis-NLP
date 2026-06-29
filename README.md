# Aspect Sentiment Analyzer

## Setup
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Run
```bash
python analyze.py
```

## Files
- `reviews.txt`  — input (one review per line)
- `output.json`  — auto-generated results
- `analyze.py`   — main script

## Output format
```json
[{
  "id": 1,
  "review": "camera quality is amazing but battery drains quickly",
  "aspects": [
    {"feature": "camera quality", "opinion": "amazing",         "sentiment": "positive"},
    {"feature": "battery",        "opinion": "drains quickly",  "sentiment": "negative"}
  ],
  "summary": {"total_aspects": 2, "positive": 1, "negative": 1, "neutral": 0}
}]
```

## Scale (1000s of reviews)
```python
import multiprocessing
with multiprocessing.Pool() as pool:
    results = pool.map(extract_fn, reviews)
```
