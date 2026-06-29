<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Aspect-Based%20Sentiment%20Analyzer&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Rule-Based%20NLP%20%E2%80%A2%20Zero%20LLM%20%E2%80%A2%20Structured%20JSON%20Output&descAlignY=55&descSize=16" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.x-4EAA25?style=for-the-badge&logo=python&logoColor=white)](https://nltk.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![No LLM](https://img.shields.io/badge/No%20LLM-Zero%20API%20Cost-FF6B6B?style=for-the-badge&logo=lightning&logoColor=white)]()
[![JSON Output](https://img.shields.io/badge/Output-Structured%20JSON-00B4D8?style=for-the-badge&logo=json&logoColor=white)]()

<br/>

> **Extract product features, opinions & sentiment from customer reviews — no deep learning, no API keys, no cost.**

<br/>

```
"camera quality is amazing but battery drains quickly"
        │                           │
   ┌────▼────┐                 ┌────▼────┐
   │ FEATURE │                 │ FEATURE │
   │ camera  │                 │ battery │
   │ quality │                 └────┬────┘
   └────┬────┘                      │
        │                      ┌────▼────────┐
   ┌────▼──────┐                │  OPINION    │
   │  OPINION  │                │drains quickly│
   │  amazing  │                └────┬────────┘
   └────┬──────┘                     │
        │                       ┌────▼─────┐
   ┌────▼──────┐                 │SENTIMENT │
   │SENTIMENT  │                 │NEGATIVE  │
   │ POSITIVE  │                 └──────────┘
   └───────────┘
```

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Format](#-output-format)
- [Core Logic Deep Dive](#-core-logic-deep-dive)
- [Scaling to 1000s of Reviews](#-scaling-to-1000s-of-reviews)
- [Limitations & Known Constraints](#-limitations--known-constraints)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)

---

## 🔍 Overview

**Aspect-Based Sentiment Analysis (ABSA)** goes beyond document-level polarity ("this review is positive") to extract *what* customers think about *specific product features*.

| Capability | Details |
|---|---|
| **Approach** | POS tagging + rule-based extraction |
| **Sentiment Engine** | VADER (`SentimentIntensityAnalyzer`) with custom lexicon |
| **Aspect Detection** | Noun-phrase chunking via NLTK `pos_tag` |
| **Opinion Binding** | Copula detection + proximity-based adjective/verb matching |
| **Negation Handling** | ✅ |
| **Clause Splitting** | Adversative conjunctions (`but`, `however`, `though`...) + comma + `and` |
| **LLM / ML Required** | ❌ None |
| **API Key Required** | ❌ None |
| **Output** | Structured JSON |

---

## ⚙️ How It Works

### Pipeline (per review)

```
Raw Review Text
      │
      ▼
┌─────────────────────┐
│  1. Split Clauses   │  ← "but / however / though / and / ,"
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. POS Tagging     │  ← nltk.pos_tag()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. NP Chunking     │  ← NN/NNS/NNP/NNPS tokens
│     (Feature)       │  ← compound noun-phrase walk-back/forward
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Opinion Binding │  ← copula detection (is/are/was/were)
│                     │  ← proximity ADJ/ADV/VERB after NP
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. VADER Scoring   │  ← opinion words (2×weighted) + clause
│     + Negation      │  ← score flipped on negation tokens
└──────────┬──────────┘
           │
           ▼
  {feature, opinion, sentiment}
```

### Sentiment Thresholds

| compound score | Label |
|---|---|
| ≥ `0.05` | `positive` |
| ≤ `-0.05` | `negative` |
| between | `neutral` |

Negation tokens (`not`, `no`, `never`, `neither`, `barely`, `hardly`, `without`) flip the compound score.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        analyze.py                           │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ split_clauses│──▶│extract_chunk │──▶│ extract_aspects│  │
│  └──────────────┘   └──────┬───────┘   └───────┬────────┘  │
│                            │                   │            │
│              ┌─────────────▼──────┐    ┌───────▼────────┐  │
│              │   collect_np()     │    │    main()      │  │
│              │   get_opinions()   │    │  reads reviews │  │
│              │   get_sentiment()  │    │  writes JSON   │  │
│              └────────────────────┘    └────────────────┘  │
└─────────────────────────────────────────────────────────────┘

Dependencies:
  nltk ──▶ punkt, punkt_tab, averaged_perceptron_tagger(_eng), vader_lexicon
```

---

## 📁 Project Structure

```
Aspect-Based-Sentiment-Analysis-NLP/
│
├── analyze.py          # Core ABSA engine (279 lines)
├── reviews.txt         # Input — one review per line
├── pyproject.toml      # Project metadata
├── requirements.txt    # Python dependencies
├── .python-version     # Pinned Python version
├── .gitignore
│
└── output/
    └── output.json     # Auto-generated results
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip

### Steps

```bash
# 1. Clone
git clone https://github.com/techakash32/Aspect-Based-Sentiment-Analysis-NLP.git
cd Aspect-Based-Sentiment-Analysis-NLP

# 2. (Optional) Virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. NLTK data downloads (handled automatically on first run, or manually)
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')
"
```

---

## 🎯 Usage

### 1. Add reviews

Edit `reviews.txt` — one review per line:

```
The camera quality is stunning but battery drains quickly.
Sound is loud and clear, however the build feels cheap.
Display is vivid and responsive. Setup was confusing and slow.
```

### 2. Run

```bash
python analyze.py
```

### 3. Check output

```bash
cat output/output.json
```

### Console output

```
Processing 3 reviews...
Done -> output/output.json
Reviews      : 3
Total aspects: 8
```

---

## 📦 Output Format

```json
[
  {
    "id": 1,
    "review": "camera quality is amazing but battery drains quickly",
    "aspects": [
      {
        "feature": "camera quality",
        "opinion": "amazing",
        "sentiment": "positive"
      },
      {
        "feature": "battery",
        "opinion": "drains quickly",
        "sentiment": "negative"
      }
    ],
    "summary": {
      "total_aspects": 2,
      "positive": 1,
      "negative": 1,
      "neutral": 0
    }
  }
]
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `id` | int | 1-indexed review ID |
| `review` | str | Original review text |
| `aspects[].feature` | str | Extracted noun phrase (product attribute) |
| `aspects[].opinion` | str | Opinion word(s) bound to feature |
| `aspects[].sentiment` | str | `positive` / `negative` / `neutral` |
| `summary.total_aspects` | int | Total aspect count for review |
| `summary.positive` | int | Positive aspect count |
| `summary.negative` | int | Negative aspect count |
| `summary.neutral` | int | Neutral aspect count |

---

## 🔬 Core Logic Deep Dive

### Clause Splitting (`split_clauses`)

Splits on adversative conjunctions first, then commas, then `and` (only when both sides are self-contained NP+opinion clauses).

```python
# self-contained check: must have ≥1 noun AND ≥1 opinion-POS token
def _is_self_contained(s):
    toks = nltk.word_tokenize(s)
    tag  = nltk.pos_tag(toks)
    return (any(t in NOUN_TAGS for _,t in tag) and
            any(t in OPN_TAGS  for _,t in tag))
```

### Noun-Phrase Chunking (`collect_np`)

Walks **back** ≤4 tokens (adj/noun) then **forward** over consecutive nouns — builds compound NPs like `camera quality`, `battery life`, `build quality`.

### VADER Custom Lexicon

40+ domain words added to `SID.lexicon` with polarity scores:

```python
SID.lexicon.update({
    'stunning':  3.0,  'laggy': -2.5,
    'overheats': -2.5, 'vivid':  2.5,
    # ... 36 more
})
```

### Verb-Noun Filter

Tokens in `VERB_NOUNS = {"drains","drain","drops","fluctuates",...}` are excluded from feature extraction to prevent mis-tagging action verbs as product features.

---

## 📈 Scaling to 1000s of Reviews

```python
import multiprocessing
from analyze import extract_aspects

def process(review):
    return extract_aspects(review)

with open("reviews.txt") as f:
    reviews = [l.strip() for l in f if l.strip()]

with multiprocessing.Pool() as pool:
    results = pool.map(process, reviews)
```

> ⚠️ NLTK's `pos_tag` is CPU-bound. `multiprocessing.Pool` is preferred over `threading` for CPU-heavy workloads.

---

## ⚠️ Limitations & Known Constraints

| Constraint | Detail |
|---|---|
| **Language** | English only (uses `en_core_web_sm` POS model + VADER) |
| **Sarcasm** | Not detected — `"great, just great"` scores positive |
| **Implicit aspects** | `"This is expensive"` (no explicit noun) may be missed |
| **Complex syntax** | Deeply nested clauses may split incorrectly |
| **Domain coverage** | VADER lexicon custom words are product-review tuned; other domains need re-tuning |
| **No training data** | Rule-based; accuracy depends on grammar quality of input |

---

## 🗺️ Roadmap

- [ ] spaCy dependency parser for richer aspect-opinion binding
- [ ] CSV / JSONL batch input support
- [ ] Streamlit demo UI
- [ ] Domain-configurable VADER lexicon via external YAML
- [ ] Confidence score per aspect
- [ ] Multi-language support (FR, ES, DE)
- [ ] REST API wrapper (FastAPI)

---

## 🤝 Contributing

1. Fork the repo
2. Create branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "feat: description"`
4. Push: `git push origin feature/your-feature`
5. Open Pull Request

---

## 📄 License

MIT © [techakash32](https://github.com/techakash32)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" />

**If this helped you, drop a ⭐ — it means a lot!**

</div>
