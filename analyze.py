import nltk, json, re, os, urllib.request, urllib.error

nltk.download('punkt',                          quiet=True)
nltk.download('punkt_tab',                      quiet=True)
nltk.download('averaged_perceptron_tagger',     quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('vader_lexicon',                  quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

SID = SentimentIntensityAnalyzer()

# ── add missing words to VADER ────────────────────────────────────────────────
SID.lexicon.update({
    'unhelpful': -3.0, 'unresponsive': -2.5, 'excessive': -2.5,
    'overheats': -2.5, 'overheat': -2.5, 'incomplete': -2.0,
    'delayed': -2.0,   'delay': -1.5,     'fluctuates': -1.5,
    'repetitive': -2.0,'inadequate': -2.5, 'confusing': -2.5,
    'complicated': -2.0,'unstable': -2.5,  'limited': -1.5,
    'average':  -0.5,  'vivid': 2.5,       'rich': 2.0,
    'secure': 2.0,     'useful': 2.0,      'simple': 1.5,
    'attractive': 2.5, 'comfortable': 2.0, 'accurate': 2.0,
    'responsive': 2.0, 'crisp': 2.0,       'stunning': 3.0,
    'outstanding': 3.0,'powerful': 2.5,    'premium': 2.5,
    'smooth': 2.0,     'enjoyable': 2.5,   'easy': 2.0,
    'strong': 2.0,     'clear': 1.5,       'loud': 1.5,
    'sharp': 2.0,      'bright': 1.5,      'vibrant': 2.5,
    'noisy': -2.0,     'laggy': -2.5,      'blurry': -2.0,
    'drains': -2.0,    'drain': -2.0,      'quickly': -1.0,
    'frequently':-1.5, 'late': -2.0,       'slow': -2.5,
    'heavy': -1.5,     'expensive': -2.0,  'poor': -3.0,
    'horrible':-3.5,   'terrible':-3.5,    'weak': -2.5,
    'bad': -3.0,       'buggy': -2.5,
})

# ── POS constants ─────────────────────────────────────────────────────────────
NOUN_TAGS  = {"NN","NNS","NNP","NNPS"}
ADJ_TAGS   = {"JJ","JJR","JJS"}
ADV_TAGS   = {"RB","RBR","RBS"}
VERB_TAGS  = {"VB","VBD","VBG","VBN","VBP","VBZ"}
OPN_TAGS   = ADJ_TAGS | ADV_TAGS | VERB_TAGS
COPULA     = {"is","are","was","were","'s","be"}
NEGATIONS  = {"not","no","never","neither","barely","hardly","without"}

# known verb-nouns that must NOT be treated as features
VERB_NOUNS = {"drains","drain","drops","fluctuates","overheats","overheat",
              "captures","runs","heats","charges"}

# ── clause splitting ──────────────────────────────────────────────────────────
CONJ_RE = re.compile(
    r'\b(but|however|though|although|yet|whereas|while)\b', re.I)
COMMA_RE = re.compile(r',\s*(?=[a-z])', re.I)
AND_RE   = re.compile(r'\band\b', re.I)

def _is_self_contained(s):
    toks = nltk.word_tokenize(s)
    tag  = nltk.pos_tag(toks)
    return (any(t in NOUN_TAGS for _,t in tag) and
            any(t in OPN_TAGS  for _,t in tag))

def split_clauses(text):
    # step 1: split on adversative conjunctions
    parts = CONJ_RE.split(text)
    parts = [p.strip() for p in parts
             if p.strip() and p.strip().lower()
             not in {"but","however","though","although","yet","whereas","while"}]
    # step 2: split on comma
    tmp = []
    for p in parts:
        sub = COMMA_RE.split(p)
        tmp.extend([s.strip() for s in sub if s.strip()])
    parts = tmp
    # step 3: split on 'and' only when both sides self-contained
    final = []
    for part in parts:
        subs = AND_RE.split(part)
        if len(subs) == 1:
            final.append(part)
            continue
        buf = ""
        for s in subs:
            s = s.strip()
            candidate = (buf + " and " + s).strip() if buf else s
            if _is_self_contained(s):
                if buf:
                    final.append(buf)
                buf = s
            else:
                buf = candidate
        if buf:
            final.append(buf)
    return final if final else [text]

# ── noun-phrase extraction ────────────────────────────────────────────────────
def collect_np(tagged, start):
    """
    Walk BACK over adj/noun to build full compound NP,
    then walk FORWARD over nouns.
    Stops walk-back at copula, conjunction, or >3 tokens back.
    """
    pre = start
    limit = max(0, start - 4)
    while pre > limit:
        w, t = tagged[pre - 1]
        if t in NOUN_TAGS | ADJ_TAGS and w.lower() not in COPULA:
            pre -= 1
        else:
            break
    # forward over nouns
    j = start
    while j < len(tagged) and tagged[j][1] in NOUN_TAGS:
        j += 1
    np_tokens = [tagged[k][0] for k in range(pre, j)]
    return np_tokens, pre, j

# ── opinion extraction ────────────────────────────────────────────────────────
def get_opinions(tagged, np_start, np_end):
    """
    Detect copula after NP → grab what follows.
    Else grab OPN tokens immediately after NP.
    Also collect adj immediately before NP (not inside it).
    """
    # check copula within 3 tokens after NP
    cop_idx = None
    for k in range(np_end, min(np_end + 3, len(tagged))):
        if tagged[k][0].lower() in COPULA:
            cop_idx = k
            break

    if cop_idx is not None:
        # "X is/was ADJ ADV" → take adj/adv after copula
        after = []
        for k in range(cop_idx + 1, min(cop_idx + 5, len(tagged))):
            w, t = tagged[k]
            if t in ADJ_TAGS | ADV_TAGS:
                after.append(w)
            elif t in VERB_TAGS and w.lower() not in COPULA:
                after.append(w)
            elif w.lower() in {"too","very","so","quite","rather","extremely"}:
                after.append(w)
            else:
                break
    else:
        after = [tagged[k][0] for k in range(np_end, min(np_end + 4, len(tagged)))
                 if tagged[k][1] in OPN_TAGS
                 and tagged[k][0].lower() not in COPULA]

    return after

# ── sentiment ─────────────────────────────────────────────────────────────────
def get_sentiment(clause_words, opinion_words):
    # use opinion words + clause for VADER
    opinion_text = " ".join(opinion_words)
    clause_text  = " ".join(clause_words)
    # weight opinion text more
    combined = opinion_text + " " + opinion_text + " " + clause_text
    score = SID.polarity_scores(combined)['compound']

    neg = any(w.lower() in NEGATIONS for w in clause_words)
    if neg:
        score = -score

    if score >= 0.05:  return "positive"
    if score <= -0.05: return "negative"
    return "neutral"

# ── per-chunk extraction ──────────────────────────────────────────────────────
def extract_chunk(chunk, seen, aspects):
    tokens = nltk.word_tokenize(chunk)
    tagged = nltk.pos_tag(tokens)

    i = 0
    while i < len(tagged):
        word, tag = tagged[i]

        # skip known verb-nouns mistagged as NN
        if tag in NOUN_TAGS and word.lower() in VERB_NOUNS:
            i += 1
            continue

        if tag not in NOUN_TAGS:
            i += 1
            continue

        np_tokens, pre, j = collect_np(tagged, i)

        # filter: remove trailing verb-nouns from NP
        while np_tokens and np_tokens[-1].lower() in VERB_NOUNS:
            np_tokens.pop()
            j -= 1
        if not np_tokens:
            i = j
            continue

        feature_key = " ".join(np_tokens).lower().strip()
        if len(feature_key) < 2 or feature_key in seen:
            i = j
            continue
        seen.add(feature_key)

        opinions = get_opinions(tagged, pre, j)

        # strip opinion words that are actually part of NP or are copula
        np_lower = {t.lower() for t in np_tokens}
        opinions = [w for w in opinions if w.lower() not in np_lower | COPULA]

        # deduplicate
        seen_op, unique_op = set(), []
        for w in opinions:
            if w.lower() not in seen_op:
                seen_op.add(w.lower())
                unique_op.append(w)
        opinions = unique_op

        if not opinions:
            i = j
            continue

        clause_words = [t for t,_ in tagged]
        aspects.append({
            "feature"  : " ".join(np_tokens),
            "opinion"  : " ".join(opinions),
            "sentiment": get_sentiment(clause_words, opinions)
        })
        i = j

# ── public API ────────────────────────────────────────────────────────────────
def extract_aspects(review):
    aspects, seen = [], set()
    for chunk in split_clauses(review):
        extract_chunk(chunk, seen, aspects)
    return aspects

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    input_file  = "reviews.txt"
    output_dir  = "output"
    output_file = os.path.join(output_dir, "output.json")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reviews = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    print(f"Processing {len(reviews)} reviews...")

    output = []
    for idx, review in enumerate(reviews, 1):
        try:
            aspects = extract_aspects(review)
        except Exception as e:
            print(f"Review {idx} failed: {e}")
            aspects = []
        output.append({
            "id"     : idx,
            "review" : review,
            "aspects": aspects,
            "summary": {
                "total_aspects": len(aspects),
                "positive": sum(1 for a in aspects if a["sentiment"] == "positive"),
                "negative": sum(1 for a in aspects if a["sentiment"] == "negative"),
                "neutral" : sum(1 for a in aspects if a["sentiment"] == "neutral")
            }
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.flush()

    total = sum(r["summary"]["total_aspects"] for r in output)
    print(f"Done -> {output_file}")
    print(f"Reviews      : {len(output)}")
    print(f"Total aspects: {total}")

if __name__ == "__main__":
    main()
