# Clothing Search Engine

Information Retrieval Assignment: a small search engine over a 100-document clothing product corpus, implementing ranked retrieval (VSM) and positional (phrase/proximity) search.

# Setup
bash
pip install nltk streamlit
python -c "import nltk; nltk.download('stopwords')"

# How to run

**Terminal interface:**
bash
cd src
python main.py

**Web interface:**
bash
cd src
streamlit run app.py

**Generate index output files** (writes to `output/`):
bash
cd src
python generate_outputs.py

# Part A: Pre-processing and stop-word policy

Text is lowercased, punctuation is stripped (replaced with spaces so words don't merge together), and each token is stemmed using NLTK's Porter Stemmer. Stop-word removal uses **NLTK's standard English stopword list** (`nltk.corpus.stopwords`), applied identically to every document and every query- this consistency is what makes document and query vectors comparable in Part B.


# Part B: Vector Space Model

- Document term weight: `1 + log10(tf)` - no idf.
- Query term weight: `(1 + log10(tf)) * log10(N/df)`.
- Both vectors are cosine-normalized; ranking is by cosine similarity, ties broken by ascending document ID.

# Part C: Positional index

Every posting stores the exact token positions of a term within a document (after the same tokenize/stem/stopword pipeline as Part A). This supports:
- **Exact phrase search** - terms must appear at consecutive positions, in order.
- **Ordered proximity search** - `term1 WITHIN/k term2` — term2 must occur within k positions after term1.

# Novelty additions

1. **Fuzzy/typo-tolerant matching** - if a query term isn't in the corpus vocabulary, `difflib.get_close_matches` finds the closest real term (similarity cutoff 0.75) and substitutes it, so typos like "coton shrt" still return relevant results.
2. **Streamlit web interface** - a styled browser-based UI (`app.py`) as an alternative to the terminal interface, with the same underlying search logic.

# Part E: Testing

All mandatory test queries (10 free-text, 5 phrase, 3 proximity with different k values, 1 nonexistent-term query) were run manually through both interfaces; results are captured in `screenshots/`.

# Part E — Test Analysis

# Top-10 results

See screenshots in `screenshots/freetext/`, `screenshots/phrase/`, `screenshots/proximity/`, and `screenshots/nonexistent_term/` for the top-10 (or full match) output of every required test query.

# Cases where positional information changes the result set/order:
