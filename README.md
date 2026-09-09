# Clothing Search Engine

Information Retrieval Assignment: a small search engine over a 100-document clothing product corpus, implementing ranked retrieval (VSM) and positional (phrase/proximity) search.

**Submitted by: Anvi Gupta (2410110520), Antara Shyam (2410110518)**
**Github Link: https://github.com/anvii24/clothing-search-engine.git**

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

# Test Analysis Results

# Top-10 results

See screenshots in `screenshots/freetext/`, `screenshots/phrase/`, `screenshots/proximity/`, and `screenshots/nonexistent_term/` for the top-10 (or full match) output of every required test query.

# Cases where positional information changes the result set/order:

Case 1: "cotton shirt"

In the free-text (VSM) search, D011 ("Men's Oversized Graphic T-Shirt") and D071 both rank in the top 10 (score 0.2375) because the document contains both "cotton" and "shirt" somewhere in its description VSM only checks co-occurrence, not word order or adjacency.

The exact phrase search for "cotton shirt" excludes both documents entirely. Checking the positions in the positional index, "cotton" and "shirt" appear far apart in these descriptions and never as consecutive tokens, so there is no valid phrase match, even though the free-text score treated them as relevant.

This shows VSM can over-rank documents that merely mention both query words anywhere, while phrase search correctly filters to only documents where the words form the exact expression the user searched for.

Case 2: "regular fit"

In the free-text search, D076, D036, D016, D056, D096 (all "Women's Casual Fit Dress") rank in the top 10 for the query "regular fit" because each description contains the word "fit" (from "casual fit"), and VSM's cosine similarity gives partial term overlap.

The exact phrase search excludes all five. Their positional data shows "fit" is preceded by "casual," not "regular", so position-based matching (checking that "regular" sits immediately before "fit") correctly rejects them, while VSM cannot differentiate between "regular fit" from "casual fit" since it only sees that "fit" is present.
