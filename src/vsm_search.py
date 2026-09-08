import math
from preprocess import load_corpus, tokenize
from indexer import build_inverted_index

N_DOCS = 100

def build_document_weights(index):
    """
    Computes the 'lnc' side of the scheme for every document:
      - raw weight per term = 1 + log10(tf)   [log-tf, NO idf]
      - norm per document   = sqrt(sum of squared raw weights)  [for cosine normalization]

    We store RAW weights (not yet divided by norm) plus the norm separately,
    so at scoring time we can do: (raw_weight / norm) without recomputing
    log10 repeatedly for every query.

    Returns:
      doc_weights: {docid: {term: raw_weight}}
      doc_norms:   {docid: norm}
    """
    doc_weights = {}
    doc_sumsq = {}  # running sum of squared weights per doc, used to build the norm

    for term, data in index.items():
        for docid, tf in data["postings"].items():
            weight = 1 + math.log10(tf)

            doc_weights.setdefault(docid, {})[term] = weight
            doc_sumsq[docid] = doc_sumsq.get(docid, 0) + weight ** 2

    # Converting summed squares into actual norms (sqrt), one per document
    doc_norms = {docid: math.sqrt(sumsq) for docid, sumsq in doc_sumsq.items()}

    return doc_weights, doc_norms


def build_query_vector(query, index):
    """
    Computes the 'ltc' side of the scheme for a single query string:
      - tokenize the query the SAME way documents were tokenized (fair comparison)
      - raw weight per term = (1 + log10(tf)) * log10(N / df)   [log-tf WITH idf]
      - normalize the whole query vector to unit length (cosine normalization)

    If a query term never appears in the corpus, it has no df and contributes
    nothing - it's simply skipped, which is exactly what should happen
    when e.g. testing a nonsense/absent term (Part E requirement).

    Returns: {term: normalized_weight}
    """
    tokens = tokenize(query)

    # Counting term frequency within the query itself
    term_counts = {}
    for term in tokens:
        term_counts[term] = term_counts.get(term, 0) + 1

    raw_weights = {}
    for term, tf in term_counts.items():
        if term not in index:
            continue  # if term doesn't exist anywhere in the corpus, skip it

        df = index[term]["df"]
        idf = math.log10(N_DOCS / df)
        raw_weights[term] = (1 + math.log10(tf)) * idf

    # Cosine-normalizing the query vector
    norm = math.sqrt(sum(w ** 2 for w in raw_weights.values()))
    if norm == 0:
        return {}  # no recognizable terms: query matches nothing

    normalized = {term: w / norm for term, w in raw_weights.items()}
    return normalized


def search(query, index, doc_weights, doc_norms, top_k=10):
    """
    Runs a free-text query and returns the top_k ranked documents.

    Approach: cosine similarity = dot product of two normalized vectors.
    We only need to look at documents that share at least one term with the
    query (any document with zero shared terms has similarity 0 anyway),
    so we loop over query terms -> their postings, not the whole corpus.

    Returns: list of (docid, score), sorted by score desc, then docid asc.
    """
    query_vec = build_query_vector(query, index)

    scores = {}  # docid: running dot product 

    for term, q_weight in query_vec.items():
        postings = index[term]["postings"]  # {docid: tf} for docs containing this term
        for docid in postings:
            d_weight = doc_weights[docid][term]  # raw document weight for this term
            scores[docid] = scores.get(docid, 0) + (q_weight * d_weight)

    # Dividing each accumulated score by that document's norm to finish the cosine calc
    final_scores = []
    for docid, dot_product in scores.items():
        cosine_sim = dot_product / doc_norms[docid]
        final_scores.append((docid, cosine_sim))

    # Sorting from highest similarity first; ties broken by smaller docid first
    final_scores.sort(key=lambda pair: (-pair[1], pair[0]))

    return final_scores[:top_k]


if __name__ == "__main__":
    docs = load_corpus("../data/corpus_100.txt")
    index = build_inverted_index(docs)
    doc_weights, doc_norms = build_document_weights(index)

    # Building a quick docid: title lookup for readable output
    titles = {d["docid"]: d["title"] for d in docs}

    test_query = "cotton shirt"
    results = search(test_query, index, doc_weights, doc_norms)

    print(f"Query: '{test_query}'")
    print(f"Top {len(results)} results:\n")
    for docid, score in results:
        print(f"{docid} | score={score:.4f} | {titles[docid]}")