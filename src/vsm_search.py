import math
import difflib
from preprocess import load_corpus, tokenize
from indexer import build_inverted_index

N_DOCS = 100

def build_document_weights(index):
    """
    Computes the 'lnc' side of the scheme for every document:
      - raw weight per term = 1 + log10(tf)   [log-tf, NO idf]
      - norm per document   = sqrt(sum of squared raw weights)  [for cosine normalization]
    """
    doc_weights = {}
    doc_sumsq = {}

    for term, data in index.items():
        for docid, tf in data["postings"].items():
            weight = 1 + math.log10(tf)

            doc_weights.setdefault(docid, {})[term] = weight
            doc_sumsq[docid] = doc_sumsq.get(docid, 0) + weight ** 2

    doc_norms = {docid: math.sqrt(sumsq) for docid, sumsq in doc_sumsq.items()}

    return doc_weights, doc_norms


def fuzzy_match(term, vocabulary, cutoff=0.75):
    """
    Feature: typo-tolerant matching. If 'term' isn't in the corpus vocabulary
    at all, look for the closest real term using difflib's string similarity
    (based on character-level overlap, not meaning). 

    cutoff=0.75 means: only accept a match if it's at least 75% similar -
    this avoids matching completely unrelated words just because they share
    a couple letters.

    Returns the closest matching real term, or None if nothing is close enough.
    """
    matches = difflib.get_close_matches(term, vocabulary, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def build_query_vector(query, index, use_fuzzy=True):
    """
    Computes the 'ltc' side of the scheme for a single query string.

    If use_fuzzy=True, any query term not found in the corpus vocabulary
    gets a fuzzy-matched substitute (if one exists above the similarity
    cutoff) before being skipped. It lets "coton" resolve to "cotton", 
    "shrt" resolve to "shirt", etc.

    Returns: (normalized_weights_dict, corrections_dict)
      corrections_dict maps original_term -> corrected_term, so the caller
      can display "did you mean 'cotton'?" style feedback.
    """
    tokens = tokenize(query)

    term_counts = {}
    for term in tokens:
        term_counts[term] = term_counts.get(term, 0) + 1

    raw_weights = {}
    corrections = {}
    vocabulary = list(index.keys())  #all real terms in the corpus

    for term, tf in term_counts.items():
        actual_term = term

        if term not in index:
            if use_fuzzy:
                match = fuzzy_match(term, vocabulary)
                if match:
                    actual_term = match
                    corrections[term] = match
                else:
                    continue  #no close match found: truly unknown term, skip
            else:
                continue

        df = index[actual_term]["df"]
        idf = math.log10(N_DOCS / df)
        raw_weights[actual_term] = raw_weights.get(actual_term, 0) + (1 + math.log10(tf)) * idf

    norm = math.sqrt(sum(w ** 2 for w in raw_weights.values()))
    if norm == 0:
        return {}, corrections

    normalized = {term: w / norm for term, w in raw_weights.items()}
    return normalized, corrections

def search(query, index, doc_weights, doc_norms, top_k=10, use_fuzzy=True):
    """
    Runs a free-text query and returns the top_k ranked documents.

    Returns: (results, corrections)
      results = list of (docid, score), sorted by score desc, then docid asc
      corrections = {original_typo: corrected_term} - empty dict if none applied
    """
    query_vec, corrections = build_query_vector(query, index, use_fuzzy=use_fuzzy)

    scores = {}

    for term, q_weight in query_vec.items():
        postings = index[term]["postings"]
        for docid in postings:
            d_weight = doc_weights[docid][term]
            scores[docid] = scores.get(docid, 0) + (q_weight * d_weight)

    final_scores = []
    for docid, dot_product in scores.items():
        cosine_sim = dot_product / doc_norms[docid]
        final_scores.append((docid, cosine_sim))

    final_scores.sort(key=lambda pair: (-pair[1], pair[0]))

    return final_scores[:top_k], corrections


if __name__ == "__main__":
    docs = load_corpus("../data/corpus_100.txt")
    index = build_inverted_index(docs)
    doc_weights, doc_norms = build_document_weights(index)

    titles = {d["docid"]: d["title"] for d in docs}

    # Normal query (no typos
    test_query = "cotton shirt"
    results, corrections = search(test_query, index, doc_weights, doc_norms)
    print(f"Query: '{test_query}'")
    if corrections:
        print(f"(Corrected: {corrections})")
    print(f"Top {len(results)} results:\n")
    for docid, score in results:
        print(f"{docid} | score={score:.4f} | {titles[docid]}")

    # Typo'd query
    print("\n" + "=" * 50)
    typo_query = "coton shrt"
    results2, corrections2 = search(typo_query, index, doc_weights, doc_norms)
    print(f"Query: '{typo_query}'")
    if corrections2:
        print(f"(Corrected: {corrections2})")
    print(f"Top {len(results2)} results:\n")
    for docid, score in results2:
        print(f"{docid} | score={score:.4f} | {titles[docid]}")