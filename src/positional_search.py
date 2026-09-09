from preprocess import load_corpus, tokenize
from indexer import build_positional_index


def phrase_search(phrase, pos_index):
    """
    Exact phrase search using the positional index.

    Logic: tokenize the phrase the same way as documents (so "cotton shirt"
    becomes ["cotton", "shirt"] after stemming). Then, for a document to match:
      - it must contain ALL the phrase's terms
      - there must exist SOME starting position p such that:
          term1 is at position p, term2 is at position p+1, term3 at p+2 and so on
        i.e. the terms occur back-to-back, in order.

    We only check documents that contain the FIRST term - any doc missing
    it can't possibly match the full phrase, so this avoids scanning docs
    that can never qualify.

    Returns: list of (docid, [matching start positions]), sorted by docid.
    """
    phrase_terms = tokenize(phrase)

    if not phrase_terms or phrase_terms[0] not in pos_index:
        return []  #First word doesn't exist in the corpus

    first_term = phrase_terms[0]
    candidate_docs = pos_index[first_term]["postings"].keys()

    matches = []
    for docid in candidate_docs:
        #Starting positions where the first term occurs in this doc
        start_positions = pos_index[first_term]["postings"][docid]["positions"]

        matching_starts = []
        for start_pos in start_positions:
            is_match = True
            #Checking that every subsequent phrase term sits at start_pos + offset
            for offset, term in enumerate(phrase_terms):
                if term not in pos_index:
                    is_match = False
                    break
                doc_positions = pos_index[term]["postings"].get(docid, {}).get("positions", [])
                if (start_pos + offset) not in doc_positions:
                    is_match = False
                    break
            if is_match:
                matching_starts.append(start_pos)

        if matching_starts:
            matches.append((docid, matching_starts))

    matches.sort(key=lambda pair: pair[0])  #Sorting by doc id
    return matches


def proximity_search(term1, term2, k, pos_index):
    """
    Ordered proximity search: term1 must occur, and term2 must occur within
    k positions AFTER it (this matches the assignment's examples, e.g.
    "cotton WITHIN/3 shirt" - term1 first, term2 within k positions later).

    Logic: for every position p1 where term1 occurs, check if term2 occurs
    at any position p2 such that p1 < p2 <= p1 + k.

    Returns: list of (docid, [(p1, p2), ...]) - the actual matching position
    pairs, sorted by docid. This doubles as the "show matching positions as
    evidence" requirement in Part D.
    """
    stemmed1 = tokenize(term1)
    stemmed2 = tokenize(term2)

    if not stemmed1 or not stemmed2:
        return []
    term1, term2 = stemmed1[0], stemmed2[0]  #Single-word terms expected here

    if term1 not in pos_index or term2 not in pos_index:
        return []  #One of the terms doesn't exist in the corpus

    #Only docs containing both terms can match
    docs1 = set(pos_index[term1]["postings"].keys())
    docs2 = set(pos_index[term2]["postings"].keys())
    common_docs = docs1 & docs2

    matches = []
    for docid in common_docs:
        positions1 = pos_index[term1]["postings"][docid]["positions"]
        positions2 = pos_index[term2]["postings"][docid]["positions"]

        pairs = []
        for p1 in positions1:
            for p2 in positions2:
                if p1 < p2 <= p1 + k:
                    pairs.append((p1, p2))

        if pairs:
            matches.append((docid, pairs))

    matches.sort(key=lambda pair: pair[0])
    return matches

if __name__ == "__main__":
    docs = load_corpus("../data/corpus_100.txt")
    pos_index = build_positional_index(docs)
    titles = {d["docid"]: d["title"] for d in docs}

    #Test phrase search
    print("Phrase search: 'cotton shirt'")
    results = phrase_search("cotton shirt", pos_index)
    for docid, positions in results[:10]:
        print(f"{docid} | {titles[docid]} | matched at positions {positions}")

    #Test proximity search
    print("\nProximity search: 'cotton' WITHIN/3 'shirt'")
    prox_results = proximity_search("cotton", "shirt", 3, pos_index)
    for docid, pairs in prox_results[:10]:
        print(f"{docid} | {titles[docid]} | position pairs {pairs}")