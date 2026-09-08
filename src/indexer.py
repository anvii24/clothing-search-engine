from preprocess import load_corpus, tokenize


def build_inverted_index(docs):
    """
    Builds the inverted index required by Part A:
        term -> { "df": document_frequency, "postings": {docid: term_frequency} }

    "postings" is a dict of docid -> tf so lookups are O(1) instead of scanning a list.
    """
    index = {}

    for doc in docs:
        docid = doc["docid"]
        tokens = tokenize(doc["text"])

        # Counting how many times each stemmed term appears in this doc
        term_counts = {}
        for term in tokens:
            term_counts[term] = term_counts.get(term, 0) + 1

        # Merging this docs term counts into the global index
        for term, tf in term_counts.items():
            if term not in index:
                index[term] = {"df": 0, "postings": {}}
            index[term]["postings"][docid] = tf
            index[term]["df"] += 1  # one more doc contains this term

    return index


def print_index_sample(index, n=10):
    """Prints the first n terms of the index, for a quick sanity check."""
    for i, (term, data) in enumerate(index.items()):
        if i >= n:
            break
        print(f"{term} (df={data['df']}): {data['postings']}")


if __name__ == "__main__":
    docs = load_corpus("../data/corpus_100.txt")
    index = build_inverted_index(docs)

    print("Total unique terms in index:", len(index))
    print("\nSample of index:")
    print_index_sample(index)

    # Quick check on a term we know is common
    print("\nEntry for 'cotton':")
    print(index.get("cotton"))