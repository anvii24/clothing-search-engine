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


def build_positional_index(docs):
    """
    Builds the positional index required by Part C:
        term -> { "df": document_frequency,
                   "postings": {docid: {"tf": term_frequency, "positions": [p1, p2, ...]}} }

    Positions are 0-indexed positions WITHIN THE STEMMED TOKEN LIST of that
    document (same tokenize() pipeline as Part A - lowercase, punctuation
    stripped, stopwords removed, stemmed). Position numbers refer to
    positions AFTER stopword removal, which keeps things consistent with
    how the doc is indexed everywhere else in this project.
    """
    index = {}

    for doc in docs:
        docid = doc["docid"]
        tokens = tokenize(doc["text"])  #clean token list as before

        for pos, term in enumerate(tokens):
            if term not in index:
                index[term] = {"df": 0, "postings": {}}

            if docid not in index[term]["postings"]:
                index[term]["postings"][docid] = {"tf": 0, "positions": []}
                index[term]["df"] += 1  #first time we see this term in this doc

            index[term]["postings"][docid]["tf"] += 1
            index[term]["postings"][docid]["positions"].append(pos)

    return index

def print_positional_sample(index, n=5):
    """Prints the first n terms of the positional index, for a sanity check."""
    for i, (term, data) in enumerate(index.items()):
        if i >= n:
            break
        print(f"{term} (df={data['df']}):")
        for docid, info in list(data["postings"].items())[:3]:  #Showing up to 3 docs per term
            print(f"   {docid}: tf={info['tf']}, positions={info['positions']}")


if __name__ == "__main__":
    docs = load_corpus("../data/corpus_100.txt")

    #Part A
    print("Inverted Index (Part A)")
    index = build_inverted_index(docs)

    print("Total unique terms in index:", len(index))
    print("\nSample of index:")
    print_index_sample(index)

    # Quick check on a term we know is common
    print("\nEntry for 'cotton':")
    print(index.get("cotton"))

    #Part C: Positional Index
    print("\nPositional Index (Part C)")
    pos_index = build_positional_index(docs)

    print("Total unique terms in positional index:", len(pos_index))
    print_positional_sample(pos_index)

    print("\nEntry for 'cotton':")
    print(pos_index.get("cotton"))