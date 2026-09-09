from preprocess import load_corpus
from indexer import build_inverted_index, build_positional_index
from vsm_search import build_document_weights, search
from positional_search import phrase_search, proximity_search


def load_everything():
    """
    Loads the corpus and builds every index/structure needed, once, at startup.
    Doing this once here, keeps the app fast.
    """
    docs = load_corpus("../data/corpus_100.txt")
    index = build_inverted_index(docs)
    pos_index = build_positional_index(docs)
    doc_weights, doc_norms = build_document_weights(index)
    titles = {d["docid"]: (d["title"], d["category"]) for d in docs}

    return index, pos_index, doc_weights, doc_norms, titles


def run_free_text_search(query, index, doc_weights, doc_norms, titles):
    """
    Part B mode: ranked free-text search using VSM cosine similarity.
    Also surfaces fuzzy-match corrections if any typo terms were fixed.
    """
    results, corrections = search(query, index, doc_weights, doc_norms)

    if corrections:
        correction_str = ", ".join(f"'{k}' -> '{v}'" for k, v in corrections.items())
        print(f"\n(Note: applied fuzzy correction: {correction_str})")

    if not results:
        print("\nNo matching documents found.")
        return

    print(f"\nTop {len(results)} results for: '{query}'\n")
    print(f"{'DocID':<8}{'Score':<10}{'Category':<15}{'Title'}")
    print("-" * 70)
    for docid, score in results:
        title, category = titles[docid]
        print(f"{docid:<8}{score:<10.4f}{category:<15}{title}")


def run_phrase_search(phrase, pos_index, titles):
    """
    Part C mode: exact phrase search using the positional index.
    Prints the matching positions too, as evidence the positional index is
    actually driving the match.
    """
    results = phrase_search(phrase, pos_index)

    if not results:
        print(f"\nNo documents contain the exact phrase: '{phrase}'")
        return

    print(f"\n{len(results)} document(s) contain the exact phrase: '{phrase}'\n")
    print(f"{'DocID':<8}{'Category':<15}{'Title':<45}{'Positions'}")
    print("-" * 100)
    for docid, positions in results:
        title, category = titles[docid]
        print(f"{docid:<8}{category:<15}{title:<45}{positions}")


def run_proximity_search(term1, term2, k, pos_index, titles):
    """
    Part C mode: ordered proximity search - term1 must occur, then term2
    within k positions after it. Prints the actual matching position pairs
    as evidence.
    """
    results = proximity_search(term1, term2, k, pos_index)

    if not results:
        print(f"\nNo documents have '{term1}' within {k} positions of '{term2}'")
        return

    print(f"\n{len(results)} document(s) have '{term1}' within {k} positions before '{term2}'\n")
    print(f"{'DocID':<8}{'Category':<15}{'Title':<45}{'Position pairs (term1, term2)'}")
    print("-" * 100)
    for docid, pairs in results:
        title, category = titles[docid]
        print(f"{docid:<8}{category:<15}{title:<45}{pairs}")


def main():
    print("Loading corpus and building indexes...")
    index, pos_index, doc_weights, doc_norms, titles = load_everything()
    print("Ready.\n")

    while True:
        print("=" * 70)
        print("Clothing Search Engine")
        print("1. Free-text search (ranked, VSM)")
        print("2. Exact phrase search")
        print("3. Proximity search (word1 WITHIN/k word2)")
        print("4. Exit")
        choice = input("\nChoose an option (1-4): ").strip()

        if choice == "1":
            query = input("Enter free-text query: ").strip()
            run_free_text_search(query, index, doc_weights, doc_norms, titles)

        elif choice == "2":
            phrase = input("Enter exact phrase (e.g. 'cotton shirt'): ").strip()
            run_phrase_search(phrase, pos_index, titles)

        elif choice == "3":
            term1 = input("Enter first term: ").strip()
            term2 = input("Enter second term: ").strip()
            k = input("Enter k (max positions apart): ").strip()
            try:
                k = int(k)
                run_proximity_search(term1, term2, k, pos_index, titles)
            except ValueError:
                print("k must be a whole number.")

        elif choice == "4":
            print("Goodbye.")
            break

        else:
            print("Invalid choice, try again.")

        print() 


if __name__ == "__main__":
    main()