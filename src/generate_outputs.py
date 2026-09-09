import os
from preprocess import load_corpus
from indexer import build_inverted_index, build_positional_index

OUTPUT_DIR = "../output"


def dump_inverted_index(index, filepath):
    """
    Writes the full Part A inverted index to a text file, one term per line:
        term (df=X): {docid: tf, docid: tf, ...}
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"INVERTED INDEX - {len(index)} unique terms\n")
        f.write("=" * 60 + "\n\n")
        for term in sorted(index.keys()):
            data = index[term]
            f.write(f"{term} (df={data['df']}): {data['postings']}\n")


def dump_positional_index(pos_index, filepath):
    """
    Writes the full Part C positional index to a text file, one term per line,
    with each document's tf and positions shown.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"POSITIONAL INDEX - {len(pos_index)} unique terms\n")
        f.write("=" * 60 + "\n\n")
        for term in sorted(pos_index.keys()):
            data = pos_index[term]
            f.write(f"{term} (df={data['df']}):\n")
            for docid, info in sorted(data["postings"].items()):
                f.write(f"    {docid}: tf={info['tf']}, positions={info['positions']}\n")
            f.write("\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    docs = load_corpus("../data/corpus_100.txt")
    index = build_inverted_index(docs)
    pos_index = build_positional_index(docs)

    dump_inverted_index(index, os.path.join(OUTPUT_DIR, "inverted_index.txt"))
    print("Wrote inverted_index.txt")

    dump_positional_index(pos_index, os.path.join(OUTPUT_DIR, "positional_index.txt"))
    print("Wrote positional_index.txt")

    print("\nBoth files generated in:", os.path.abspath(OUTPUT_DIR))