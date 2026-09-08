import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

# Reused across the whole pipeline so we don't recreate them every call
STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


def load_corpus(filepath):
    """
    Parses the clothing corpus file into a list of document dicts.
    Each dict has: docid, category, title, text
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    docs = []
    doc_blocks = re.findall(r"<DOC>(.*?)</DOC>", raw, re.DOTALL)

    for block in doc_blocks:
        docid = re.search(r"<DOCID>(.*?)</DOCID>", block, re.DOTALL).group(1).strip()
        category = re.search(r"<CATEGORY>(.*?)</CATEGORY>", block, re.DOTALL).group(1).strip()
        title = re.search(r"<TITLE>(.*?)</TITLE>", block, re.DOTALL).group(1).strip()
        text = re.search(r"<TEXT>(.*?)</TEXT>", block, re.DOTALL).group(1).strip()

        docs.append({
            "docid": docid,
            "category": category,
            "title": title,
            "text": text
        })

    return docs


def tokenize(text):
    """
    Converts raw text into a clean list of stemmed tokens, with stopwords removed.

    Pipeline:
      1. Lowercase everything
      2. Strip punctuation (replace with a space so words don't get glued together,
         e.g. "T-Shirt" -> "t shirt", not "tshirt")
      3. Split on whitespace to get raw word tokens
      4. Drop stopwords (common words with no search value)
      5. Stem each remaining word to its root form

    NOTE: this function does NOT track positions - it just returns the clean
    token list. We track positions separately in the indexer for Part C.
    """
    # Lowercasing
    text = text.lower()

    # Removing punctuation by replacing each punctuation char with a space
    for punct in string.punctuation:
        text = text.replace(punct, " ")

    # Spliting into raw tokens on whitespace
    raw_tokens = text.split()

    # Removing stopwords, then stemming what's left
    clean_tokens = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue  # skip stopwords entirely - they never enter the index
        stemmed = STEMMER.stem(tok)
        clean_tokens.append(stemmed)

    return clean_tokens


if __name__ == "__main__":
    docs = load_corpus("../data/corpus_100.txt")
    print("Total documents loaded:", len(docs))

    # Checking the tokenizer on the first document
    sample_tokens = tokenize(docs[0]["text"])
    print("\nSample tokenized text (doc D001):")
    print(sample_tokens)