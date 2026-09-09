import streamlit as st
from preprocess import load_corpus
from indexer import build_inverted_index, build_positional_index
from vsm_search import build_document_weights, search
from positional_search import phrase_search, proximity_search

st.set_page_config(page_title="Clothing Search Engine", page_icon="◆", layout="centered")


@st.cache_resource
def load_everything():
    """
    Loads the corpus and builds every index/structure needed.
    @st.cache_resource means this only runs ONCE per app session, not on
    every interaction.
    """
    docs = load_corpus("../data/corpus_100.txt")
    index = build_inverted_index(docs)
    pos_index = build_positional_index(docs)
    doc_weights, doc_norms = build_document_weights(index)
    titles = {d["docid"]: (d["title"], d["category"]) for d in docs}
    return index, pos_index, doc_weights, doc_norms, titles


index, pos_index, doc_weights, doc_norms, titles = load_everything()


# CUSTOM STYLING

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --bg: #0B0B12;
    --surface: #15141F;
    --border: #2A2838;
    --text: #F1EEFA;
    --muted: #8D8AA3;
    --pink: #FF5C8A;
    --cyan: #5CE1E6;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg);
    background-image:
        radial-gradient(circle, rgba(255,255,255,0.15) 1.5px, transparent 1.5px);
    background-size: 24px 24px;
    color: var(--text);
    font-family: 'Manrope', sans-serif;
}

/* Headline block */
.app-header { margin-bottom: 2.2rem; }
.app-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.05;
}
.app-header .accent-bar {
    height: 4px;
    width: 64px;
    margin: 14px 0 16px 0;
    background: linear-gradient(90deg, var(--pink), var(--cyan));
    border-radius: 2px;
}
.app-header p {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0;
}

/* Mode switcher -> pill tabs instead of default radio bullets */
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 8px 18px !important;
    transition: all 0.15s ease;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, var(--pink), var(--cyan));
    border-color: transparent;
}
div[data-testid="stRadio"] label:has(input:checked) p {
    color: #0B0B12 !important;
    font-weight: 600 !important;
}
div[data-testid="stRadio"] label [data-baseweb="radio"] { display: none; }
div[data-testid="stRadio"] label p {
    color: var(--text);
    margin: 0;
    font-family: 'Manrope', sans-serif;
}

/* Text + number inputs -> minimal underline style */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: none !important;
    border-bottom: 2px solid var(--border) !important;
    border-radius: 6px 6px 0 0 !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.05rem !important;
    padding: 12px 14px !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-bottom: 2px solid var(--pink) !important;
    box-shadow: none !important;
}

/* Button -> gradient pill */
div[data-testid="stButton"] button {
    background: linear-gradient(90deg, var(--pink), var(--cyan));
    color: #0B0B12;
    font-weight: 600;
    border: none;
    border-radius: 999px;
    padding: 10px 28px;
    font-family: 'Manrope', sans-serif;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.88;
}

/* Section labels */
.field-label {
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 6px;
}

/* Result rows - catalog style, not a spreadsheet table */
.result-row {
    border-top: 1px solid var(--border);
    padding: 14px 0;
    display: flex;
    align-items: center;
    gap: 14px;
}
.result-docid {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: var(--cyan);
    width: 52px;
    flex-shrink: 0;
}
.result-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: var(--muted);
    flex-shrink: 0;
}
.result-title {
    flex-grow: 1;
    font-size: 0.95rem;
}
.result-score-bar {
    height: 5px;
    border-radius: 3px;
    background: var(--border);
    width: 90px;
    flex-shrink: 0;
    overflow: hidden;
}
.result-score-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--pink), var(--cyan));
}
.result-positions {
    font-size: 0.8rem;
    color: var(--muted);
    flex-shrink: 0;
}

/* Notice banners - replace default st.info/st.warning boxes */
.notice {
    border-left: 3px solid var(--cyan);
    background: var(--surface);
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    color: var(--text);
}
.notice.warning { border-left-color: var(--pink); color: var(--muted); }
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown(
    '<div class="app-header">'
    '<h1>Clothing search engine</h1>'
    '<div class="accent-bar"></div>'
    '</div>',
    unsafe_allow_html=True
)

mode = st.radio(
    "Search mode",
    ["Free-text", "Exact phrase", "Proximity"],
    label_visibility="collapsed",
    horizontal=True,
)

st.write("")  # small spacer


def render_notice(text, warning=False):
    css_class = "notice warning" if warning else "notice"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def render_results(rows, score_key=None, position_key=None):
    """
    Renders result rows in the catalog style defined above, instead of
    st.table(). Each row is docID + category chip + title, plus either
    a score bar (free-text mode) or matched positions (phrase/proximity).

    """
    max_score = max((r.get(score_key, 0) for r in rows), default=1) if score_key else 1

    for r in rows:
        docid, title, category = r["docid"], r["title"], r["category"]

        if score_key:
            pct = int((r[score_key] / max_score) * 100) if max_score else 0
            extra_html = (
                f'<div class="result-score-bar"><div class="result-score-fill" style="width:{pct}%"></div></div>'
                f'<div class="result-positions">{r[score_key]:.4f}</div>'
            )
        elif position_key:
            extra_html = f'<div class="result-positions">{r[position_key]}</div>'
        else:
            extra_html = ""

        row_html = (
            f'<div class="result-row">'
            f'<div class="result-docid">{docid}</div>'
            f'<div class="result-chip">{category}</div>'
            f'<div class="result-title">{title}</div>'
            f'{extra_html}'
            f'</div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)

# MODES
if mode == "Free-text":
    st.markdown('<div class="field-label">query</div>', unsafe_allow_html=True)
    query = st.text_input("", placeholder="e.g. cotton shirt", label_visibility="collapsed")

    if st.button("Search") and query:
        results, corrections = search(query, index, doc_weights, doc_norms)

        if corrections:
            correction_str = ", ".join(f"'{k}' → '{v}'" for k, v in corrections.items())
            render_notice(f"Corrected: {correction_str}")

        if not results:
            render_notice("No matching documents found.", warning=True)
        else:
            rows = [{"docid": docid, "title": titles[docid][0], "category": titles[docid][1], "score": score}
                    for docid, score in results]
            render_results(rows, score_key="score")

elif mode == "Exact phrase":
    st.markdown('<div class="field-label">phrase</div>', unsafe_allow_html=True)
    phrase = st.text_input("", placeholder="e.g. cotton shirt", label_visibility="collapsed")

    if st.button("Search") and phrase:
        results = phrase_search(phrase, pos_index)

        if not results:
            render_notice(f"No documents contain the exact phrase '{phrase}'.", warning=True)
        else:
            rows = [{"docid": docid, "title": titles[docid][0], "category": titles[docid][1], "positions": positions}
                    for docid, positions in results]
            render_results(rows, position_key="positions")

else:  # Proximity
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="field-label">term 1</div>', unsafe_allow_html=True)
        term1 = st.text_input("", key="t1", label_visibility="collapsed")
    with col2:
        st.markdown('<div class="field-label">term 2</div>', unsafe_allow_html=True)
        term2 = st.text_input("", key="t2", label_visibility="collapsed")
    with col3:
        st.markdown('<div class="field-label">k</div>', unsafe_allow_html=True)
        k = st.number_input("", min_value=1, value=3, step=1, label_visibility="collapsed")

    if st.button("Search") and term1 and term2:
        results = proximity_search(term1, term2, int(k), pos_index)

        if not results:
            render_notice(f"No documents have '{term1}' within {k} positions of '{term2}'.", warning=True)
        else:
            rows = [{"docid": docid, "title": titles[docid][0], "category": titles[docid][1], "positions": pairs}
                    for docid, pairs in results]
            render_results(rows, position_key="positions")