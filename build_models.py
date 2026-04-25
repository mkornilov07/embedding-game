"""Download gensim models, filter to top N nouns/verbs, save as .kv files."""
import os

import nltk
import gensim.downloader as api
from gensim.models import KeyedVectors

MAX_VOCAB = 5000
ALLOWED_POS = {"NN", "VB"}

MODELS = [
    "word2vec-google-news-300",
    "glove-wiki-gigaword-50",
    "glove-wiki-gigaword-100",
    "glove-wiki-gigaword-200",
    "glove-wiki-gigaword-300",
    "glove-twitter-25",
    "glove-twitter-50",
    "glove-twitter-100",
    "glove-twitter-200",
    "fasttext-wiki-news-subwords-300",
]

os.makedirs("data", exist_ok=True)

for name in MODELS:
    out_path = f"data/{name}.kv"
    print(f"\n=== {name} ===")
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping.")
        continue

    print("Loading...")
    path = api.load(name, return_path=True)
    full_model = KeyedVectors.load_word2vec_format(
        path, binary=name.startswith("word2vec"), limit=MAX_VOCAB
    )

    candidates = [w for w in full_model.index_to_key if w.isalpha() and len(w) >= 3]
    tags = nltk.pos_tag(candidates)
    filtered = [w for w, tag in tags if tag in ALLOWED_POS]

    small = KeyedVectors(full_model.vector_size)
    small.add_vectors(filtered, full_model[filtered])
    small.save(out_path)
    print(f"Saved {len(filtered)} words to {out_path}")
