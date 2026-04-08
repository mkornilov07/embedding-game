"""Download gensim models, filter to top 10k nouns/verbs, save as .kv files."""
import nltk
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors

MAX_VOCAB = 10000
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

for name in MODELS:
    out_path = f"data/{name}.kv"
    print(f"\n=== {name} ===")
    print("Loading...")
    full_model = api.load(name)

    candidates = [w for w in full_model.index_to_key[:MAX_VOCAB] if w.isalpha() and len(w) >= 3]
    tags = nltk.pos_tag(candidates)
    filtered = [w for w, tag in tags if tag in ALLOWED_POS]

    vectors = np.array([full_model[w] for w in filtered])
    small = KeyedVectors(full_model.vector_size)
    small.add_vectors(filtered, vectors)
    small.save(out_path)
    print(f"Saved {len(filtered)} words to {out_path}")
