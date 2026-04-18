"""Download gensim models, filter to top 10k nouns/verbs, save as .kv files."""
import gzip
import os
import shutil
import urllib.request

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
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping.")
        continue
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


# === ConceptNet Numberbatch (English) ===
# Speer, Chin, Havasi (2017), AAAI. Retrofitted embeddings — semantically
# cleaner neighbors than pure distributional vectors. Not on gensim-data, so
# fetched directly. Vocabulary is aligned against glove-wiki-gigaword-300 so
# ordering matches (frequency-sorted via glove), and top-N vocab capping in
# the generator works the same way.
NUMBERBATCH_URL = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"
NUMBERBATCH_KV = "data/conceptnet-numberbatch-en.kv"
NUMBERBATCH_GZ = "data/numberbatch-en-19.08.txt.gz"
NUMBERBATCH_TXT = "data/numberbatch-en-19.08.txt"
NUMBERBATCH_REF = "data/glove-wiki-gigaword-300.kv"

print("\n=== conceptnet-numberbatch-en ===")
if os.path.exists(NUMBERBATCH_KV):
    print(f"{NUMBERBATCH_KV} already exists, skipping.")
else:
    if not os.path.exists(NUMBERBATCH_GZ):
        print(f"Downloading {NUMBERBATCH_URL} (~1GB gzipped)...")
        urllib.request.urlretrieve(NUMBERBATCH_URL, NUMBERBATCH_GZ)
    if not os.path.exists(NUMBERBATCH_TXT):
        print("Decompressing...")
        with gzip.open(NUMBERBATCH_GZ, "rb") as fin, open(NUMBERBATCH_TXT, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    print("Loading numberbatch text...")
    nb_model = KeyedVectors.load_word2vec_format(NUMBERBATCH_TXT, binary=False)

    print(f"Loading reference vocab from {NUMBERBATCH_REF}...")
    ref_model = KeyedVectors.load(NUMBERBATCH_REF)
    words = [w for w in ref_model.index_to_key if w in nb_model.key_to_index]

    vectors = np.array([nb_model[w] for w in words])
    small = KeyedVectors(nb_model.vector_size)
    small.add_vectors(words, vectors)
    small.save(NUMBERBATCH_KV)
    print(f"Saved {len(words)} words to {NUMBERBATCH_KV}")
