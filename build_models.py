"""Download/filter embedding models and save their ABTT-processed sidecars."""
from pathlib import Path

import nltk
import gensim.downloader as api
from gensim.models import KeyedVectors

from game.embeddings import apply_abtt
from game.generators import AVAILABLE_MODELS

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

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

for name in MODELS:
    out_path = DATA_DIR / f"{name}.kv"
    print(f"\n=== {name} ===")
    if out_path.exists():
        print(f"{out_path} already exists, skipping.")
    else:
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
        small.save(str(out_path))
        print(f"Saved {len(filtered)} words to {out_path}")

# Process every available base model, including models (such as ConceptNet)
# that aren't downloaded by this script. Sidecars aren't model-list entries.
for name in AVAILABLE_MODELS:
    base_path = DATA_DIR / f"{name}.kv"
    if not base_path.exists():
        continue
    abtt_path = DATA_DIR / f"{name}-abtt.kv"
    if abtt_path.exists():
        print(f"{abtt_path} already exists, skipping.")
        continue

    print(f"Creating ABTT vectors for {name}...")
    model = KeyedVectors.load(str(base_path))
    apply_abtt(model).save(str(abtt_path))
    print(f"Saved ABTT vectors to {abtt_path}")
