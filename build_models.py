"""Build filtered embedding models and their ABTT-processed sidecars."""
import argparse
from itertools import islice
from pathlib import Path

import nltk
import gensim.downloader as api
import numpy as np
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
CHAIN_SOURCE_LIMIT = 100_000
CHAIN_VOCAB_SIZES = (10_000, 20_000)

def filter_words(words):
    """Tag each word in isolation, preserving its frequency order."""
    candidates = [word for word in words if word.isalpha() and len(word) >= 3]
    tagged = nltk.pos_tag_sents([[word] for word in candidates])
    return [sentence[0][0] for sentence in tagged if sentence[0][1] in ALLOWED_POS]


def build_chain_vocabs():
    with (DATA_DIR / "glove.6B.300d.txt").open(encoding="utf-8") as source:
        words = [line.partition(" ")[0] for line in islice(source, CHAIN_SOURCE_LIMIT)]
    filtered = filter_words(words)
    wanted = set(filtered)

    vectors = {}
    with (DATA_DIR / "numberbatch-en-19.08.txt").open(encoding="utf-8") as source:
        vector_size = int(next(source).split()[1])
        for line in source:
            word, _, values = line.partition(" ")
            if word in wanted:
                vectors[word] = np.fromstring(values, sep=" ", dtype=np.float32)

    selected = [word for word in filtered if word in vectors][:max(CHAIN_VOCAB_SIZES)]
    if len(selected) < max(CHAIN_VOCAB_SIZES):
        raise ValueError(f"Only {len(selected)} matching words; increase CHAIN_SOURCE_LIMIT.")

    for size in CHAIN_VOCAB_SIZES:
        name = f"conceptnet-numberbatch-en-{size // 1000}k"
        model = KeyedVectors(vector_size=vector_size)
        model.add_vectors(selected[:size], np.stack([vectors[word] for word in selected[:size]]))
        model.save(str(DATA_DIR / f"{name}.kv"))
        apply_abtt(model).save(str(DATA_DIR / f"{name}-abtt.kv"))
        print(f"Saved {size} words to {name}.kv and {name}-abtt.kv")


def build_models():
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

            filtered = filter_words(full_model.index_to_key)
            small = KeyedVectors(full_model.vector_size)
            small.add_vectors(filtered, full_model[filtered])
            small.save(str(out_path))
            print(f"Saved {len(filtered)} words to {out_path}")

    # Process every available base model, including ConceptNet. Sidecars aren't model-list entries.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-vocabs", action="store_true", help="Rebuild the 10k/20k ConceptNet chain vocabularies")
    args = parser.parse_args()
    if args.chain_vocabs:
        build_chain_vocabs()
    else:
        build_models()
