"""Build the curated ConceptNet Numberbatch .kv files used by the site."""
from itertools import islice
from pathlib import Path

import nltk
import numpy as np
from gensim.models import KeyedVectors

from game.embeddings import apply_abtt

DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_WORD_LIMIT = 100_000
VOCAB_SIZES = (3_000, 10_000)
ALLOWED_POS = {"NN", "VB"}
PROFANITIES = {
    "asshole", "bastard", "bitch", "bullshit", "cock", "cocksucker",
    "cunt", "dick", "fag", "faggot", "fuck", "fucker", "motherfucker",
    "nigga", "niggas", "nigger", "piss", "shit", "shitty", "slut", "whore",
}


def filter_words(words):
    """Keep nouns and verbs in their original frequency order."""
    candidates = [word for word in words if word.isalpha() and len(word) >= 3]
    tagged = nltk.pos_tag_sents([[word] for word in candidates])
    return [
        sentence[0][0] for sentence in tagged
        if sentence[0][1] in ALLOWED_POS and sentence[0][0] not in PROFANITIES
    ]


def build_conceptnet_vocabs():
    with (DATA_DIR / "glove.6B.300d.txt").open(encoding="utf-8") as source:
        ranked_words = [line.partition(" ")[0] for line in islice(source, SOURCE_WORD_LIMIT)]
    wanted = set(filter_words(ranked_words))

    vectors = {}
    with (DATA_DIR / "numberbatch-en-19.08.txt").open(encoding="utf-8") as source:
        vector_size = int(next(source).split()[1])
        for line in source:
            word, _, values = line.partition(" ")
            if word in wanted:
                vectors[word] = np.fromstring(values, sep=" ", dtype=np.float32)

    selected = [word for word in ranked_words if word in vectors]
    if len(selected) < max(VOCAB_SIZES):
        raise ValueError(f"Only {len(selected)} matching words; increase SOURCE_WORD_LIMIT.")

    for size in VOCAB_SIZES:
        if size == 3_000:
            name = "conceptnet-numberbatch-en-3k"
        else:
            name = f"conceptnet-numberbatch-en-{size // 1000}k"
        model = KeyedVectors(vector_size=vector_size)
        model.add_vectors(selected[:size], np.stack([vectors[word] for word in selected[:size]]))
        model.save(str(DATA_DIR / f"{name}.kv"))
        apply_abtt(model).save(str(DATA_DIR / f"{name}-abtt.kv"))
        print(f"Saved {size} ConceptNet vectors and ABTT sidecar for data/{name}.kv")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    build_conceptnet_vocabs()
