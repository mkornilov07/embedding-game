"""Build a 5,000-word embedding model from conversational English frequencies."""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from gensim.models import KeyedVectors

from build_models import filter_words
from game.embeddings import apply_abtt
from fetch_openai_embeddings import MODEL, fetch_embedding


ROOT = Path(__file__).resolve().parent
DEFAULT_FREQUENCY_LIST = ROOT / "data" / "SUBTLEXus74286wordstextversion.tsv"
DEFAULT_EXISTING_EMBEDDINGS = ROOT / "data" / f"openai-{MODEL}-5000.jsonl"
DEFAULT_CACHE = ROOT / "data" / f"openai-{MODEL}-subtlex-us-5000.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / f"openai-{MODEL}-subtlex-us-5000.kv"
DEFAULT_ABTT_OUTPUT = ROOT / "data" / f"openai-{MODEL}-subtlex-us-5000-abtt.kv"
MAX_DIMENSIONS = 3072


def conversational_words(frequency_path, limit):
    """Return the highest-frequency words that pass the game's POS/shape filters."""
    counts = Counter()
    with frequency_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            word = row["Word"].strip().lower()
            if word.isalpha() and len(word) >= 3:
                counts[word] += int(row["FREQcount"])

    ranked = [word for word, _ in counts.most_common()]
    eligible = filter_words(ranked)
    if len(eligible) < limit:
        raise ValueError(f"Found only {len(eligible)} eligible words; need {limit}.")
    return eligible[:limit]


def read_embeddings(path, wanted_words, dimensions):
    embeddings = {}
    if not path.exists():
        return embeddings
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            word = record["word"]
            vector = record.get("embedding", [])
            if (record.get("model") == MODEL and word in wanted_words
                    and len(vector) == dimensions):
                embeddings[word] = vector
    return embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-list", type=Path, default=DEFAULT_FREQUENCY_LIST)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EXISTING_EMBEDDINGS,
                        help="Reuse matching vectors already fetched for the news list.")
    parser.add_argument("--cache", type=Path,
                        help="Resumable JSONL cache for the conversational vocabulary.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--abtt-output", type=Path)
    parser.add_argument("--dimensions", type=int, default=MAX_DIMENSIONS,
                        help="Embedding dimensions (1–3072; default: 3072).")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not 1 <= args.dimensions <= MAX_DIMENSIONS:
        parser.error(f"--dimensions must be between 1 and {MAX_DIMENSIONS}")
    if not args.frequency_list.is_file():
        parser.error(f"SUBTLEX-US frequency list not found: {args.frequency_list}")

    model_name = f"openai-{MODEL}"
    if args.dimensions != MAX_DIMENSIONS:
        model_name += f"-{args.dimensions}d"
    model_name += "-subtlex-us-5000"
    if args.cache is None:
        args.cache = DEFAULT_CACHE if args.dimensions == MAX_DIMENSIONS else (
            ROOT / "data" / f"{model_name}.jsonl"
        )
    if args.output is None:
        args.output = DEFAULT_OUTPUT if args.dimensions == MAX_DIMENSIONS else (
            ROOT / "data" / f"{model_name}.kv"
        )
    if args.abtt_output is None:
        args.abtt_output = DEFAULT_ABTT_OUTPUT if args.dimensions == MAX_DIMENSIONS else (
            ROOT / "data" / f"{model_name}-abtt.kv"
        )

    words = conversational_words(args.frequency_list, args.limit)
    wanted = set(words)
    cached = read_embeddings(args.cache, wanted, args.dimensions)
    cached.update(read_embeddings(args.embeddings, wanted, args.dimensions))
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    missing = [word for word in words if word not in cached]
    if missing and not api_key:
        parser.error("Set OPENAI_API_KEY in the environment or .env file to fetch missing vectors.")

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("a", encoding="utf-8") as cache:
        for index, word in enumerate(missing, start=1):
            vector = fetch_embedding(word, api_key, args.dimensions)
            record = {
                "word": word,
                "model": MODEL,
                "dimensions": args.dimensions,
                "embedding": vector,
            }
            cache.write(json.dumps(record, separators=(",", ":")) + "\n")
            cache.flush()
            cached[word] = vector
            print(f"[{index}/{len(missing)}] embedded {word}")

    vectors = np.asarray([cached[word] for word in words], dtype=np.float32)
    if vectors.shape != (len(words), args.dimensions):
        raise ValueError(
            f"Expected {len(words)} vectors of {args.dimensions} dimensions, "
            f"got shape {vectors.shape}."
        )
    model = KeyedVectors(vector_size=args.dimensions)
    model.add_vectors(words, vectors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    apply_abtt(model).save(str(args.abtt_output))
    print(f"Saved raw and ABTT {len(words)}-word models to {args.output} and {args.abtt_output}")


if __name__ == "__main__":
    main()
