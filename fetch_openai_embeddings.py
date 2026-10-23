"""Fetch one OpenAI text embedding per frequent, game-eligible English word."""
import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from build_models import filter_words


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "glove.6B.300d.txt"
DEFAULT_OUTPUT = ROOT / "data" / "openai-text-embedding-3-large-5000.jsonl"
MODEL = "text-embedding-3-large"


def frequent_eligible_words(source_path, limit):
    """Return the first `limit` words in frequency order that pass game filters."""
    words = []
    candidates = []
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            word = line.partition(" ")[0]
            if word.isalpha() and len(word) >= 3:
                candidates.append(word)
            if len(candidates) == 10_000:
                words.extend(filter_words(candidates))
                candidates.clear()
                if len(words) >= limit:
                    return words[:limit]
    if candidates:
        words.extend(filter_words(candidates))
    if len(words) < limit:
        raise ValueError(f"Found only {len(words)} eligible words in {source_path}; need {limit}.")
    return words[:limit]


def fetch_embedding(word, api_key, dimensions=3072):
    """Make one API request for one word, retrying transient failures."""
    body = json.dumps({
        "model": MODEL,
        "input": word,
        "dimensions": dimensions,
    }).encode("utf-8")
    request = Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(6):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read())["data"][0]["embedding"]
        except HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise RuntimeError(f"OpenAI API returned HTTP {error.code}: {error.read().decode(errors='replace')}") from error
            if attempt == 5:
                raise
        except (TimeoutError, URLError):
            if attempt == 5:
                raise
        time.sleep(min(2 ** attempt, 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Frequency-ranked GloVe vocabulary file.")
    parser.add_argument("--output", type=Path,
                        help="JSONL output file; completed words are skipped on reruns.")
    parser.add_argument("--dimensions", type=int, default=3072,
                        help="Embedding dimensions (1–3072; default: 3072).")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Number of eligible words to embed (default: 5000).")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not 1 <= args.dimensions <= 3072:
        parser.error("--dimensions must be between 1 and 3072")
    if not args.source.is_file():
        parser.error(f"Word list not found: {args.source}")
    if args.output is None:
        args.output = DEFAULT_OUTPUT
        if args.dimensions != 3072:
            args.output = args.output.with_name(
                f"{args.output.stem}-{args.dimensions}d{args.output.suffix}"
            )

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("Set OPENAI_API_KEY in the environment or .env file.")
    words = frequent_eligible_words(args.source, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as output:
            for line in output:
                if line.strip():
                    completed.add(json.loads(line)["word"])

    with args.output.open("a", encoding="utf-8") as output:
        for index, word in enumerate(words, start=1):
            if word in completed:
                continue
            record = {
                "word": word,
                "model": MODEL,
                "dimensions": args.dimensions,
                "embedding": fetch_embedding(word, api_key, args.dimensions),
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            print(f"[{index}/{len(words)}] embedded {word}")

    print(f"Saved {len(words)} word embeddings to {args.output}")


if __name__ == "__main__":
    main()
