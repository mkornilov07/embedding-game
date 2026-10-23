# Local development

Set `DJANGO_DEBUG=true` in `.env` or your environment. Local Django then uses the ignored `db.dev.sqlite3` file, even if `DATABASE_URL` points to PostgreSQL. Production (`DJANGO_DEBUG=false`) continues to use `DATABASE_URL`.

With the virtual environment active, run:

```sh
python manage.py migrate
python manage.py seed_dev_puzzles
python manage.py runserver
```

`seed_dev_puzzles` creates five sample puzzles and can be rerun without duplicating them.

## Embedding files

The website loads `data/<model-name>.kv` directly. It does not download embeddings,
change their vectors, or trim their vocabularies at runtime. The model names and
labels are listed in `game/generators.py`; each file has its own word-sum and
word-chain defaults in the database. Run migrations after updating the code.

Build the ConceptNet 3,000-word and 10,000-word files (plus their ABTT variants) with
`python build_models.py`. The script expects `glove.6B.300d.txt`,
`numberbatch-en-19.08.txt`, and the NLTK English tagger data under `data/` or in
the NLTK data path.

Build the conversational OpenAI file and its ABTT variant with
`python build_conversational_embeddings.py`. It expects the SUBTLEX-US frequency
file at `data/SUBTLEXus74286wordstextversion.tsv` and `OPENAI_API_KEY` for any
words not already cached. The build scripts write the final `.kv` files under
`data/`.

Source and intermediate files under `data/` are excluded from Git. The catalogued
`.kv` files used by the website are tracked in Git and included in deployments.
