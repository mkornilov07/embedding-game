# Local development

Set `DJANGO_DEBUG=true` in `.env` or your environment. Local Django then uses the ignored `db.dev.sqlite3` file, even if `DATABASE_URL` points to PostgreSQL. Production (`DJANGO_DEBUG=false`) continues to use `DATABASE_URL`.

With the virtual environment active, run:

```sh
python manage.py migrate
python manage.py seed_dev_puzzles
python manage.py runserver
```

`seed_dev_puzzles` creates five sample puzzles and can be rerun without duplicating them.
