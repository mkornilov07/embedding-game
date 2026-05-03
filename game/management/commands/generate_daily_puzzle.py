"""Generate today's daily puzzle. Idempotent.

Run via Heroku Scheduler (or local cron) once per day:
    python manage.py generate_daily_puzzle

A daily puzzle is 4 word sums from the embedding model + 1 from the
curated pool. If the model generator fails or returns fewer than 4,
the shortfall is filled from the curated pool so the daily puzzle
still ships. Failures bubble up as `CommandError` so the scheduler
records a non-zero exit.
"""
import time
import traceback

from django.core.management.base import BaseCommand, CommandError
from django.db import OperationalError, connection, transaction
from django.utils import timezone

from game.generators import (
    DEFAULT_MODEL,
    generate_curated_word_sums,
    generate_word_sums,
)
from game.models import Puzzle, WordSum


MODEL_COUNT = 4
CURATED_COUNT = 1
TOTAL = MODEL_COUNT + CURATED_COUNT

# Neon's serverless Postgres returns "Control plane request failed" when a
# scheduler dyno hits an idle branch that hasn't woken yet. Retry with
# exponential backoff before giving up.
DB_PROBE_ATTEMPTS = 6
DB_PROBE_BASE_DELAY = 2.0


class Command(BaseCommand):
    help = "Create today's daily puzzle if it doesn't already exist."

    def handle(self, *args, **opts):
        self._wait_for_db()
        today = timezone.now().date()
        name = f"{today.strftime('%B')} {today.day}, {today.year}"

        if Puzzle.objects.filter(is_daily=True, created_at__date=today).exists():
            self.stdout.write(f"Daily puzzle for {today} already exists; nothing to do.")
            return

        model_combos = self._safe_generate_model(MODEL_COUNT)
        curated_needed = TOTAL - len(model_combos)
        curated_combos = self._safe_generate_curated(curated_needed)
        combos = model_combos + curated_combos

        if len(combos) < TOTAL:
            raise CommandError(
                f"Could not assemble {TOTAL} word sums "
                f"(model={len(model_combos)}, curated={len(curated_combos)}); aborting."
            )

        try:
            with transaction.atomic():
                puzzle = Puzzle.objects.create(name=name, is_daily=True)
                for c in combos:
                    WordSum.objects.create(
                        puzzle=puzzle,
                        addend1=c["addend1"].strip(),
                        addend2=c["addend2"].strip(),
                        sum_word=c["sum_word"].strip(),
                    )
        except Exception as e:
            raise CommandError(f"Failed to persist daily puzzle: {e}") from e

        self.stdout.write(self.style.SUCCESS(
            f"Created daily puzzle {puzzle.pk} \"{name}\" "
            f"({len(model_combos)} model + {len(curated_combos)} curated)."
        ))

    def _wait_for_db(self):
        """Probe the database with retries.

        Issues a `SELECT 1` rather than just calling `ensure_connection()`
        so we surface control-plane errors that only show up on first
        query (Neon may accept the TCP connect but fail the first
        statement while the branch is still warming).
        """
        last_err = None
        for i in range(DB_PROBE_ATTEMPTS):
            try:
                connection.close()
                connection.ensure_connection()
                with connection.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                if i > 0:
                    self.stdout.write(f"DB reachable after {i + 1} attempts.")
                return
            except OperationalError as e:
                last_err = e
                delay = DB_PROBE_BASE_DELAY * (2 ** i)
                self.stderr.write(
                    f"DB probe {i + 1}/{DB_PROBE_ATTEMPTS} failed: {e!s}; "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
        raise CommandError(
            f"Database unreachable after {DB_PROBE_ATTEMPTS} attempts: {last_err}"
        )

    def _safe_generate_model(self, count):
        if count <= 0:
            return []
        try:
            combos = generate_word_sums(count=count, model_name=DEFAULT_MODEL) or []
        except Exception:
            self.stderr.write(
                f"Model generator raised; falling back to curated.\n"
                f"{traceback.format_exc()}"
            )
            return []
        if len(combos) < count:
            self.stderr.write(
                f"Model generator returned {len(combos)}/{count} word sums; "
                f"topping up from curated."
            )
        return combos

    def _safe_generate_curated(self, count):
        if count <= 0:
            return []
        try:
            return generate_curated_word_sums(count=count)
        except Exception as e:
            self.stderr.write(
                f"Curated generator raised: {e}\n{traceback.format_exc()}"
            )
            return []
