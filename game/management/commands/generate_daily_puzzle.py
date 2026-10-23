"""Generate today's daily puzzle. Idempotent.

Run via Heroku Scheduler (or local cron) once per day:
    python manage.py generate_daily_puzzle

Odd days use word sums; even days use word chains. Failures bubble up as
`CommandError` so the scheduler records a non-zero exit.
"""
import random
import time
import traceback

from django.core.management.base import BaseCommand, CommandError
from django.db import OperationalError, connection, transaction
from django.utils import timezone

from game.generators import (
    generate_curated_word_sums,
    generate_word_chain,
    generate_word_sums,
)
from game.calibration import get_chain_defaults
from game.models import Puzzle, WordChain, WordChainSlot, WordSum


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

        last_puzzle = (
            Puzzle.objects.filter(is_daily=True)
            .order_by("-created_at", "-pk")
            .first()
        )
        if last_puzzle and not last_puzzle.solved:
            self.stdout.write(
                f'Daily puzzle "{last_puzzle.name}" is not solved; skipping {today}.'
            )
            return

        if today.day % 2 == 0:
            chains = self._generate_chains()
            try:
                with transaction.atomic():
                    puzzle = Puzzle.objects.create(name=name, is_daily=True)
                    for words in chains:
                        chain = WordChain.objects.create(puzzle=puzzle)
                        WordChainSlot.objects.bulk_create([
                            WordChainSlot(
                                chain=chain, position=index, word=word,
                                kind=(WordChainSlot.PREFILLED if index in (0, len(words) - 1)
                                      else WordChainSlot.DRAGGABLE),
                            )
                            for index, word in enumerate(words)
                        ])
            except Exception as e:
                raise CommandError(f"Failed to persist daily puzzle: {e}") from e
            self.stdout.write(self.style.SUCCESS(
                f'Created daily puzzle {puzzle.pk} "{name}" ({len(chains)} word chains).'
            ))
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

    def _generate_chains(self):
        defaults = get_chain_defaults()
        chains = []
        for _ in range(random.randint(4, 6)):
            length = random.randint(4, 6)
            try:
                chain = generate_word_chain(
                    length,
                    model_name=defaults.model_name,
                    start_percent=defaults.start_percent,
                    min_previous_similarity=defaults.min_previous_similarity,
                    max_earlier_similarity=defaults.max_earlier_similarity,
                )
            except Exception as e:
                raise CommandError(f"Could not generate a daily word chain: {e}") from e
            if not chain:
                raise CommandError(f"Could not generate a daily word chain of length {length}.")
            chains.append(chain)
        return chains

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
            combos = generate_word_sums(count=count) or []
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
