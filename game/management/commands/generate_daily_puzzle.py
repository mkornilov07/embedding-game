"""Generate today's daily puzzle. Idempotent.

Run via Heroku Scheduler (or local cron) once per day:
    python manage.py generate_daily_puzzle
"""
from django.core.management.base import BaseCommand
from django.utils import timezone

from game.generators import DEFAULT_MODEL, generate_word_sums
from game.models import Puzzle, WordSum


DAILY_COUNT = 5


class Command(BaseCommand):
    help = "Create today's daily puzzle if it doesn't already exist."

    def handle(self, *args, **opts):
        today = timezone.now().date()
        name = f"{today.strftime('%B')} {today.day}, {today.year}"

        if Puzzle.objects.filter(is_daily=True, created_at__date=today).exists():
            self.stdout.write(f"Daily puzzle for {today} already exists; nothing to do.")
            return

        combos = generate_word_sums(count=DAILY_COUNT, model_name=DEFAULT_MODEL)
        if not combos:
            self.stderr.write("Generator returned no combinations; aborting.")
            return

        puzzle = Puzzle.objects.create(name=name, is_daily=True)
        for c in combos:
            WordSum.objects.create(
                puzzle=puzzle,
                addend1=c["addend1"].strip(),
                addend2=c["addend2"].strip(),
                sum_word=c["sum_word"].strip(),
            )
        self.stdout.write(self.style.SUCCESS(
            f"Created daily puzzle {puzzle.pk} \"{name}\" with {len(combos)} word sums."
        ))
