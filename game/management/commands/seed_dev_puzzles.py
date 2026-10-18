"""Add five repeatable sample puzzles to the local SQLite database."""

import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from game.generators import CURATED_PATH
from game.models import Puzzle, WordChain, WordChainSlot, WordSum


CHAIN_GROUPS = (
    (
        ("actor", "entertainer", "entertainment", "recreation"),
        ("journalist", "writer", "screenplay", "film", "footage"),
        ("engineering", "technology", "innovation", "creativity", "imagination", "fantasy"),
    ),
    (
        ("deficit", "shortage", "famine", "hungry"),
        ("venture", "invest", "fund", "cash", "currency"),
        ("resolution", "resolve", "fix", "repair", "restoration", "preservation"),
    ),
)


class Command(BaseCommand):
    help = "Create five sample puzzles in the local development database."

    def handle(self, *args, **options):
        if not settings.DEBUG or connection.vendor != "sqlite":
            raise CommandError("Sample puzzles can only be seeded into the development SQLite database.")

        with open(CURATED_PATH, encoding="utf-8") as source:
            curated = json.load(source)

        created_count = 0
        with transaction.atomic():
            for index in range(5):
                puzzle, created = Puzzle.objects.get_or_create(
                    name=f"Development Puzzle {index + 1}",
                    created_by=None,
                    is_daily=False,
                    for_duel=False,
                )
                if not created:
                    continue
                created_count += 1
                if index < 3:
                    for words in curated[index * 5:(index + 1) * 5]:
                        WordSum.objects.create(puzzle=puzzle, **words)
                else:
                    for words in CHAIN_GROUPS[index - 3]:
                        chain = WordChain.objects.create(puzzle=puzzle)
                        WordChainSlot.objects.bulk_create([
                            WordChainSlot(
                                chain=chain, position=position, word=word,
                                kind=(WordChainSlot.PREFILLED if position in (0, len(words) - 1)
                                      else WordChainSlot.DRAGGABLE),
                            )
                            for position, word in enumerate(words)
                        ])

        self.stdout.write(self.style.SUCCESS(f"Created {created_count} development puzzles."))
