"""Application services shared by HTTP views and real-time events.

Views should translate requests into responses. The game rules and state changes
live here so they can be tested without rendering templates.
"""

import json
import random

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.urls import reverse
from django.db.models import Q
from django.utils import timezone

from .models import Duel, DuelProgress, PuzzleCompletion, WordSum


def puzzle_word_sums(puzzle):
    """Return a puzzle's rows as ``(addends, result)`` validation keys."""
    rows = []
    for combination in puzzle.combinations.all():
        try:
            word_sum = combination.wordsum
        except WordSum.DoesNotExist:
            continue
        rows.append((frozenset((word_sum.addend1, word_sum.addend2)), word_sum.sum_word))
    return rows


def board_context(puzzle):
    """Build the shuffled JSON payload consumed by the shared board."""
    rows = list(puzzle.combinations.select_related("wordsum"))
    random.shuffle(rows)

    words = []
    combinations = []
    for row in rows:
        try:
            word_sum = row.wordsum
        except WordSum.DoesNotExist:
            continue
        combo_index = len(combinations)
        words.extend([
            {"text": word_sum.addend1, "combo": combo_index},
            {"text": word_sum.addend2, "combo": combo_index},
            {"text": word_sum.sum_word, "combo": combo_index},
        ])
        combinations.append({"id": row.pk, "type": "word_sum"})

    random.shuffle(words)
    return json.dumps(words), json.dumps(combinations), len(combinations)


def check_row(word_sums, slot_words):
    """Return green and yellow slot indexes for one attempted row."""
    first, second, result = slot_words
    addends = frozenset((first, second))

    if (addends, result) in word_sums:
        return [0, 1, 2], []

    word_set = {first, second, result}
    if any(word_set == valid_addends | {valid_result}
           for valid_addends, valid_result in word_sums):
        return [], [0, 1, 2]

    if any(addends == valid_addends for valid_addends, _ in word_sums):
        return [0, 1], []

    for valid_addends, valid_result in word_sums:
        if result == valid_result:
            for index in range(2):
                if slot_words[index] in valid_addends:
                    return [index, 2], []
    return [], []


def mark_puzzle_completed(user, puzzle):
    if not puzzle.solved:
        puzzle.solved = True
        puzzle.save(update_fields=["solved"])
    if user is not None and user.is_authenticated:
        now = timezone.now()
        completion, _ = PuzzleCompletion.objects.get_or_create(
            user=user,
            puzzle=puzzle,
            defaults={"first_opened_at": now},
        )
        if completion.first_solved_at is None:
            completion.first_solved_at = now
            completion.completed_at = now
            if completion.first_opened_at is None:
                completion.first_opened_at = now
            completion.save(update_fields=[
                "first_opened_at", "first_solved_at", "completed_at",
            ])


def record_puzzle_open(user, puzzle):
    if not user.is_authenticated:
        return
    completion, _ = PuzzleCompletion.objects.get_or_create(
        user=user,
        puzzle=puzzle,
        defaults={"first_opened_at": timezone.now()},
    )
    if completion.first_opened_at is None:
        completion.first_opened_at = timezone.now()
        completion.save(update_fields=["first_opened_at"])


def record_hint_used(user, puzzle):
    if not user.is_authenticated:
        return
    completion, _ = PuzzleCompletion.objects.get_or_create(
        user=user,
        puzzle=puzzle,
        defaults={"first_opened_at": timezone.now(), "used_hints": True},
    )
    if completion.first_solved_at is None and not completion.used_hints:
        completion.used_hints = True
        completion.save(update_fields=["used_hints"])


def user_has_open_duel(user):
    return Duel.objects.filter(
        status__in=(Duel.STATUS_PENDING, Duel.STATUS_ACTIVE),
    ).filter(Q(inviter=user) | Q(opponent=user)).exists()


def send_lobby_event(user_id, payload):
    _send(f"lobby_{user_id}", "lobby_event", payload)


def send_duel_event(duel_id, payload):
    _send(f"duel_{duel_id}", "duel_event", payload)


def _send(group, event_type, payload):
    async_to_sync(get_channel_layer().group_send)(
        group, {"type": event_type, "data": payload}
    )


def duel_summary(duel):
    return {
        "id": duel.pk,
        "inviter": duel.inviter.username,
        "opponent": duel.opponent.username,
        "puzzle_name": duel.puzzle.name,
        "url": reverse("duel_detail", args=[duel.pk]),
    }


def finish_duel(duel, winner):
    duel.status = Duel.STATUS_COMPLETED
    duel.winner = winner
    duel.completed_at = timezone.now()
    duel.save(update_fields=["status", "winner", "completed_at"])


def record_solved_row(duel, user, slot_words, word_sums):
    """Persist a solved row and return ``(count, finished)``."""
    progress, _ = DuelProgress.objects.get_or_create(duel=duel, user=user)
    solved_key = (tuple(sorted(slot_words[:2])), slot_words[2])
    existing = {(tuple(row[0]), row[1]) for row in progress.solved_rows}
    if solved_key not in existing:
        progress.solved_rows.append([list(solved_key[0]), solved_key[1]])
        progress.save(update_fields=["solved_rows"])

    finished = len(progress.solved_rows) >= len(word_sums)
    if finished:
        finish_duel(duel, user)
        mark_puzzle_completed(user, duel.puzzle)
    return len(progress.solved_rows), finished
