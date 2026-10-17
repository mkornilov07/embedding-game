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

from .generators import WORD_CHAIN_MIN_SIMILARITY, validate_word_chain
from .models import Duel, DuelProgress, PuzzleCompletion, WordChain, WordSum


def puzzle_word_sums(puzzle):
    """Return only word-sum rows as ``(addends, result)`` validation keys."""
    rows = []
    for combination in puzzle.combinations.all():
        try:
            word_sum = combination.wordsum
        except WordSum.DoesNotExist:
            continue
        rows.append((frozenset((word_sum.addend1, word_sum.addend2)), word_sum.sum_word))
    return rows


def word_chain_for(combination):
    try:
        return combination.wordchain
    except WordChain.DoesNotExist:
        return None


def puzzle_combinations(puzzle):
    return list(
        puzzle.combinations.select_related("wordsum", "wordchain")
        .prefetch_related("wordchain__slots").order_by("pk")
    )


def check_word_chain(chain, words):
    """Return per-slot feedback and invalid adjacent-link indexes for a chain."""
    slots = list(chain.slots.all())
    if len(words) != len(slots):
        return [], [], list(range(max(len(words), len(slots))))

    result = validate_word_chain(words)
    invalid = set(result["invalid_slots"])
    for index, slot in enumerate(slots):
        if slot.kind == "prefilled" and words[index] != slot.word:
            invalid.add(index)

    invalid_links = [
        index for index, similarity in enumerate(result["similarities"])
        if similarity is None or similarity < WORD_CHAIN_MIN_SIMILARITY
    ]
    return [index for index in range(len(slots)) if index not in invalid], [], invalid_links


def combination_is_solved(combination, words):
    """Check a submitted row according to its concrete combination type."""
    try:
        word_sum = combination.wordsum
    except WordSum.DoesNotExist:
        chain = word_chain_for(combination)
        if chain is None:
            return False
        green, _, _ = check_word_chain(chain, words)
        return len(green) == chain.slots.count()

    return (
        len(words) == 3
        and frozenset(words[:2]) == frozenset((word_sum.addend1, word_sum.addend2))
        and words[2] == word_sum.sum_word
    )


def board_context(puzzle):
    """Build the shuffled JSON payload consumed by the shared board."""
    rows = puzzle_combinations(puzzle)
    random.shuffle(rows)

    words = []
    combinations = []
    for row in rows:
        try:
            word_sum = row.wordsum
        except WordSum.DoesNotExist:
            chain = word_chain_for(row)
            if chain is None:
                continue
            slots = []
            for slot_index, slot in enumerate(chain.slots.all()):
                slots.append({
                    "kind": slot.kind,
                    **({"word": slot.word} if slot.kind == "prefilled" else {}),
                })
                if slot.kind == "draggable":
                    words.append({
                        "text": slot.word,
                        "combo": len(combinations),
                        "slot": slot_index,
                    })
            combinations.append({"id": row.pk, "type": "word_chain", "slots": slots})
            continue
        combo_index = len(combinations)
        words.extend([
            {"text": word_sum.addend1, "combo": combo_index},
            {"text": word_sum.addend2, "combo": combo_index},
            {"text": word_sum.sum_word, "combo": combo_index},
        ])
        combinations.append({
            "id": row.pk,
            "type": "word_sum",
            "slots": [{"kind": "draggable"}] * 3,
        })

    random.shuffle(words)
    return json.dumps(words), json.dumps(combinations), len(combinations)


def check_word_sum_row(word_sums, slot_words):
    """Return green and yellow slot indexes for a word-sum attempt."""
    if len(slot_words) != 3:
        return [], []
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


# Compatibility for callers/tests from before combinations became polymorphic.
check_row = check_word_sum_row


def record_solved_combination(duel, user, combination, slot_words, total):
    """Record one solved duel combination, retaining compatibility with old sum keys."""
    progress, _ = DuelProgress.objects.get_or_create(duel=duel, user=user)
    try:
        word_sum = combination.wordsum
    except WordSum.DoesNotExist:
        solved_key = ["combination", combination.pk]
    else:
        solved_key = [sorted(slot_words[:2]), slot_words[2]]

    def freeze(value):
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((key, freeze(item)) for key, item in value.items()))
        return value

    existing = {freeze(row) for row in progress.solved_rows}
    if freeze(solved_key) not in existing:
        progress.solved_rows.append(solved_key)
        progress.save(update_fields=["solved_rows"])

    finished = len(progress.solved_rows) >= total
    if finished:
        finish_duel(duel, user)
        mark_puzzle_completed(user, duel.puzzle)
    return len(progress.solved_rows), finished


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


def record_solved_word_sum(duel, user, slot_words, word_sums):
    """Persist a solved word sum and return ``(count, finished)``."""
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


record_solved_row = record_solved_word_sum
