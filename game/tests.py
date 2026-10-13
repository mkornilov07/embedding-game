from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone
from datetime import timedelta
from unittest.mock import patch

from .models import Duel, Puzzle, PuzzleCompletion, WordSum
from .calibration import as_dict, generate_round, get_defaults, select_candidate
from .services import check_row, mark_puzzle_completed, record_solved_row


class RowCheckingTests(TestCase):
    def setUp(self):
        self.rows = [
            (frozenset(("rain", "bow")), "rainbow"),
            (frozenset(("sun", "light")), "sunlight"),
        ]

    def test_exact_row_is_green(self):
        self.assertEqual(check_row(self.rows, ["rain", "bow", "rainbow"]), ([0, 1, 2], []))

    def test_correct_words_in_wrong_order_are_yellow(self):
        self.assertEqual(check_row(self.rows, ["rain", "rainbow", "bow"]), ([], [0, 1, 2]))

    def test_matching_addends_are_partial_green(self):
        self.assertEqual(check_row(self.rows, ["rain", "bow", "sunlight"]), ([0, 1], []))


class DuelProgressTests(TestCase):
    def test_solved_rows_are_counted_once(self):
        user_model = get_user_model()
        user = user_model.objects.create_user("player", password="password")
        opponent = user_model.objects.create_user("opponent", password="password")
        puzzle = Puzzle.objects.create(name="Test")
        WordSum.objects.create(puzzle=puzzle, addend1="rain", addend2="bow", sum_word="rainbow")
        duel = Duel.objects.create(inviter=user, opponent=opponent, puzzle=puzzle, status=Duel.STATUS_ACTIVE)
        rows = [(frozenset(("rain", "bow")), "rainbow")]

        count, finished = record_solved_row(duel, user, ["rain", "bow", "rainbow"], rows)
        self.assertEqual((count, finished), (1, True))

        count, finished = record_solved_row(duel, user, ["bow", "rain", "rainbow"], rows)
        self.assertEqual((count, finished), (1, True))

        duel.refresh_from_db()
        self.assertEqual(duel.status, Duel.STATUS_COMPLETED)
        puzzle.refresh_from_db()
        self.assertTrue(puzzle.solved)


class PuzzleCompletionTests(TestCase):
    def test_guest_completion_marks_puzzle_solved(self):
        puzzle = Puzzle.objects.create(name="Guest puzzle")

        mark_puzzle_completed(None, puzzle)

        puzzle.refresh_from_db()
        self.assertTrue(puzzle.solved)


class PuzzleListTests(TestCase):
    def test_puzzle_displays_distinct_solver_count(self):
        user_model = get_user_model()
        first_user = user_model.objects.create_user("first", password="password")
        second_user = user_model.objects.create_user("second", password="password")
        puzzle = Puzzle.objects.create(name="Popular")
        PuzzleCompletion.objects.create(user=first_user, puzzle=puzzle)
        PuzzleCompletion.objects.create(user=second_user, puzzle=puzzle)

        response = Client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["puzzles"][0].solver_count, 2)


class PuzzleLeaderboardTests(TestCase):
    def test_leaderboard_orders_by_time_and_shows_hint_usage(self):
        user_model = get_user_model()
        fast = user_model.objects.create_user("fast", password="password")
        slow = user_model.objects.create_user("slow", password="password")
        puzzle = Puzzle.objects.create(name="Timed")
        opened_at = timezone.now()
        PuzzleCompletion.objects.create(
            user=slow,
            puzzle=puzzle,
            first_opened_at=opened_at,
            first_solved_at=opened_at + timedelta(seconds=20),
            used_hints=True,
        )
        PuzzleCompletion.objects.create(
            user=fast,
            puzzle=puzzle,
            first_opened_at=opened_at,
            first_solved_at=opened_at + timedelta(seconds=5),
        )

        response = Client().get(reverse("puzzle_leaderboard", args=[puzzle.pk]))

        self.assertEqual([entry["username"] for entry in response.context["leaderboard"]], ["fast", "slow"])
        self.assertTrue(response.context["leaderboard"][1]["used_hints"])


class CalibrationTests(TestCase):
    @patch("game.calibration.generate_word_sums")
    def test_rounds_store_candidates_and_selection(self, generate):
        generate.return_value = [{"type": "word_sum", "addend1": "a", "addend2": "b", "sum_word": "c"}]
        user = get_user_model().objects.create_user("admin", password="password", is_staff=True)
        parameters = as_dict(get_defaults())

        first_round = generate_round(user, parameters)
        next_round = select_candidate(user, first_round.pk, 0)

        first_round.refresh_from_db()
        self.assertEqual(len(first_round.noise_vectors), 9)
        self.assertEqual(len(first_round.candidates), 9)
        self.assertEqual(first_round.selected_index, 0)
        self.assertEqual(first_round.clicks.count(), 1)
        self.assertNotEqual(first_round.pk, next_round.pk)

# Create your tests here.
