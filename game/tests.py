import json
from datetime import timedelta
from unittest.mock import patch

import numpy as np
from gensim.models import KeyedVectors
from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from .models import Duel, Puzzle, PuzzleCompletion, WordChain, WordChainSlot, WordSum
from .calibration import as_dict, generate_round, get_defaults, select_candidate
from .generators import (
    WORD_CHAIN_3K_MODEL, WORD_CHAIN_10K_MODEL, WORD_CHAIN_EXPANDED_MODEL,
    WORD_CHAIN_MAX_EARLIER_SIMILARITY, WORD_CHAIN_MIN_SIMILARITY,
    _best_arrangement, _similarity_search,
    generate_word_chain, validate_word_chain,
)
from .services import board_context, check_row, mark_puzzle_completed, record_solved_row


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


class PuzzleCreationDefaultsTests(TestCase):
    def test_form_uses_current_database_generation_defaults(self):
        user = get_user_model().objects.create_user("creator", password="password")
        defaults = get_defaults()
        defaults.model_name = "glove-twitter-50"
        defaults.related_pairs = False
        defaults.sum_scoring_method = "raw_sum"
        defaults.abtt = False
        defaults.top_n_vocab = 1500
        defaults.refine_iterations = 2
        defaults.min_similarity = 0.37
        defaults.min_refined_similarity = 0.42
        defaults.min_gap_ratio = 1.15
        defaults.max_synonym_similarity = 0.31
        defaults.save()
        client = Client()
        client.force_login(user)

        response = client.get(reverse("create_puzzle"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'value="glove-twitter-50" selected')
        self.assertContains(response, 'value="raw_sum" selected')
        self.assertContains(response, '<input type="checkbox" id="related-pairs">')
        self.assertContains(response, '<input type="checkbox" id="abtt">')
        for value in ("1500", "2", "0.37", "0.42", "1.15", "0.31"):
            self.assertContains(response, f'value="{value}"')


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


class WordSumGenerationTests(TestCase):
    def test_scoring_methods_include_raw_vector_sum(self):
        model = KeyedVectors(vector_size=2)
        model.add_vectors(
            ["a", "b", "balanced", "x_weighted"],
            [[10, 0], [0, 1], [1, 1], [1, 0.1]],
        )

        cosadd = _similarity_search(model, "cosadd")
        raw_sum = _similarity_search(model, "raw_sum")

        self.assertEqual(cosadd(positive=["a", "b"], topn=1)[0][0], "balanced")
        self.assertEqual(raw_sum(positive=["a", "b"], topn=1)[0][0], "x_weighted")

    def test_repeated_nearest_word_queries_use_cache(self):
        model = KeyedVectors(vector_size=2)
        model.add_vectors(["a", "b", "c"], [[1, 0], [0, 1], [1, 1]])
        similar = _similarity_search(model, "cosadd")

        similar(positive=["a", "b"], topn=2)
        similar(positive=["b", "a"], topn=1)

        self.assertEqual(similar.cache_info().hits, 1)

    def test_best_arrangement_rejects_nonorthogonal_addends(self):
        model = KeyedVectors(vector_size=2)
        model.add_vectors(["a", "b", "c"], [[1, 0], [1, 0.05], [0, 1]])
        similar = lambda positive, topn: [("c", 1.0)]

        addends, sum_word, _ = _best_arrangement(("a", "b", "c"), similar, model)

        self.assertIsNone(addends)
        self.assertIsNone(sum_word)


class WordChainTests(TestCase):
    def setUp(self):
        self.model = KeyedVectors(vector_size=2)
        self.model.add_vectors(
            ["sun", "star", "light", "rain"],
            [[1, 0], [0.96, 0.28], [0.84, 0.54], [0, 1]],
        )
        self.model.fill_norms()
        self.model._normalized_vectors = self.model.vectors / self.model.norms[:, None]
        self.model_patch = patch(
            "game.generators._load_model",
            return_value=(self.model, list(self.model.index_to_key)),
        )
        self.model_loader = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def make_chain(self, puzzle):
        chain = WordChain.objects.create(puzzle=puzzle)
        WordChainSlot.objects.bulk_create([
            WordChainSlot(chain=chain, position=0, word="rain", kind="prefilled"),
            WordChainSlot(chain=chain, position=1, word="light", kind="draggable"),
            WordChainSlot(chain=chain, position=2, word="sun", kind="typed"),
        ])
        return chain

    def test_validation_rejects_unknown_words_and_low_similarity(self):
        self.assertTrue(validate_word_chain(["rain", "light", "sun"])["valid"])
        self.assertTrue(validate_word_chain(["light", "rain"])["valid"])
        self.assertFalse(validate_word_chain(["sun", "missing"])["valid"])
        self.assertFalse(validate_word_chain(["sun", "rain"])["valid"])

    def test_validation_rejects_nonadjacent_similarity(self):
        result = validate_word_chain(["sun", "star", "light"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_slots"], [0, 2])
        self.assertTrue(all(score >= WORD_CHAIN_MIN_SIMILARITY for score in result["similarities"]))

    def test_chain_similarity_debug_is_admin_only(self):
        user = get_user_model().objects.create_user("creator", password="password")
        admin = get_user_model().objects.create_user(
            "admin", password="password", is_staff=True,
        )
        payload = json.dumps({"words": ["light", "rain"]})

        self.client.force_login(user)
        self.assertNotContains(self.client.get(reverse("create_puzzle")), 'id="show-chain-sim"')
        response = self.client.post(
            reverse("validate_word_chain"), data=payload, content_type="application/json",
        )
        self.assertNotIn("similarities", response.json())

        self.client.force_login(admin)
        self.assertContains(self.client.get(reverse("create_puzzle")), 'id="show-chain-sim"')
        response = self.client.post(
            reverse("validate_word_chain"), data=payload, content_type="application/json",
        )
        self.assertAlmostEqual(
            response.json()["similarities"][0], self.model.similarity("light", "rain"),
        )

    def test_generator_uses_distinct_words_and_returns_valid_chain(self):
        self.assertEqual(WORD_CHAIN_MIN_SIMILARITY, 0.47)
        self.assertEqual(WORD_CHAIN_MAX_EARLIER_SIMILARITY, 0.31)
        with patch("game.generators.random.choice", return_value=0):
            chain = generate_word_chain(3)
        self.model_loader.assert_called_with(WORD_CHAIN_10K_MODEL, abtt=True)
        self.assertEqual(len(chain), 3)
        self.assertEqual(len(set(chain)), 3)
        self.assertTrue(validate_word_chain(chain)["valid"])

        creator = get_user_model().objects.create_user("generator", password="password")
        self.client.force_login(creator)
        with patch("game.generators.random.choice", return_value=0):
            response = self.client.post(
                reverse("generate_combinations"),
                data=json.dumps({"type": "word_chain", "length": 3}),
                content_type="application/json",
            )
        generated = response.json()["combinations"][0]
        self.assertEqual(len(generated["words"]), 3)
        self.assertTrue(all(slot["kind"] == "draggable" for slot in generated["words"]))
        self.assertTrue(validate_word_chain([slot["word"] for slot in generated["words"]])["valid"])

        page = self.client.get(reverse("create_puzzle"))
        self.assertEqual(page.context["chain_min_similarity"], 0.47)
        self.assertEqual(page.context["chain_max_earlier_similarity"], 0.31)

    def test_generator_respects_previous_and_earlier_similarity_limits(self):
        with patch("game.generators.random.choice", return_value=3):
            chain = generate_word_chain(
                3, min_previous_similarity=0.45, max_earlier_similarity=0.5,
                start_percent=100,
            )
        self.assertEqual(chain[0], "rain")
        for first, second in zip(chain, chain[1:]):
            self.assertGreaterEqual(self.model.similarity(first, second), 0.45)
        self.assertLessEqual(self.model.similarity(chain[0], chain[2]), 0.5)
        self.assertIsNone(generate_word_chain(
            3, min_previous_similarity=0.8, max_earlier_similarity=0.5,
        ))

    def test_starting_word_uses_top_percentage_for_each_vocabulary(self):
        for model_name in (
            WORD_CHAIN_3K_MODEL, WORD_CHAIN_10K_MODEL, WORD_CHAIN_EXPANDED_MODEL,
        ):
            for percent, expected in ((20, "sun"), (100, "rain")):
                with self.subTest(model_name=model_name, percent=percent):
                    with patch("game.generators.random.choice", side_effect=lambda seeds: max(seeds)):
                        with patch("game.generators.random.choices", side_effect=lambda options, **_: [options[-1]]):
                            chain = generate_word_chain(
                                3, model_name=model_name, start_percent=percent,
                            )
                    self.assertEqual(chain[0], expected)

        for percent in (0, 101, float("nan")):
            with self.assertRaises(ValueError):
                generate_word_chain(3, start_percent=percent)

    def test_expanded_generators_use_on_demand_similarities(self):
        for model_name in (WORD_CHAIN_10K_MODEL, WORD_CHAIN_EXPANDED_MODEL):
            with self.subTest(model_name=model_name):
                with patch.object(self.model, "similar_by_key", wraps=self.model.similar_by_key) as similar:
                    with patch("game.generators.random.choice", return_value=0):
                        chain = generate_word_chain(3, model_name=model_name)
                self.assertTrue(similar.called)
                self.assertTrue(validate_word_chain(chain, model_name=model_name)["valid"])
                self.assertFalse(hasattr(self.model, "_chain_similarities"))

    def test_debug_tools_and_custom_chain_settings_require_admin(self):
        user = get_user_model().objects.create_user("debug_user", password="password")
        admin = get_user_model().objects.create_user(
            "debug_admin", password="password", is_staff=True,
        )
        payload = json.dumps({
            "type": "word_chain", "length": 3,
            "min_previous_similarity": 0.8, "max_earlier_similarity": 1,
        })

        self.client.force_login(user)
        self.assertEqual(self.client.get(reverse("debug_tools")).status_code, 302)
        response = self.client.post(
            reverse("generate_combinations"), data=payload, content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

        self.client.force_login(admin)
        page = self.client.get(reverse("debug_tools"))
        self.assertContains(page, "Word-sum calibration")
        self.assertContains(page, "Generate word sums")
        self.assertContains(page, "Generate word chains")
        self.assertContains(page, '<option value="3k">Original (3k)</option>')
        self.assertContains(page, '<option value="10k" selected>Current game (10k)</option>')
        self.assertContains(page, '<option value="expanded">Expanded (20k)</option>')
        self.assertContains(page, 'id="debug-chain-start-percent" type="number" min="0.01" max="100" step="any" value="20"')
        self.assertContains(page, 'id="debug-min-previous" type="number" min="-1" max="1" step="any" value="0.47"')
        self.assertContains(page, 'id="debug-max-earlier" type="number" min="-1" max="1" step="any" value="0.31"')
        response = self.client.post(
            reverse("generate_combinations"), data=payload, content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["similarities"]), 2)
        self.assertTrue(all(score >= 0.8 for score in response.json()["similarities"]))
        self.assertFalse(response.json()["puzzle_valid"])

        for vocabulary in ("3k", "10k", "expanded"):
            expanded = json.dumps({
                "type": "word_chain", "length": 3,
                "vocabulary": vocabulary, "start_percent": 100,
            })
            self.client.force_login(user)
            self.assertEqual(self.client.post(
                reverse("generate_combinations"), data=expanded, content_type="application/json",
            ).status_code, 403)
            self.client.force_login(admin)
            with patch("game.generators.random.choice", return_value=3):
                response = self.client.post(
                    reverse("generate_combinations"), data=expanded, content_type="application/json",
                )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["puzzle_valid"])
            self.assertEqual(len(response.json()["similarities"]), 2)

    @patch("game.views.generate_word_sums")
    def test_word_sum_preview_passes_selected_parameters(self, generate):
        admin = get_user_model().objects.create_user(
            "sum_admin", password="password", is_staff=True,
        )
        self.client.force_login(admin)
        generate.return_value = [{"type": "word_sum", "addend1": "sun", "addend2": "rain", "sum_word": "light"}]
        response = self.client.post(
            reverse("generate_combinations"),
            data=json.dumps({
                "type": "word_sum", "model": "conceptnet-numberbatch-en", "count": 1,
                "top_n_vocab": 0, "refine_iterations": 0,
                "min_similarity": 0.2, "min_refined_similarity": 0.3,
                "min_gap_ratio": 1.1, "max_synonym_similarity": 0.6,
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["combinations"]), 1)
        self.assertEqual(generate.call_args.kwargs["top_n_vocab"], 0)
        self.assertEqual(generate.call_args.kwargs["refine_iterations"], 0)
        self.assertEqual(generate.call_args.kwargs["min_gap_ratio"], 1.1)

    def test_mixed_puzzle_payload_and_alternative_valid_chain(self):
        puzzle = Puzzle.objects.create(name="Mixed")
        word_sum = WordSum.objects.create(
            puzzle=puzzle, addend1="sun", addend2="rain", sum_word="light",
        )
        chain = self.make_chain(puzzle)

        words_json, combinations_json, total = board_context(puzzle)
        bank_words = [entry["text"] for entry in json.loads(words_json)]
        chain_payload = next(
            combo for combo in json.loads(combinations_json) if combo["type"] == "word_chain"
        )
        self.assertEqual(total, 2)
        self.assertEqual(bank_words.count("light"), 2)
        self.assertEqual(chain_payload["slots"], [
            {"kind": "prefilled", "word": "rain"},
            {"kind": "draggable"},
            {"kind": "typed"},
        ])

        entries = [
            {"id": word_sum.pk, "words": ["sun", "rain", "light"]},
            {"id": chain.pk, "words": ["rain", "light", "star"]},
        ]
        response = self.client.post(
            reverse("check_puzzle", args=[puzzle.pk]),
            data=json.dumps(entries), content_type="application/json",
        )
        self.assertTrue(response.json()["correct"])
        puzzle.refresh_from_db()
        self.assertTrue(puzzle.solved)

        entries[1]["words"] = ["rain", "sun", "star"]
        response = self.client.post(
            reverse("check_row", args=[puzzle.pk]),
            data=json.dumps({"combination_id": chain.pk, "words": entries[1]["words"]}),
            content_type="application/json",
        )
        self.assertNotEqual(len(response.json()["green_slots"]), 3)
        self.assertIn(0, response.json()["invalid_links"])

        entries[1]["words"] = ["rain", "light", "rain"]
        response = self.client.post(
            reverse("check_row", args=[puzzle.pk]),
            data=json.dumps({"combination_id": chain.pk, "words": entries[1]["words"]}),
            content_type="application/json",
        )
        self.assertNotEqual(len(response.json()["green_slots"]), 3)
        self.assertEqual(response.json()["invalid_links"], [])

    def test_editor_save_rejects_invalid_chain_without_erasing_puzzle(self):
        creator = get_user_model().objects.create_user("chain_creator", password="password")
        self.client.force_login(creator)
        puzzle = Puzzle.objects.create(name="Original", created_by=creator)
        WordSum.objects.create(puzzle=puzzle, addend1="sun", addend2="rain", sum_word="light")

        invalid = {
            "id": puzzle.pk,
            "name": "Changed",
            "combinations": [{
                "type": "word_chain",
                "words": [
                    {"word": "sun", "kind": "prefilled"},
                    {"word": "missing", "kind": "typed"},
                ],
            }],
        }
        response = self.client.post(
            reverse("save_puzzle"), data=json.dumps(invalid), content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        puzzle.refresh_from_db()
        self.assertEqual(puzzle.name, "Original")
        self.assertEqual(puzzle.combinations.count(), 1)

        invalid["combinations"][0]["words"][1]["word"] = "rain"
        response = self.client.post(
            reverse("save_puzzle"), data=json.dumps(invalid), content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

        invalid["combinations"][0]["words"] = [
            {"word": word, "kind": "typed"}
            for word in ("sun", "star", "light")
        ]
        response = self.client.post(
            reverse("save_puzzle"), data=json.dumps(invalid), content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

        invalid["combinations"][0]["words"] = [
            {"word": "sun", "kind": "prefilled"},
            {"word": "star", "kind": "typed"},
        ]
        response = self.client.post(
            reverse("save_puzzle"), data=json.dumps(invalid), content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(WordChain.objects.get(puzzle=puzzle).slots.count(), 2)
        self.assertContains(self.client.get(reverse("edit_puzzle", args=[puzzle.pk])), '"type": "word_chain"')

    def test_duel_counts_sums_and_chains_once(self):
        player = get_user_model().objects.create_user("chain_player", password="password")
        opponent = get_user_model().objects.create_user("chain_opponent", password="password")
        puzzle = Puzzle.objects.create(name="Mixed duel")
        word_sum = WordSum.objects.create(
            puzzle=puzzle, addend1="sun", addend2="rain", sum_word="light",
        )
        chain = self.make_chain(puzzle)
        duel = Duel.objects.create(
            inviter=player, opponent=opponent, puzzle=puzzle, status=Duel.STATUS_ACTIVE,
        )
        self.client.force_login(player)

        def solve(combination, words):
            return self.client.post(
                reverse("duel_row_solved", args=[duel.pk]),
                data=json.dumps({"combination_id": combination.pk, "words": words}),
                content_type="application/json",
            ).json()

        self.assertEqual(solve(word_sum, ["sun", "rain", "light"])["count"], 1)
        self.assertEqual(solve(word_sum, ["rain", "sun", "light"])["count"], 1)
        result = solve(chain, ["rain", "light", "sun"])
        self.assertEqual(result["count"], 2)
        self.assertTrue(result["finished"])
        duel.refresh_from_db()
        self.assertEqual(duel.status, Duel.STATUS_COMPLETED)
