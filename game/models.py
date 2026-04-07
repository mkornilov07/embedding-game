from django.conf import settings
from django.db import models


class Puzzle(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class WordCombination(models.Model):
    puzzle = models.ForeignKey(Puzzle, on_delete=models.CASCADE, related_name="combinations")

    def __str__(self):
        return f"WordCombination #{self.pk}"


class WordSum(WordCombination):
    addend1 = models.CharField(max_length=100)
    addend2 = models.CharField(max_length=100)
    sum_word = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.addend1} + {self.addend2} = {self.sum_word}"


class PuzzleCompletion(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="completions")
    puzzle = models.ForeignKey(Puzzle, on_delete=models.CASCADE, related_name="completions")
    completed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "puzzle")

    def __str__(self):
        return f"{self.user} completed {self.puzzle}"
