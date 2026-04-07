from django.db import models


class WordSum(models.Model):
    addend1 = models.CharField(max_length=100)
    addend2 = models.CharField(max_length=100)
    sum_word = models.CharField(max_length=100)

    class Meta:
        unique_together = ("addend1", "addend2", "sum_word")

    def __str__(self):
        return f"{self.addend1} + {self.addend2} = {self.sum_word}"
