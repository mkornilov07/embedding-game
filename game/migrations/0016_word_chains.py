# Generated manually for word-chain combinations.
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("game", "0015_sum_scoring_method"),
    ]

    operations = [
        migrations.CreateModel(
            name="WordChain",
            fields=[
                (
                    "wordcombination_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="game.wordcombination",
                    ),
                ),
            ],
            bases=("game.wordcombination",),
        ),
        migrations.CreateModel(
            name="WordChainSlot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("position", models.PositiveIntegerField()),
                (
                    "kind",
                    models.CharField(
                        choices=[
                            ("prefilled", "Pre-filled"),
                            ("draggable", "Drag and drop"),
                            ("typed", "Type"),
                        ],
                        max_length=16,
                    ),
                ),
                ("word", models.CharField(max_length=100)),
                (
                    "chain",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="slots",
                        to="game.wordchain",
                    ),
                ),
            ],
            options={
                "ordering": ["position"],
                "unique_together": {("chain", "position")},
            },
        ),
    ]
