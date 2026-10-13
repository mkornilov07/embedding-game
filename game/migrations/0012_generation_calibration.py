import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def create_initial_generation_defaults(apps, schema_editor):
    GenerationDefaults = apps.get_model("game", "GenerationDefaults")
    GenerationDefaults.objects.get_or_create(
        key="default",
        defaults={
            "model_name": "conceptnet-numberbatch-en",
            "related_pairs": True,
            "cosmul": True,
            "abtt": True,
            "top_n_vocab": 3000,
            "refine_iterations": 1,
            "min_similarity": 0.25,
            "min_refined_similarity": 0.5,
            "min_gap_ratio": 1.0,
            "max_synonym_similarity": 0.45,
        },
    )


class Migration(migrations.Migration):

    dependencies = [
        ("game", "0011_puzzlecompletion_timing"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="GenerationDefaults",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("key", models.CharField(default="default", max_length=32, unique=True)),
                ("model_name", models.CharField(default="conceptnet-numberbatch-en", max_length=100)),
                ("related_pairs", models.BooleanField(default=True)),
                ("cosmul", models.BooleanField(default=True)),
                ("abtt", models.BooleanField(default=True)),
                ("top_n_vocab", models.PositiveIntegerField(default=3000)),
                ("refine_iterations", models.PositiveIntegerField(default=1)),
                ("min_similarity", models.FloatField(default=0.25)),
                ("min_refined_similarity", models.FloatField(default=0.5)),
                ("min_gap_ratio", models.FloatField(default=1.0)),
                ("max_synonym_similarity", models.FloatField(default=0.45)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name="CalibrationRound",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("parameters_before", models.JSONField()),
                ("noise_vectors", models.JSONField()),
                ("candidates", models.JSONField()),
                ("selected_index", models.PositiveIntegerField(blank=True, null=True)),
                ("parameters_after", models.JSONField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("selected_at", models.DateTimeField(blank=True, null=True)),
                ("calibrator", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name="CalibrationClick",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("event", models.CharField(max_length=64)),
                ("candidate_index", models.PositiveIntegerField(blank=True, null=True)),
                ("payload", models.JSONField(default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("round", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="clicks", to="game.calibrationround")),
            ],
        ),
        migrations.RunPython(create_initial_generation_defaults, migrations.RunPython.noop),
    ]
