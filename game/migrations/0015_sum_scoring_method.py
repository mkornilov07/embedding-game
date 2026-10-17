from django.db import migrations, models


def copy_scoring_method(apps, schema_editor):
    defaults = apps.get_model("game", "GenerationDefaults")
    defaults.objects.filter(cosmul=True).update(sum_scoring_method="cosmul")
    defaults.objects.filter(cosmul=False).update(sum_scoring_method="cosadd")


def restore_cosmul_flag(apps, schema_editor):
    defaults = apps.get_model("game", "GenerationDefaults")
    defaults.objects.filter(sum_scoring_method="cosmul").update(cosmul=True)
    defaults.objects.exclude(sum_scoring_method="cosmul").update(cosmul=False)


class Migration(migrations.Migration):
    dependencies = [("game", "0014_calibrationround_learning_rate")]

    operations = [
        migrations.AddField(
            model_name="generationdefaults",
            name="sum_scoring_method",
            field=models.CharField(
                choices=[
                    ("cosadd", "3CosAdd"),
                    ("cosmul", "3CosMul"),
                    ("raw_sum", "Raw vector sum cosine"),
                ],
                default="cosmul",
                max_length=16,
            ),
        ),
        migrations.RunPython(copy_scoring_method, restore_cosmul_flag),
        migrations.RemoveField(model_name="generationdefaults", name="cosmul"),
    ]
