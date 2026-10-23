from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("game", "0021_restore_generation_defaults")]

    operations = [
        migrations.AlterField(
            model_name="generationdefaults",
            name="model_name",
            field=models.CharField(
                default="conceptnet-numberbatch-en-3k", max_length=100,
            ),
        ),
    ]
