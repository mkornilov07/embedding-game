from django.db import migrations, models


def copy_words(apps, schema_editor):
    Example = apps.get_model("game", "PreferenceLabelExample")
    for item in Example.objects.all().iterator():
        data = item.example
        if item.example_type == "word_chain":
            words = data.get("words", [])
        else:
            words = [data.get("addend1"), data.get("addend2"), data.get("sum_word")]
        item.words = words
        item.save(update_fields=["words"])


class Migration(migrations.Migration):

    dependencies = [("game", "0023_preferencelabelexample")]

    operations = [
        migrations.AddField(
            model_name="preferencelabelexample",
            name="words",
            field=models.JSONField(null=True),
        ),
        migrations.RunPython(copy_words, migrations.RunPython.noop),
        migrations.RemoveField(model_name="preferencelabelexample", name="model_name"),
        migrations.RemoveField(model_name="preferencelabelexample", name="example"),
        migrations.RemoveField(model_name="preferencelabelexample", name="generation_parameters"),
        migrations.RemoveField(model_name="preferencelabelexample", name="labeled_by"),
        migrations.RemoveField(model_name="preferencelabelexample", name="created_at"),
        migrations.RemoveField(model_name="preferencelabelexample", name="labeled_at"),
        migrations.AlterField(
            model_name="preferencelabelexample",
            name="words",
            field=models.JSONField(),
        ),
        migrations.AlterModelOptions(
            name="preferencelabelexample",
            options={"ordering": ["pk"]},
        ),
    ]
