from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("game", "0013_calibrationround_noise_stdev"),
    ]

    operations = [
        migrations.AddField(
            model_name="calibrationround",
            name="learning_rate",
            field=models.FloatField(default=0.1),
        ),
    ]
