from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0005_puzzle_for_duel'),
    ]

    operations = [
        migrations.AddField(
            model_name='puzzle',
            name='pinned',
            field=models.BooleanField(default=False),
        ),
    ]
