# Generated by Django 3.1.3 on 2021-04-20 00:38

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('AppCDV', '0005_auto_20210419_0006'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelosclustering',
            name='usuario',
            field=models.CharField(default='adminsite', max_length=100),
        ),
        migrations.AlterField(
            model_name='modelosclustering',
            name='fecha_creacion',
            field=models.DateTimeField(default=datetime.datetime(2021, 4, 20, 0, 38, 10, 775608, tzinfo=utc)),
        ),
    ]
