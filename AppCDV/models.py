from django.db import models
from django.utils import timezone

# Create your models here.

class empty(models.Model):
    modelid = models.CharField(primary_key=True, max_length=50)

    def __str__(self):
        return '{}'.format(self.modelid)


# Create your models here.
class Student (models.Model):
    nombre = models.CharField(max_length = 50 )
    apellidos = models.CharField(max_length = 100 )
    edad = models.IntegerField()
    email =  models.EmailField()


fuentes_seleccionar = [
    (0, 'No'),
    (1, 'Sí')
]

class Fuentes (models.Model):
    fuente = models.CharField(max_length = 100 )
    ubicacion = models.CharField(max_length = 500 )
    atributo = models.CharField(max_length=100)
    seleccionar = models.IntegerField( null = False, blank = False, choices = fuentes_seleccionar)

    def __str__(self):
        return self.fuente + " | "+ self.atributo


class Datos(models.Model):
    fuente = models.CharField(max_length=100)
    ubicacion = models.CharField(max_length=500)
    atributo = models.CharField(max_length=100)
    seleccionar = models.IntegerField()

modelos_status = [
    (1, 'K-means'),
    (2, 'Modelo2'),
    (3, 'Modelo3')
]

class ModelosClustering(models.Model):
    id_modelo = models.AutoField(primary_key=True)
    nombre_modelo = models.CharField(max_length=50, null=False, blank=False)
    algoritmo = models.CharField(default='K-means', max_length=50, null=False, blank=False)
    fecha_creacion = models.DateTimeField(default=timezone.now)
    estado = models.IntegerField(default=0, blank=True, null=True)
    autor = models.CharField(max_length=100, null=False, blank=False)
    nombre_archivo = models.CharField(default='', max_length=100, null=False, blank=False)
    caracteristicas = models.TextField(default='', null=False, blank=False)
    def __str__(self):
        return self.nombre_modelo
