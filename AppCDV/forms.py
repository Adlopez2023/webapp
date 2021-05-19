from django import forms
from .models import ModelosClustering

'''
class ModelosForm(forms.ModelForm):
    class Meta:
        model = Modelos
        fields = ['modelo']

        def __init__(self, *args, **kwargs):
            super().__init__(*args **kwargs)

            self.fields['modelo'].widget.attrs.update({
                'class': 'form-control',
                #default=1,
            })
'''

class NModel(forms.ModelForm):
	class Meta:
		model = ModelosClustering
		fields = [
			'id_modelo',
			'nombre_modelo',
			'estado',
			'fecha_creacion',
			'algoritmo',
            'autor',
			'nombre_archivo',
			'caracteristicas',
		]
		labels = {
			'id_modelo': 'ID Modelo',
			'nombre_modelo': 'Nombre',
			'estado': 'Estado',
			'fecha_creacion': 'Fecha de creación',
			'algoritmo': 'Algoritmo',
            'autor': 'Autor',
			'nombre_archivo': 'Nombre del archivo',
			'caracteristicas': 'Características seleccionadas',
		}
		widgets = {
			'id_modelo': forms.TextInput(attrs={'class':'form-control'}),
			'nombre_modelo': forms.TextInput(attrs={'class':'form-control'}),
			'estado': forms.HiddenInput(),
			'fecha_creacion': forms.TextInput(attrs={'class':'form-control','readonly':'readonly'}),
			'algoritmo': forms.TextInput(attrs={'class':'form-control','readonly':'readonly'}),
            'autor': forms.TextInput(attrs={'class':'form-control','readonly':'readonly'}),
			'nombre_archivo': forms.TextInput(attrs={'class':'form-control','readonly':'readonly'}),
			'caracteristicas': forms.TextInput(attrs={'class':'form-control','readonly':'readonly'}),
		}

class ModelosForm(forms.ModelForm):
    class Meta:
        model = ModelosClustering
        fields = [
			'id_modelo',
			'nombre_modelo',
			'estado',
			'nombre_archivo',
			'caracteristicas',
		]
        labels = {
			'id_modelo': 'ID Modelo',
			'nombre_modelo': 'Nombre',
			'estado': 'Estado',
			'nombre_archivo': 'Nombre del archivo',
			'caracteristicas': 'Características seleccionadas',

        }
        widgets = {
			'id_modelo': forms.TextInput(attrs={'class':'form-control'}),
			'nombre_modelo': forms.TextInput(attrs={'class':'form-control'}),
			'estado': forms.Select(attrs={'class':'form-control'}),
			'nombre_archivo': forms.TextInput(attrs={'class':'form-control','readonly':'readonly'}),
			'caracteristicas': forms.TextInput(attrs={'class':'form-control','readonly':'readonly'}),
        }


        # def __init__(self, *args, **kwargs):
        #     super().__init__(*args **kwargs)

            # self.fields['estado'].widget.attrs.update({
            #     'class': 'form-control',
            #     'default' : '1',
            # })