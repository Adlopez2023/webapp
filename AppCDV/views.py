from django.http import HttpResponse,JsonResponse, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from .forms import NModel
from django.db.models import Count
from django.urls import reverse, reverse_lazy
from AppCDV.models import empty, ModelosClustering
from django.views.generic import ListView, CreateView, UpdateView, DeleteView
from django.core.cache import cache
from django.conf import settings
from django.core.cache.backends.base import DEFAULT_TIMEOUT

# importar librerias adicionales
import glob
import pandas as pd
import sys
import os
from pandasgui import show
import sweetviz as sv

#importar modelo de datos
from AppCDV.models import Fuentes, ModelosClustering
from AppCDV.forms import NModel, ModelosForm

#importar archivo de funciones del clustering
import threading, time
from .clustering_v3 import *

rutaf = os.path.dirname(os.path.realpath(__file__))
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Create your views here.
def index(request):
    context={}
    idUser=''
    #se borra información del usuario en caché
    if 'currentUser'  in cache:
        cache.delete('currentUser')
    if 'idUser' in cache:
        idUser = cache.get('idUser')
        cache.delete('idUser')
    #se borra información del archivo en caché
    if 'NomArchivo'+idUser  in cache:
        Nombre_archivo = cache.get('NomArchivo'+idUser)
        cache.delete('NomArchivo'+idUser)
        if idUser+'_'+Nombre_archivo in cache:
            cache.delete(idUser+'_'+Nombre_archivo)
    if 'listaVar'+idUser  in cache:
        cache.delete('listaVar'+idUser)
    return render(request, 'index.html', context)

def up_file(request):
    context={}
    listaVar=''
    nombreArchivo=''
    something=False
    cols = list()
    idUser=str(request.user.id)
    # userName = str(request.user.username)
    if request.method == "POST":
        if bool(request.FILES.get('myFile', False)) == True:
            dataSet = request.FILES["myFile"]
            try:
                csv=pd.read_csv(dataSet, low_memory=False, encoding='latin-1', sep=",")
                print(csv.shape)
                if csv.shape[1] == 0:
                    csv=pd.read_csv(dataSet, low_memory=False, encoding='latin-1', sep="|")
            except:
                print("ERROR #: Archivo separado por coma, o con estructura inconsistente")

            for col in csv.columns:
                cols.append(col)
                something=True
                nombreArchivo='Archivo cargado: '+dataSet.name
            #Se guarda en caché el nombre del archivo y el archivo
            print('la identificación usuario: '+idUser)
            if 'NomArchivo'+idUser not in cache:
                cache.set('NomArchivo'+idUser, dataSet.name)
                cache.set(idUser+'_'+dataSet.name,csv)
            context={"something": something, "cols": cols, "listaVar":listaVar, "nombreArchivo": nombreArchivo}
        else: 
            listaVar = request.POST.get('ListSelec')
            #Se guarda en caché el listado de variables seleccionadas
            #if 'listaVar'+idUser not in cache:                
            cache.set('listaVar'+idUser, listaVar)
            something=False
            context={"something": something, "cols": cols, "listaVar":listaVar, "nombreArchivo": nombreArchivo}
    else: #Limpia caché cuando no hace post
        if 'NomArchivo'+idUser  in cache:
            Nombre_archivo = cache.get('NomArchivo'+idUser)
            cache.delete('NomArchivo'+idUser)
            if idUser+'_'+Nombre_archivo in cache:
                cache.delete(idUser+'_'+Nombre_archivo)
        if 'listaVar'+idUser  in cache:
            cache.delete('listaVar'+idUser)        
    return render(request, 'UploadFile1.html', context)

def result(request):
    #if (request.method == 'POST'):
    context = {
    "resultado": "02",
    }
    return render(request, 'results.html', context)

######


def home_l(request):

    # consultar los ModelosClustering (valores unicos)
    lista_modelos = ModelosClustering.objects.order_by().values('nombre_modelo').distinct()
    # Mostrar por consola los resultados
    for k in lista_modelos:
        for h in k.values():
            print(h)
    context['prueba'] = "home_l"
    #return HttpResponse("Listado de estudiantes")
    return render(request, 'consultar_fuentes.html', context)

# Create your views here.

class HomeModelsLists(ListView):
    model = ModelosClustering
    template_name = 'home.html'
    paginate_by = 5

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        idUser=''
        #Se captura el usuario autenticado
        currentUser=self.request.user
        idUser=str(currentUser.id)
        if 'currentUser' not in cache:
            cache.set('currentUser', currentUser)
        if 'idUser' not in cache:
            cache.set('idUser', idUser)
        #Se limpia la caché cuando ingresa a listado de modelos:
        if 'NomArchivo'+idUser  in cache:
            Nombre_archivo = cache.get('NomArchivo'+idUser)
            cache.delete('NomArchivo'+idUser)
            if idUser+'_'+Nombre_archivo in cache:
                cache.delete(idUser+'_'+Nombre_archivo)
        if 'listaVar'+idUser  in cache:
            cache.delete('listaVar'+idUser)       
        return context

class NewModel(CreateView):
    model = ModelosClustering
    form_class = NModel
    template_name = 'NewModel.html'
     
    def form_valid(self, form):
        record = form.save()
        nombreArchivo=''
        archivoCSV=''
        #se recupera el nuevo id_modelo generado por el form
        id_modelo = record.id_modelo

        #Se recupera información del usuario
        userName = str(self.request.user.username)
        idUser=str(self.request.user.id)

        #Se recupera el nombre del archivo
        if 'NomArchivo'+idUser in cache:
            nombreArchivo = cache.get('NomArchivo'+idUser)
        if idUser+'_'+nombreArchivo in cache:
            archivoCSV = cache.get(idUser+'_'+nombreArchivo)
            #cambio a slash para so mac
            archivoCSV.to_csv(rutaf+'//static//Data//'+nombreArchivo+'_'+ userName+'.csv', sep = "|", index = False)
            print("Archivo guardado en Servidor Django: nombre_archivo + autor")

        modelo = ModelosClustering.objects.get(id_modelo=id_modelo)
        return redirect('entrenar_modelos',str(id_modelo))
        
        #return HttpResponseRedirect('../home/')
        #return super().form_valid(form)
    
    def get_form(self, form_class=None):        
        form = super(NewModel, self).get_form(form_class)
        Nombre_archivo=''
        listaVar=''
        currentUser=''
        idUser=str(self.request.user.id)
        if 'NomArchivo'+idUser in cache:
            Nombre_archivo = cache.get('NomArchivo'+idUser)
            dat=cache.get(idUser+'_'+Nombre_archivo)
            #print(dat.head(2))
        if 'listaVar'+idUser in cache:
            listaVar = cache.get('listaVar'+idUser)
        if 'currentUser' in cache:
            currentUser = cache.get('currentUser')
        else:
            currentUser=self.request.user
            cache.set('currentUser', currentUser)
        form.fields['nombre_archivo'].initial = Nombre_archivo 
        form.fields['caracteristicas'].initial = listaVar
        form.fields['autor'].initial = currentUser
        return form
    
    def render_to_response(self,context, **response_kwargs):
        if self.request.is_ajax():
            NomArchivo = context["NomArchivo"]
            listaVar = context["listaVar"]
            return JsonResponse({'NomArchivo':NomArchivo,'listaVar':listaVar})
        else:
            response_kwargs.setdefault('content_type', self.content_type)
            return self.response_class(
                request = self.request,
                template = self.get_template_names(),
                context = context,
                using = self.template_engine,
                **response_kwargs
            )
    def get_success_url(self):
        return reverse('nuevo_modelo')

class ResultsModel(UpdateView):
    model = ModelosClustering
    form_class = NModel
    template_name = 'results.html'

    def get_form(self, form_class=None):        
        form = super(ResultsModel, self).get_form(form_class)
        return form

    def render_to_response(self,context, **response_kwargs):
        #se recupera el id del modelo seleccionado (pasado por la url)
        id_modelo=str(self.object.id_modelo)
        archivos_modelo=listar_archivos('Output',id_modelo)
        context['id_modelo']=id_modelo
        
        context['Resultados']=obtenerArchivo(archivos_modelo,'Clustering_results_')
        context['Xgboost']=obtenerArchivo(archivos_modelo,'XGBOOST_')
        
        try:
            mcdfp = pd.read_csv(rutaf+'//static//Output//Mapa_calor_all_p_'+id_modelo+'.csv', sep = "|", encoding='latin-1')
            mcdfp.drop(['Tipo'], axis = 1, inplace = True)
            mcdfn = pd.read_csv(rutaf+'//static//Output//Mapa_calor_all_n_'+id_modelo+'.csv', sep = "|", encoding='latin-1')
            mcdfn.drop(['Tipo'], axis = 1, inplace = True)

            context['DataFramep'] = mcdfp
            context['DataFramen'] = mcdfn
            
            are = pd.read_csv(rutaf+'//static//Output//Association_rules_'+id_modelo+'.csv', sep = "|", encoding='latin-1')
            context['DataFrameAR'] = are


        except:
            print("ERROR #: No se encontraron los archivos de resultados")
        return self.response_class(
                request = self.request,
                template = self.get_template_names(),
                context = context,
                using = self.template_engine,
                **response_kwargs
            )

    def get_success_url(self):
        return reverse('resultados')




######################################################################
def obtenerArchivo(lista, subcadena):
    for item in lista:
        if subcadena in item:
            return item


def listar_archivos(carpeta, id_model):
    # consultar archivos csv en ruta predefinida
    allFiles = []
    try:
        
        allFiles=allFiles+glob.glob(os.path.dirname(os.path.realpath(__file__))+'\\'+carpeta+'\\*_'+str(id_model)+'.*')

        '''
        allFiles = allFiles + glob.glob(os.path.dirname(os.path.realpath(__file__))+'/'+carpeta+"/*_"+ id_model + ".*")'''

    except:
        print("ERROR #: No hay archivos en la carpeta seleccionada -", carpeta)
    return allFiles

def eliminar_archivos (id_modelo):
    ### DETENER EL HILO ANTES DE... !!!
    print("Eliminado archivos creados para el modelo:", id_modelo)

    archivos = listar_archivos("static\\*", id_modelo)
    if len(archivos) > 0 :
        for archivo in archivos:
            try:
                print(archivo)
                os.remove(archivo)
            except:
                print("Archivo no encontrado - ", archivo)
        return 1
    else:
        print("ERROR #: No hay archivos en la carpeta seleccionada - Output")
        return 0


class DeleteModel(DeleteView):
    
    model = ModelosClustering
    template_name = 'delete_models.html'
    def get_success_url(self): 	
        # Identificar el modelo para eliminar los archivos asociados
        print(str(self.request).split("/")[2])
        eliminar_archivos(str(self.request).split("/")[2])		
        return reverse_lazy('listar_modelos')

def delete_model(request, id_modelo=None):
    if id:
        modelToDelete = get_object_or_404(ModelosClustering,id_modelo=id_modelo)
        modelToDelete.delete()    
        eliminar_archivos(id_modelo) 
    return HttpResponseRedirect('/home')


def DeleteModelT(request, id_modelo):
    modelos = ModelosClustering.objects.get(id_modelo=id_modelo)
    modelos.delete()    
    eliminar_archivos(id_modelo) 
    
    return redirect('listar_modelos',)


def Ejecucion_clustering1 ( id_model, ruta, autor, nombre_archivo, caracteristicas, confidence, support):
    try:
        print("Iniciando entrenamiento K-Means!")
        modelo = ModelosClustering.objects.get(id_modelo=id_model)

        print("Cargando el conjunto de datos procesado y filtrado")
        print(ruta+'\\static\\Data\\'+nombre_archivo+'_'+autor+'.csv')
        print(caracteristicas.split(','))
        try:
            data = pd.read_csv(ruta+'\\static\\Data\\'+nombre_archivo+'_'+autor+'.csv', 
                                sep = ",", low_memory = False,
                                usecols = caracteristicas.split(',')) 
        except:
            data = pd.read_csv(ruta+'\\static\\Data\\'+nombre_archivo+'_'+autor+'.csv', 
                                sep = "|", low_memory = False,
                                usecols = caracteristicas.split(',')) 
        
        print("data filtrada ", data.shape)


        if data.shape[0] > 0:
           
            dataO = data.copy()
            print("Preparación de los datos...data_preparation")
            data_prepared = data_preparation(data)
            print(data_prepared.shape)

            print("Seleccionando el mejor K...Silhouette_method")
            ks = min(silhouette_method(data_prepared, 2, 15) , 11)
            print(ks)
            
            print("Creando diagrama de codo...create_elbow")
            distortionsE = create_elbow(data_prepared, 2, 11, id_model, ruta)
            print("Seleccionando el mejor K...best_k_elbow")
            kE = min(best_k_elbow(distortionsE) , 11)
            print(kE)
            
            k = max(2,min(ks, kE)) # Seleccionar el valor mínimo, siempre que sea mayor o igual a 2
            print("K seleccionado: ", k)

            
            print("Aplicando K-Means...apply_Kmeans")
            results, kmeans = apply_Kmeans(data_prepared, k, id_model, ruta)
            print(results.shape)
            
            if (dataO.shape[0] == results.shape[0]):
                print("Asignar cluster a data original")
                dataO = pd.merge(dataO,results[['cluster']], left_index=True, right_index=True)
                # dataO['cluster'] = results['cluster']
                dataO.to_csv(ruta+"\\static\\Output\\Clustering_dataO_"+str(id_model)+".csv", index = False, sep ="|")
                print(dataO.shape)
                #generar csv mapa de calor
                heat_map(ruta, id_model)
                #perfilamiento datos
                profiling(ruta, id_model)
            else:
                print("ERROR # : Dataset original no coincide con results de K-means")
            
            
#             # Revisar en dataset pequeño no converge multiclase            
#             try:
#                 print("Generar caracteristicas representativas ALL...GetFeaturesList")
#                 feat = GetFeaturesList(results, 'all', id_model, ruta)
#                 print("Graficando distancia entre clusters ALL...distance_clusters")
#                 distance_clusters(results, kmeans, id_model, ruta, feat)
#                 print(feat)
#             except:
#                 print("ERROR # : Error multiclase - Cluster ALL")
            

            print("Generar caracteristicas representativas por cluster...GetFeaturesList")
            print(results.cluster.unique())
            for clusteri in results.cluster.unique():
                try:
                    print("cluster #", clusteri)
                    print(results[results.cluster == clusteri].shape)
                    feat = GetFeaturesList(results, clusteri, id_model, ruta)
                    print(feat)
                    
                except:
                    print("ERROR # : Verificar el número de Cluster ", clusteri)
            
            try:
                print("Graficando distancia entre clusters...distance_clusters")        
                distance_clusters(results, kmeans, id_model, ruta, feat, 'all')
            except:
                print("ERROR #: Al graficar distancia entre clusters...distance_clusters")
            
            #  Análisis de asociación 
            print("Análisis de asociación ... analize_clusters", dataO.shape)
            t = analize_clusters(dataO, confidence, support)
            
            print("Explicar reglas ... explain_rules ", t.shape)
            explain_rules(t, id_model, ruta)

            #exportar resultados a excel
            archivos = listar_archivos("static\\Output", id_model)
            imagenes = listar_archivos("static\\imagenes", id_model)
            export_excel(id_model, ruta, archivos, imagenes)
            
            
            try:
                print(dataO.columns)
                #Visualizar datos agrupados 
                # gui=show(dataO) #
                dataO['cluster'] = dataO['cluster'].apply(str)
                report=sv.compare_intra(dataO,dataO["cluster"]=="0",dataO.cluster.unique(),pairwise_analysis="on")
                report.show_html(  filepath=rutaf+'//static//consolidado//Visualización_resultados'+id_model+'.html', 
                open_browser=True, 
                layout='widescreen', 
                scale=0.7)

                #Modificar archivo sweetviz_defaults.ini en el ambiente virtual logo =0
            except:
                print("ERROR #: Al visualizar datos agrupados")

            print("Entrenamiento finalizado, modelo:",str(id_model))
            return 1


        else:
            print("ERROR #: No hay datos para el entrenamiento, revisar formato")
            return 0

    except:
        print("ERROR #: Error durante la ejecución del clustering1")
        return 0



def Train(request,id_modelo):
    print("Train id_modelo ",str(id_modelo))
    modelo = ModelosClustering.objects.get(id_modelo=id_modelo)
    
    return render(request, 'TrainModel.html', {'modelo':modelo, 'estado':1} )


def Training(request,id_modelo):
    print("Training id_modelo ",str(id_modelo))
    modelo = ModelosClustering.objects.get(id_modelo=id_modelo)
    autor = modelo.autor
    nombre_archivo = modelo.nombre_archivo
    caracteristicas = modelo.caracteristicas
    rutaf = os.path.dirname(os.path.realpath(__file__))
    print(rutaf)

    print(rutaf)
    if request.method == "GET":
        

        print("Entrenando (Manual) - ", id_modelo)
        modelo.estado = 3
        modelo.save()
        # definir confianza y soporte para el análisis de asociación
        confidence = 0.8
        support = 0.8
        if Ejecucion_clustering1(id_modelo, rutaf, autor, nombre_archivo, caracteristicas, confidence, support) == 1:
            print("Entrenamiento exitoso del modelo - ", id_modelo)
            modelo.estado = 1
            modelo.save()
            return redirect('resultados', id_modelo)

        else:
            print("ERROR #: Entrenamiento Fallido")
            modelo.estado = 2
            modelo.save()

    return render(request, 'TrainModel2.html', {'modelo':modelo,'estado':0} )


######