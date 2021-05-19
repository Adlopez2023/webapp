#!/usr/bin/python
# -*- coding: utf-8 -*-

"""#Clustering.py version:3 """

#!pip install -Iv mlxtend==0.18.0

import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import glob, os
import xlsxwriter

import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules

# %matplotlib inline

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


"""# Clustering"""

def apply_Kmeans(X, k, id_model, ruta):
  try:
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(X)
    result = kmeans.predict(X)
    result = pd.DataFrame(result)
    result = result.merge(X, left_index=True, right_index=True)
    result.rename(columns={0: 'cluster'}, inplace=True)
    result.to_csv(ruta+"\\static\\Output\\Clustering_results_transformed_"+str(id_model)+".csv", index = False, sep ="|")

    return (result, kmeans )
  except:
    print("ERROR #: COnjunto de datos no puede ser agrupado")
    return (0, 0)


def create_elbow(X, inf, sup, id_model, ruta):
  try:
    distortions = []
    K = range(inf,sup)
    for k in K:
      kmeanModel = KMeans(n_clusters=k)
      kmeanModel.fit(X)
      distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    # plt.show()
    plt.savefig(ruta+'\\static\\imagenes\\elbow_diagram_'+str(id_model)+'.png' , bbox_inches='tight')
    # plt.savefig(ruta+'/static/imagenes/elbow_diagram_'+str(id_model)+'.png')

    return distortions
  except:
    return 0


def data_preparation(X):
  categorical_columns = X.select_dtypes(include='object').columns.tolist()
  preparation = pd.get_dummies(data=X, columns=categorical_columns)
  #preparation.dropna(inplace=True) # cambia el tamaño del dataset cuando hay nulos
  preparation.fillna(-9999, inplace=True)
  cols = preparation.columns
  min_max_scaler = preprocessing.MinMaxScaler()
  preparation = min_max_scaler.fit_transform(preparation)
  preparation = pd.DataFrame(preparation, columns=cols)

  return preparation


def silhouette_method(X, inf, sup):
  try:
    averages = []
    K = range(inf, sup)
    for k in K:
      kmeanModel = KMeans(n_clusters=k)
      cluster_labels = kmeanModel.fit_predict(X)
      silhouette_avg = silhouette_score(X, cluster_labels)
      averages.append(silhouette_score(X, cluster_labels))
      # print("For n_clusters =", k,"The average silhouette_score is :", silhouette_avg)
    # print("The best k is: ", inf + averages.index(max(averages)))
    return inf + averages.index(max(averages))
  except:
    return 2


def best_k_elbow(distortions):
  try:
    for i in range(0, len(distortions) - 1):
      value = distortions[i + 1] / distortions[i]
      # print(str(i+1)+' '+str(i)+' '+ str(value))
      if value > 0.94:
        # print('Best K is ' + str(i))
        break

    return i
  except:
    return 2


"""# Distance between clusters Functions"""


def distance_clusters(result, kmeans, id_model, ruta, top3G, cluster ):
  try:
    clustersAll = result['cluster'].unique()
    f = []
    for i in top3G:
      f.append(int(result.columns.get_loc(i)))

    C = kmeans.cluster_centers_
    colores = ['red', 'green', 'blue', 'cyan', 'yellow', 'grey', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan',
               'purple']
    asignar = []
    for row in clustersAll:
      asignar.append(colores[row])
    # Create the figure
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax, auto_add_to_figure=False)
    ax.scatter(C[:, f[0] - 1], C[:, f[1] - 1], C[:, f[2] - 1], marker='*', c=asignar, s=1000, label=clustersAll)
    plt.title('Distancia entre clusters')
    plt.savefig(ruta+'\\static\\imagenes\\Distance_clusters_' + str(cluster) + '_' + str(id_model) + '.png', bbox_inches='tight')
    # plt.savefig(ruta + '/static/imagenes/Distance_clusters_' + str(id_model) + '.png', bbox_inches='tight')
    return 1
  except:
    print("ERROR #: No se graficó distancia entre clusters")
    return 0


"""# Feature Importance Functions"""


def bin_clusters(uno, elem):
  ### Binarizar los clusters ( # cluster - cluster a evaluar y 0 - los demas)
  if uno == elem:
    return 1
  else:
    return 0


# Generar la lista de caracteristicas mas representativas
def GetFeaturesList(resultF, cluster, id_model, ruta):
  resultF = resultF.copy()
  try:
    if cluster == 'all':
      modelObjective = 'multiclass'
      model = XGBClassifier(
        use_label_encoder=False,
        # HyperparameterTuning
        nthread=-1,
        objective="multi:softmax",  # "multi:softprob",  # "multi:softmax",
        eval_metric="auc",
        random_state=42
      )

    elif int(cluster) >= 0:
      modelObjective = 'binary'
      resultF['cluster'] = resultF.apply(lambda row: bin_clusters(int(cluster), row['cluster']), axis=1)
      model = XGBClassifier(
        use_label_encoder=False,
        # HyperparameterTuning
        max_depth=3,  #
        eta=0.1,  #
        subsample=0.7,  #
        colsample_bytree=0.4,  #
        gamma=0.3,  #
        max_delta_step=5,  #
        n_estimators=200,  # python
        min_child_weight=1,  #

        nthread=-1,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
      )
    else:
      modelObjective = 'Error'
      print('Error: Cluster inválido')
  except:
    modelObjective = 'multiclass'
    print('Error: Cluster inválido')

  X = resultF.drop(["cluster"], axis=1)

  y = resultF[['cluster']]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  y_train = y_train.values.ravel().astype(np.int64)
  y_test = y_test.values.ravel().astype(np.int64)

  # fit model on training data
  model.fit(X_train, y_train)

  # # make predictions for training data
  # y_predTrain = model.predict(X_train)
  # predictionsTrain = [round(value) for value in y_predTrain]

  # Feature importance
  # Gain
  feature_important = model.get_booster().get_score(importance_type='gain')
  keys = list(feature_important.keys())
  values = list(feature_important.values())
  data_g = pd.DataFrame(data=values, index=keys, columns=["gain"]).reset_index()
  # Cover
  feature_important = model.get_booster().get_score(importance_type='cover')
  keys = list(feature_important.keys())
  values = list(feature_important.values())
  data_c = pd.DataFrame(data=values, index=keys, columns=["cover"]).reset_index()
  # Concat Gain and Cover
  final = pd.merge(data_g, data_c, how="left", on=["index"]).sort_values(by="gain", ascending=False)

  # Export feature importance list

  final.to_csv(ruta + '\\static\\Output\\XGBOOST_feature_importance_' + modelObjective + '_' + str(cluster) + '_' + str(
    id_model) + '.csv', sep="|", index=False)
  # final.to_csv(ruta+'/static/Output/XGBOOST_feature_importance_' + modelObjective + '_' + str(cluster) + '_' + str(id_model) + '.csv', sep="|",index=False)


  # Create the histogram for Gain
  finalC = final.drop(['cover'], axis=1).set_index("index")

  finalC = finalC.head(10).sort_values(by="gain", ascending=True)
  finalC.plot(kind='barh', y='gain', color='blue')
  plt.ylabel('Atributos')
  plt.xlabel('Ganancia')
  plt.title('Lista de atributos más representativos')
  # plt.show
  plt.savefig(ruta + '\\static\\imagenes\\XGain_' + modelObjective + '_' + str(cluster) + '_' + str(id_model) + '.png',
              bbox_inches='tight')
  # plt.savefig(ruta+'/static/imagenes/Gain_' + modelObjective + '_' + str(cluster) + '_' + str(id_model) + '.png', bbox_inches='tight')

  # Create the histogram for Cover
  finalC = final.drop(['gain'], axis=1).set_index("index")
  finalC = finalC.head(10).sort_values(by="cover", ascending=True)
  finalC.plot(kind='barh', y='cover', color='blue')
  plt.ylabel('Atributos')
  plt.xlabel('Cobertura')
  plt.title('Lista de atributos más representativos')
  # plt.show
  plt.savefig(ruta + '\\static\\imagenes\\XCover_' + modelObjective + '_' + str(cluster) + '_' + str(id_model) + '.png',
              bbox_inches='tight')
  # plt.savefig(ruta+'/static/imagenes/XCover_' + modelObjective + '_' + str(cluster) + '_' + str(id_model) + '.png', bbox_inches='tight')

  # Identify indexes for Top3 features
  fl = final.sort_values(by="gain", ascending=False).head(3)
  fll = fl.iloc[:, 0].unique()

  return fll  # final.iloc[:, 0].unique()




"""# Association Rules"""

def data_preparation_association(X):
  try:
    number_columns = X.select_dtypes(include='number').columns.tolist()
    if len(number_columns) > 0:
      for column in number_columns:
          # if (float(X[column].min()) != 0.0) or (float(X[column].max()) != 1.0):
            # X[column] = pd.cut(X[column], bins=10, include_lowest=True)
          if X[column].nunique() >= 10:
            X[column] = pd.cut(X[column], bins=10)  
          X = X.merge(pd.get_dummies(X[column], prefix=column+"|"), left_index=True, right_index=True)
          X.drop([column], axis=1, inplace=True)
  except:
    print("no hay atributos numéricos en el dataset")

  try:
    categorical_columns = X.select_dtypes(exclude=['uint8']).columns.tolist()
    if len(categorical_columns) > 0:
      for column in categorical_columns:
        X = X.merge(pd.get_dummies(X[column], prefix=column+"|"), left_index=True, right_index=True)
        X.drop([column], axis=1, inplace=True)
  except:
    print("no hay atributos categoricos en el dataset")
  return X


def association_rules_fpgrowth(X):
  X = X.fillna(-999)
  print(X.shape)
  frequent_itemsets = fpgrowth(X, min_support=0.6, use_colnames=True)
  ar = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

  return ar

def association_rules_apriori(X):
  X = X.fillna(-999)
  frequent_itemsets = apriori(X, min_support=0.6, use_colnames=True)
  ar = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

  return ar

def association_rules_fpmax(X):

  frequent_itemsets = fpmax(X, min_support=0.6, use_colnames=True)
  ar = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

  return ar

def analize_clusters(input, confidence, support):

  association_rules = pd.DataFrame()
  k = input.cluster.unique()
  print("k input", k)
  for i in k:
    data = input.copy()
    data = data[(data['cluster'] == i)]
    data.drop(['cluster'], axis=1, inplace=True)
    print("data_preparation_Association", data.shape)
    dataset = data_preparation_association(data)
    print("association_rules_fpgrowth", dataset.shape)
    rules = association_rules_fpgrowth(dataset)
    rules = rules[(rules['confidence'] >= confidence) & (rules['support'] >= support)]
    rules['cluster'] = i
    association_rules = association_rules.append(rules)

    # rules = association_rules_apriori(dataset)
    # rules[(rules['confidence'] >= confidence) & (rules['support'] >= support)]

  return association_rules


def split_text_antecedent(elem):
  tt = str(elem).replace("_", " ").split("'")
  tt2 = tt[1].split("|")
  return "Cuando el atributo '" + str(tt2[0]) + "' posee el valor de " + str(tt2[1])



def split_text_consequents(elem):
  tt = str(elem).replace("_", " ").split("'")
  tt2 = tt[1].split("|")
  return "Entonces el atributo '" + str(tt2[0]) + "' posee el valor de " + str(tt2[1])

def explain_rules(rules, id_modelo, rutaf):
  try:
    rules['antecedents_description'] = rules['antecedents'].map(split_text_antecedent)
    rules['consequents_description'] = rules['consequents'].map(split_text_consequents)
    rules = rules[['cluster', 'antecedents_description', 'consequents_description', 'support', 'confidence']]
    rules.to_csv(rutaf + "\\static\\Output\\Association_rules_" + str(id_modelo) + ".csv", index=False, sep = "|")
    return 1
  except:
    print("ERROR #: No se explicaron las reglas de asociación")
    return 0


"""# Heat Map """

def heat_map(rutaf, id_modelo):
    print("ingresó a heat_map")
    # leer dataset original agrupado
    mc = pd.read_csv(rutaf+"\\static\\Output\\Clustering_dataO_"+str(id_modelo)+".csv", sep="|")
    print(mc.shape)
    # Impute null values
    NULL_CUT = -9999
    mc.fillna(NULL_CUT, inplace=True)

    # create header
    clusters = mc.cluster.sort_values(ascending=True).unique()
    encabezado = 'Tipo|columna|variable'
    for l in clusters:
      encabezado = encabezado + "|C_" + str(l)

    # crear mapa de calor según tipo de variable
    N_QUANTILES = 20
    cum = ""
    with open(rutaf + '\\static\\Output\\Mapa_calor_all_n_' + str(id_modelo) + '.csv', 'w', encoding='latin-1') as csv_file:
      csv_file.write(encabezado + '\n')
      for col in mc.columns:
        # Evaluate Catagorical features and booleans
        if (mc[col].dtype == 'O') or ((mc[col].nunique() <= 2) and (mc[col].dtype != 'O')):
          print("Categorica - ", col)
          for var in mc[col].unique():

            cum = str(id_modelo) + "|" + str(col) + "|" + str(var)
            for i in clusters:
              n = mc[col][(mc[col] == var) & (mc['cluster'] == i)].shape[0]
              cum = cum + "|" + str(n)
            csv_file.write(cum + '\n')
        elif (mc[col].nunique() >= 3) and (mc[col].dtype != 'O'):

          try:
            print("Numérica - ", col)
            sorted_p = mc[(mc[col] != NULL_CUT) & (~mc[col].isna())][col].sort_values(ascending=True)
            n_values = len(np.unique(sorted_p))

            cutpoints = list(
              range(max(int(min(sorted_p)), 0), n_values + 1, max(int(round(((n_values + 1) / N_QUANTILES), 0)), 1)))
            p_breaks = np.sort(cutpoints)
            limits = np.array([0, NULL_CUT])
            u_breaks = np.unique(np.concatenate((p_breaks, limits)))

            labels = np.array(str(NULL_CUT) + "_a_0")
            for j in range(1, len(u_breaks)):
              try:
                labels = np.append(labels, str(u_breaks[j]) + "_a_" + str(u_breaks[j + 1]))
              except:
                j += 1

            mc[col + '_bin'] = pd.cut(x=mc[col], bins=u_breaks
                                      , labels=labels)
            ### Solucion temporal porque hay NA after cut, se asigna a la ultima categoria
            mc[col + '_bin'] = mc[col + '_bin'].astype("O")
            mc[col + '_bin'][mc[col + '_bin'].isna()] = labels[-1]

            for var in mc[col + '_bin'].unique():
              cum = str(id_modelo) + "|" + str(col) + "|" + str(var)

              for k in clusters:
                n = mc[col + '_bin'][(mc[col + '_bin'] == var) & (mc['cluster'] == k)].shape[0]
                cum = cum + "|" + str(n)
              csv_file.write(cum + '\n')
          except:
            print("*", col, "- Tratada como categorica!")
            for var in mc[col].unique():

              cum = str(id_modelo) + "|" + str(col) + "|" + str(var)
              for i in clusters:
                n = mc[col][(mc[col] == var) & (mc['cluster'] == i)].shape[0]
                cum = cum + "|" + str(n)
              csv_file.write(cum + '\n')
        else:
          print("Columna no incluida en el mapa de calor", col)

    mcp = pd.read_csv(rutaf + '\\static\\Output\\Mapa_calor_all_n_' + str(id_modelo) + '.csv', sep="|", encoding='latin-1',
                      low_memory=False)
    filas = mcp.shape[0]
    for f in range(0, filas):
      suma = 0
      for h in clusters:
        suma = suma + mcp["C_" + str(h)].iloc[f]
      for h in clusters:
        mcp["C_" + str(h)].iloc[f] = round(mcp["C_" + str(h)].iloc[f] * 100 / suma, 2)
    mcp.to_csv(rutaf + '\\static\\Output\\Mapa_calor_all_p_' + str(id_modelo) + '.csv', sep="|", index=False)
    print(mcp.shape)
    return 1



"""# Pandas Profiling"""

def profiling(rutaf, id_modelo):
  try:
    dataO = pd.read_csv(rutaf + "\\static\\Output\\Clustering_dataO_" + str(id_modelo) + ".csv", sep="|", low_memory=False)
    print(dataO.shape)
    from pandas_profiling import ProfileReport
    for clusteri in dataO.cluster.unique():
      dataOc = dataO[dataO.cluster == clusteri]
      print(dataOc.shape)
      profile = ProfileReport(dataOc)  # , minimal = True
      profile.to_file(
        output_file= rutaf + "\\static\\Output\\Perfilamiento_cluster_" + str(clusteri) + "_" + str(id_modelo) + ".html")
    return 1
  except:
    print("ERROR # : No fue posible general el perfilamiento de cada cluster")
    return 0

"""# Export csv and imagens to Excel"""

def export_excel(id_modelo, rutaf, archivos, imagenes):
  print("Exportar a Excel..")
  #Exportar archivos csv
  writer = pd.ExcelWriter(rutaf+"\\static\\consolidado\\Resultados_consolidado_"+str(id_modelo)+".xlsx", engine='xlsxwriter')
  i = 1
  for archivo in archivos:
      try:
        df = pd.read_csv(archivo, low_memory = False, sep="|", encoding= "latin-1")
        fileName = str(archivo.split('\\Output\\')[1]).split(".")
        df.to_excel(writer, sheet_name=str(i)+"_"+str(fileName[0])[-26:], index = False)
        # print(fileName[0], fileName[1])
          
      except:
          print("ERROR # : No se puede leer o exportar el archivo", archivo)
      i+=1
  writer.close()
  
  # Exportar imagenes
  workbook = xlsxwriter.Workbook(rutaf+"\\static\\consolidado\\imagenes_consolidado_"+str(id_modelo)+".xlsx")
  i = 1
  for imagen in imagenes:
      try:
          fileName = str(imagen.split('\\imagenes\\')[1]).split(".")
          worksheet = workbook.add_worksheet(str(i)+"_"+str(fileName[0])[-26:])
          worksheet.insert_image('A1', imagen, {'x_scale': 0.8, 'y_scale': 0.8})
          print(fileName[0], fileName[1])
      except:
          print("ERROR # : No se puede leer o exportar el archivo", imagen)
      i+=1

  workbook.close()
  return 1

### Clustering - views.py

if __name__ == '__main__':

  print("clustering.py")


