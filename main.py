#Universidad del Valle de Guatemala
#Mineria de Datos
#HT4 Regresion
#Integrantes
#Bryann Alfaro
#Diego de Jesus
#Julio Herrera

'''
Referencias
Material brindado en clase
https://stackoverflow.com/questions/52404857/how-do-i-plot-for-multiple-linear-regression-model-using-matplotlib
https://medium.com/swlh/multi-linear-regression-using-python-44bd0d10082d

'''

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pyclustertend
import random
import graphviz
import sklearn.mixture as mixture
from sklearn import preprocessing, tree
import scipy.cluster.hierarchy as sch
from copy import copy
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from scipy.stats import normaltest
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
houses = pd.read_csv('train.csv', encoding='latin1', engine='python')

'''

#Conocimiento de datos
print(houses.head())

#Cantidad de observaciones y variables en la base
print(houses.shape)

#Medidas estadisticas.
print(houses.describe().transpose())

print(houses.select_dtypes(exclude=['object']).info())'''


'''#Casas que ofrecen todas las utilidades
print(houses['Utilities'].value_counts())

plt.bar(houses['Utilities'].value_counts().sort_index().dropna().index, houses['Utilities'].value_counts().sort_index().values, color='red')
plt.title('Grafico de barras para utilidades')
plt.xlabel('Utilidad')
plt.xticks(rotation=90)
plt.ylabel('Cantidad de casas')
plt.tight_layout()
plt.show()

#Calidad de casas predominante
print(houses['OverallCond'].value_counts())

plt.bar(houses['OverallCond'].value_counts().sort_index().dropna().index, houses['OverallCond'].value_counts().sort_index().values, color='red')
plt.title('Grafico de barras para condicion de las casas')
plt.xlabel('Condicion')
plt.ylabel('Cantidad de casas')
plt.show()

#Año de mas y menos produccion de casas para
print(houses['YearBuilt'].value_counts().sort_values(ascending=False).head(1))
print(houses['YearBuilt'].value_counts().sort_values(ascending=True).head(15))

#Capacidad de carros de las 5 casas mas caras y baratas

print(houses.sort_values(by='SalePrice', ascending=False)[['GarageCars','SalePrice']].head(5))
print(houses.sort_values(by='SalePrice', ascending=True)[['GarageCars','SalePrice']].head(5))

#Condicion de garage y calidad de la cocina de las 5 casas mas caras
print(houses.sort_values(by='SalePrice', ascending=False)[['GarageCond','KitchenQual','SalePrice']].head(5))'''

houses_clean = houses.select_dtypes(exclude='object').drop('Id', axis=1)

'''#preprocesamiento
corr_data = houses_clean.iloc[:,:]
mat_correlation=corr_data.corr() # se calcula la matriz , usando el coeficiente de correlacion de Pearson
plt.figure(figsize=(16,10))

#Realizando una mejor visualizacion de la matriz
sns.heatmap(mat_correlation,annot=True,cmap='BrBG')
plt.title('Matriz de correlaciones  para la base Houses')
plt.tight_layout()
plt.show()'''

# Seleccion de variables
houses_df = houses_clean[['OverallQual', 'OverallCond', 'GrLivArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'Fireplaces',
'GarageCars', 'GarageArea', 'GarageYrBlt','TotRmsAbvGrd','SalePrice']]
'''
print(houses_df.head().dropna())
print(houses_df.info())
print(houses_df.describe().transpose())
'''
houses_df.fillna(0)

#normalizar
df_norm  = (houses_df-houses_df.min())/(houses_df.max()-houses_df.min())
#print(movies_clean_norm.fillna(0))
houses_df_final = df_norm.fillna(0)
'''#Analisis de tendencia a agrupamiento

#Metodo Hopkings

random.seed(200)
print(pyclustertend.hopkins(houses_df_final, len(houses_df_final)))

#Grafico VAT e iVAT
x = houses_df_final.sample(frac=0.1)
pyclustertend.vat(x)
plt.show()
pyclustertend.ivat(x)
plt.show()

# Numero adecuado de grupos
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(houses_df_final)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Grafico de codo')
plt.xlabel('No. Clusters')
plt.ylabel('Puntaje')
plt.show()

#Kmeans
clusters=  KMeans(n_clusters=3, max_iter=300) #Creacion del modelo
clusters.fit(houses_df_final) #Aplicacion del modelo de cluster

houses_df_final['cluster'] = clusters.labels_ #Asignacion de los clusters
print(houses_df_final.head())

pca = PCA(2)
pca_movies = pca.fit_transform(houses_df_final)
pca_movies_df = pd.DataFrame(data = pca_movies, columns = ['PC1', 'PC2'])
pca_clust_movies = pd.concat([pca_movies_df, houses_df_final[['cluster']]], axis = 1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Clusters de casas', fontsize = 20)

color_theme = np.array(['red', 'green', 'blue', 'yellow','black'])
ax.scatter(x = pca_clust_movies.PC1, y = pca_clust_movies.PC2, s = 50, c = color_theme[pca_clust_movies.cluster])

plt.show()
print(pca_clust_movies)

houses_df['Cluster'] = houses_df_final['cluster']
print((houses_df[houses_df['Cluster']==0]).describe().transpose())
print((houses_df[houses_df['Cluster']==1]).describe().transpose())
print((houses_df[houses_df['Cluster']==2]).describe().transpose())

houses_df.pop('Cluster')'''

houses_copy = (houses_df.copy())
y_reg = houses_df.pop('SalePrice')
x_reg = houses_df

print(y_reg.shape, x_reg.shape)

x_reg.pop('MasVnrArea')
x_reg.pop('GarageYrBlt')

random.seed(5236)

x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.3, train_size=0.7, random_state=0)
print(x_train_reg.shape,x_test_reg.shape,y_train_reg.shape,y_test_reg.shape)

# use all as predictor
x= x_train_reg.values
y= y_train_reg.values
x_t = x_test_reg.values
y_t = y_test_reg.values

linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred = linear_model.predict(x_t)

print('All vars Coefficients: \n', linear_model.coef_)
print('All vars Mean squared error: %.2f' % mean_squared_error(y_test_reg, y_pred))
print("All vars R squared: %.2f"%r2_score(y_test_reg, y_pred))

#print(x,type(x_reg))

#Analisis VIF de todas
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(houses_df.values, i)
                          for i in range(houses_df.shape[1])]
vif["features"] = houses_df.columns
print(vif.describe)

#Mapa de correlacion
corr =  houses_copy.corr()
print('Pearson correlation coefficient matrix of each variables:\n', corr)
plt.figure(figsize=(16,10))
#Realizando una mejor visualizacion de la matriz
sns.heatmap(corr,annot=True,cmap='BrBG')
plt.title('Matriz de correlaciones')
plt.tight_layout()
plt.show()

# Regression values
x= x_train_reg['OverallQual'].values.reshape(-1,1)
y= y_train_reg.values.reshape(-1,1)
x_t = x_test_reg['OverallQual'].values.reshape(-1,1)
y_t = y_test_reg.values.reshape(-1,1)

Dt_model_reg = tree.DecisionTreeRegressor(random_state=0, max_leaf_nodes=20)

Dt_model_reg.fit(x, y)

y_pred = Dt_model_reg.predict(X = x_t)

print('OverallQual Regression Tree Coefficients: \n', linear_model.coef_)
print('OverallQual Regression Tree Mean squared error: %.2f' % mean_squared_error(y_test_reg, y_pred))
print('OverallQual Regression Tree R2 score: %.2f' % r2_score(y_test_reg, y_pred))

#Mostrar todas las graficas de regresion
'''fig, axes = plt.subplots(1,len(x_train_reg.columns.values),sharey=True,constrained_layout=True,figsize=(30,15))

e = None
for i,_e in enumerate(x_train_reg.columns):
  e = _e
  linear_model.fit(x_train_reg[e][:,np.newaxis], y_train_reg)
  axes[i].set_title("Best fit line")
  axes[i].set_xlabel(str(e))
  axes[i].set_ylabel('SalePrice')
  axes[i].scatter(x_train_reg[e][:,np.newaxis], y_train_reg,color='g')
  y_pred = linear_model.predict(x_test_reg[e][:,np.newaxis])
  axes[i].plot(x_test_reg[e][:,np.newaxis], y_pred, color='k')

plt.show()'''

# Volver a splittear, entrenar y predecir con las variables seleccionadas

x_reg.pop('YearRemodAdd')
x_reg.pop('YearBuilt')
x_reg.pop('OverallQual')
x_reg.pop('OverallCond')
#x_reg.pop('GrLivArea')
x_reg.pop('FullBath')
x_reg.pop('TotalBsmtSF')
x_reg.pop('1stFlrSF')
x_reg.pop('GarageCars')
#x_reg.pop('GarageArea')
x_reg.pop('TotRmsAbvGrd')
x_reg.pop('Fireplaces')

tic = time.time()
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.3, train_size=0.7, random_state=0)

# use all GrLivArea and GarageArea as predictor
x= x_train_reg.values
y= y_train_reg.values
x_t = x_test_reg.values
y_t = y_test_reg.values

linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred = linear_model.predict(x_t)
tactic = time.time()
print('Linear Regression time: \n', linear_model.coef_)

print('Selected vars Coefficients: \n', linear_model.coef_)
print('Selected vars Mean squared error: %.2f' % mean_squared_error(y_test_reg, y_pred))
print("Selected vars R squared: %.2f"%r2_score(y_test_reg, y_pred))

#Analisis VIF de todas
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(houses_df.values, i)
                          for i in range(houses_df.shape[1])]
vif["features"] = houses_df.columns
print('VIF de variables seleccionadas \n',vif.describe)

#Mapa de correlacion
corr =  houses_copy[['GrLivArea','GarageArea','SalePrice']].corr()
print('Pearson correlation coefficient matrix of each variables:\n', corr)

plt.figure(figsize=(16,10))
#Realizando una mejor visualizacion de la matriz
sns.heatmap(corr,annot=True,cmap='BrBG')
plt.title('Matriz de correlaciones')
plt.tight_layout()
plt.show()

# Overfitting detect

x_pred = linear_model.predict(x)

print('Train Coefficients: \n', linear_model.coef_)
print('Train Mean squared error: %.2f' % mean_squared_error(y_train_reg, x_pred))
print("Train R squared: %.2f"%r2_score(y_train_reg, x_pred))

#RESIDUALES
#Referencia: Informacion de clase

residuales = y_t - y_pred
print('Cantidad de residuales: ',len(residuales))

plt.plot(x_t,residuales, 'o', color='orange')
plt.title("Gráfico de Residuales")
plt.xlabel("Variable independiente")
plt.ylabel("Residuales")

plt.show()
sns.distplot(residuales);
plt.title("Residuales")
plt.show()

data = residuales
plt.hist(data,color='green')
plt.title(f'Histograma')
plt.xlabel(data)
plt.ylabel('Cantidad')
plt.show()
qqplot(data , line='s')
plt.title(f'QQplot para residuales')
plt.show()

plt.boxplot(residuales)
plt.show()

print('Normal Test ',normaltest(residuales))

model = Ridge()
visualizer = ResidualsPlot(model).fit(x,y).score(x_t,y_t)
plt.show()

info_data = sm.OLS(y,x).fit()
print(info_data.summary())

# show predictions
x_test_reg['SalePrice'] = y_pred
print(x_test_reg.head())

# graph y_pred and y_test
plt.scatter(x_test_reg.SalePrice, y_test_reg, color='darkblue')
plt.plot(x_test_reg.SalePrice, y_pred, color='red', linewidth=2)
plt.title('Predicciones')
plt.xlabel('SalePrice')
plt.ylabel('SalePrice')
plt.show()
