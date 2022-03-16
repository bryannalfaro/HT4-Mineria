#Universidad del Valle de Guatemala
#Mineria de Datos
#HT4 Regresion
#Integrantes
#Bryann Alfaro
#Diego de Jesus
#Julio Herrera


from cgi import test
from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import Counter
from sklearn import preprocessing, tree
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pyclustertend
import random
import graphviz
import sklearn.mixture as mixture
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from scipy.stats import normaltest
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot
import statsmodels.api as sm

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
houses_copy = (houses_df.copy())
y_reg = houses_df.pop('SalePrice')
x_reg = houses_df

print(y_reg.shape, x_reg.shape)

x_reg.pop('MasVnrArea')
x_reg.pop('GarageYrBlt')
x_reg.pop('YearRemodAdd')
#x_reg.pop('YearBuilt')
x_reg.pop('OverallQual')
x_reg.pop('OverallCond')
#x_reg.pop('GrLivArea')
x_reg.pop('FullBath')
x_reg.pop('TotalBsmtSF')
x_reg.pop('1stFlrSF')
x_reg.pop('GarageCars')
x_reg.pop('GarageArea')
x_reg.pop('TotRmsAbvGrd')
x_reg.pop('Fireplaces')
random.seed(5236)

x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.3, train_size=0.7, random_state=0)
print(x_train_reg.shape,x_test_reg.shape,y_train_reg.shape,y_test_reg.shape)



# use all GrLivArea and YearBuilt as predictor
x= x_train_reg.values
y= y_train_reg.values
x_t = x_test_reg.values
y_t = y_test_reg.values

linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred = linear_model.predict(x_t)


vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(houses_df.values, i)
                          for i in range(houses_df.shape[1])]
vif["features"] = houses_df.columns
print(vif.describe)

corr =  houses_copy[['GrLivArea','YearBuilt','SalePrice']].corr()
print('Pearson correlation coefficient matrix of each variables:\n', corr)

mask = np.zeros_like(corr, dtype=np.bool)
np.fill_diagonal(mask, val=True)

fig, ax = plt.subplots(figsize=(4, 3))

cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)
cmap.set_bad('grey')

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
fig.suptitle('Pearson correlation coefficient matrix', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=10)


print('Coefficients: \n', linear_model.coef_)
print('Mean squared error: %.2f' % mean_squared_error(y_test_reg, y_pred))
print("R squared: %.2f"%r2_score(y_test_reg, y_pred))


# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_t[:,0], x_t[:,1], y_t, c='r', marker='o')
# graph a plane using prediction
x_surf = np.linspace(x_t[:,0].min(), x_t[:,0].max(), 100)
y_surf = np.linspace(x_t[:,1].min(), x_t[:,1].max(), 100)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = linear_model.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.2, color='b')
ax.set_xlabel('GrLivArea')
ax.set_ylabel('Year Built')
ax.set_zlabel('SalePrice')
plt.show()

#RESIDUALES

residuales = y_t - y_pred
len(residuales)



plt.plot(x_t,residuales, 'o', color='darkblue')
plt.title("Gráfico de Residuales")
plt.xlabel("Variable independiente")
plt.ylabel("Residuales")

plt.show()
sns.distplot(residuales);
plt.title("Residuales")
plt.show()
plt.boxplot(residuales)
plt.show()

print('Normal Test ',normaltest(residuales))

model = Ridge()
visualizer = ResidualsPlot(model)
visualizer.fit(x,y)
visualizer.score(x_t,y_t)

plt.show()

est = sm.OLS(y,x)
est2 = est.fit()
print(est2.summary())









'''# use OverallQual as predictor
x= x_train_reg['OverallQual'].values.reshape(-1,1)
y= y_train_reg.values.reshape(-1,1)
x_t = x_test_reg['OverallQual'].values.reshape(-1,1)
y_t = y_test_reg.values.reshape(-1,1)

linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred = linear_model.predict(x_t)

print('OverallQual Coefficients: \n', linear_model.coef_)
print('OverallQual Mean squared error: %.2f' % mean_squared_error(y_test_reg, y_pred))
print('OverallQual R2 score: %.2f' % r2_score(y_test_reg, y_pred))'''


'''# use OverallQual and GrLivArea as predictors
x= x_train_reg[['OverallQual', 'GrLivArea']].values
y= y_train_reg.values.reshape(-1,1)
x_t = x_test_reg[['OverallQual', 'GrLivArea']].values
y_t = y_test_reg.values.reshape(-1,1)

linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred = linear_model.predict(x_t)

print('Multicolinealidad. Coefficients: \n', linear_model.coef_)
print('Multicolinealidad. Mean squared error: %.2f' % mean_squared_error(y_test_reg, y_pred))
print('Multicolinealidad. R2 score: %.2f' % r2_score(y_test_reg, y_pred))

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_t[:,0], x_t[:,1], y_t, c='r', marker='o')
# graph a plane using prediction
x_surf = np.linspace(x_t[:,0].min(), x_t[:,0].max(), 100)
y_surf = np.linspace(x_t[:,1].min(), x_t[:,1].max(), 100)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = linear_model.predict(np.c_[x_surf.ravel(), y_surf.ravel()]).reshape(x_surf.shape)
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.2, color='b')
ax.set_xlabel('OverallQual')
ax.set_ylabel('GrLivArea')
ax.set_zlabel('SalePrice')
plt.show()'''

