#!/usr/bin/env python
# coding: utf-8

# 
# <a href="https://colab.research.google.com/github/hernansalinas/Curso_aprendizaje_estadistico/blob/main/Sesiones/Sesion_01a_pandas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Solución a un problema general de machine learning
# 
# En esta sesión se trabajará con las datos de los precios de las casas de california para dar solución a un problema real
# 
# Referencia: Basado del libro  [hand on machine learning](https://github.com/ageron/handson-ml) y [Kaggle](https://www.kaggle.com/camnugent/california-housing-prices)
# 
# 

# 1. Generar una visión general del problema
# 2. Obtener los datos 
# 3. Explorar los datos
# 4. Descubrir y visualizar los datos
# 5. Preparar los datos para un algoritmo de machinine learning
# 6. Selecccionar los modelos 
# 7. Elegir el mejor modelo tuneando los parametros de ajuste
# 8. Presentar la solucion 
# 9. Moniterar y analizar los datos.
# 
# https://www.kaggle.com/rahulallakonda/california-housing

# 
# # Data.
# 
# 1. Leer los datos
# 2. Analizar el estado de los datos
# 3. Limpiar los datos.
# 4. Analizar los datos.
# 
# 
# ```
# 
# import pandas as pd
# 
# # Read data
# df.read_csv("")
# df.read_excel(")
# 
# # Entender los datos
# df.head()
# df.info()
# df.describe()
# df["name_columns"].unique()
# df.isnull().sum()
# df.isnan().sum()
# df.groupby(by=["name_cols"]).mean()
# df.groupby(by=["name_cols"]).count()
# df.groupby(by=["name_cols"]).sum()
# df.hist()
# 
# ```

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np


# In[2]:


#Si no recuerda los comandos de pandas, puede ver una cheat sheet 
#https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

data_link = "https://raw.githubusercontent.com/hernansalinas/Curso_aprendizaje_estadistico/main/datasets/Sesion_07_housing.csv"
df = pd.read_csv(data_link)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.isna().sum()


# In[8]:


df["ocean_proximity"].unique()


# In[9]:


cols = ["housing_median_age",	"total_rooms",	"total_bedrooms",	"population",	"households",	"median_income",	"median_house_value","ocean_proximity"]
df[cols].groupby(by=["ocean_proximity"]).mean()


# In[10]:


df.ocean_proximity.value_counts().plot(kind="bar")
plt.title('Number of houses')
plt.xlabel("Ocean proximity")
plt.ylabel('Count')
plt.show()


# In[11]:


df.hist(alpha=0.5, figsize=(10,8))


# ### [Diagrama de caja](https://en.wikipedia.org/wiki/Box_plot)
# 
# 
# ### Diagrama de caja 
# 
# ![box](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Boxplot_vs_PDF.svg/800px-Boxplot_vs_PDF.svg.png)
# 
# 
# 
# ### Interpretación de un diagrama de caja
# 
# - Desde el minimo al valor más bajo de la caja: primer cuartil, 25% de los datos 
# - Desde el valor más bajo de la caja hasta la mediana: segundo cuartil, 25% de los datos 
# - Desde la mediana hasta el valor mas alto de la caja : tercer cuartil, 25% de los datos 
# - Desde el valor mas alto de la caja hasta el máximo: Cuarto  cuartil, 25% de los datos 
# 
# 
# El rango intercuartil $IQR = Q_3-Q_1$ permite definir que datos pueden ser atipicos, basado en los siguientes limites:
# 
# $Max = Q3 + 1.5IQR$
# 
# $Min = Q1 - 1.5IQR$
# 
# 
# Veamos un ejemplo:
# 

# In[12]:


T = np.array([52, 57, 57, 58, 63, 66, 66, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 89])
Tsort = np.sort(T)
print(f"T sort:{Tsort}")

IQR=9
max_ = 75 + 1.5*IQR
min_ = 66 - 1.5*IQR
print(max_)
print(min_)
plt.boxplot(T)


#  volviendo a nuestro datos tenemos que:

# In[13]:


df2 = df[df["ocean_proximity"] == "<1H OCEAN"]
df2["median_house_value"].hist()


# In[14]:


#draw boxplot
df.boxplot(column="median_house_value", by='ocean_proximity', sym = 'k.', figsize=(18,6))
#set title
plt.title('Boxplot for Camparing price per living space for each city')
plt.show()


# ## [Matrix de correlación](https://en.wikipedia.org/wiki/Correlation)
# 
# ¿Como se determina la matrix de correlación?
# 
# ![Matrix de correlación](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Correlation_examples2.svg/1920px-Correlation_examples2.svg.png)
# 
# 
# 
# ```
# corr_matrix = df.corr()
# corr_matrix
# 
# plt.figure(figsize = (10,6))
# sns.heatmap(corr_matrix, annot = True)
# plt.show()
# ```
# 

# In[15]:


corr_matrix = df.corr()
corr_matrix


# In[16]:


# Visualización de la matrix de correlación
plt.figure(figsize = (10,6))
sns.heatmap(corr_matrix, annot = True)
plt.show()


# In[17]:


cols = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
#pd.plotting.scatter_matrix(df[cols], alpha = 0.2, figsize = (12,8))
g = sns.pairplot(df[cols], diag_kind="kde")


# In[18]:


ax = sns.scatterplot(df.median_income, df.median_house_value, alpha = 0.5)


# In[19]:


#sns.set()
#tips = sns.load_dataset("tips")

ax = sns.scatterplot(df.longitude,df.latitude, c=df.median_house_value)
sm = plt.cm.ScalarMappable()
ax.figure.colorbar(sm)


# In[20]:


# cols=df.columns
# cols
# fig, axs = plt.subplots(3, 3, figsize=(10,10))
# fig.add_gridspec(3, 3, hspace=10, wspace=40)
# k = 0
# for i in range(0,3):
#     for j in range(0,3):        
#         sns.histplot(df[cols[k]],ax=axs[i, j])
#         #sns.kdeplot(df[cols[k]], ax=axs[i, j],color="b")        
#         axs[i, j].legend(fontsize = 8)        
        
#         if(j==0):
#           axs[i, j].set_ylabel("")
#         if(i==2):
#           axs[i, j].set_xlabel("")
#         k=k+1


# In[21]:


df


# In[22]:


# Combinar caracteristics puede ser una buena forma de definir nuevas variables y mejorar
# los entrenamientos del algoritmo.


# # Preparación de data para un algoritmo de machine learning 

# # Evitar el data *Snooping bias*.
# 
# En algunos casos se sugiere dividir los datos en entrenamiento y test desde el principio dado que el cerebro puede sobreajustar el dataset y los resultados no significativos se pueden volver significativos. El procedimiento correcto es probar cualquier hipótesis en un conjunto de datos que no se utilizó para generar la hipótesis. 
# 
# 
# # *Sampling bias*
# 
# Si el dataset es lo suficientemente grande un muestreo aleatorio de la muestra puede ser considerado, sin embargo si la muestra es pequena se debe garantizar homegeniedad en el dataset de entrenamiento. 
# 
# 
# Ejemplo: 
# 
# Por ejemplo, la población de EE. UU. esta compuesto por un 51,3 % de mujeres y un 48,7 % de hombres, por lo que una encuesta bien realizada en EEUU
# trata de mantener esta proporción en la muestra: 513 mujeres y 487 hombres. Esto se llama muestreo estratificado(stratified sampling): la población se divide en subgrupos homogéneos llamados estratos(strata), y se muestrea el número correcto de instancias de cada estrato para garantizar que el
# El conjunto de prueba es representativo de la población general. Si usaran muestras puramente aleatorias, habría alrededor del 12% de posibilidades de muestrear un conjunto de prueba sesgado con menos del 49% de mujeres o más del 54% de mujeres. De cualquier manera, los resultados de la encuesta serían
# significativamente sesgada.
# 
# 

# In[23]:


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split


# In[24]:


# ¿Es significativa la muestra que se esta considerando?
train_set, test_set   = train_test_split(df, test_size = 0.2, random_state = 42)


# In[25]:


print(len(train_set))
print(len(test_set))


# In[26]:


df


# ### División del dataset en grupos
# La siguiente division debe ser basada en la experticie de lo que se esta realizando, y sobre ello se debe tomar una muestra significativa

# In[27]:


df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[28]:


df.income_cat.hist()


# In[29]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["income_cat"]):
  strat_train_set = df.loc[train_index]
  strat_test_set = df.loc[test_index]


# ### Proporciones del dataset

# In[30]:


df["income_cat"].value_counts() / len(df)


# In[31]:


strat_train_set["income_cat"].value_counts() / len(strat_train_set)


# In[32]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[33]:


train_set, test_set   = train_test_split(df, test_size = 0.2, random_state = 42)


# In[34]:


test_set["income_cat"].value_counts() / len(test_set)


# In[35]:


test_set["income_cat"].value_counts() / len(test_set)


# In[36]:


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(df),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] =abs( 100 * compare_props["Random"] / compare_props["Overall"] - 100)
compare_props["Strat. %error"] =abs( 100 * compare_props["Stratified"] / compare_props["Overall"] - 100)


# In[37]:


compare_props


# En Conclusion

# In[38]:


#Si no recuerda los comandos de pandas, puede ver una cheat sheet 
#https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
data_link = "https://raw.githubusercontent.com/hernansalinas/Curso_aprendizaje_estadistico/main/datasets/Sesion_07_housing.csv"
df = pd.read_csv(data_link)

#1. Leer los data. 
#2. Para hacer el split analizar si la muestra es significativa para el entrenamiento y test
#3. Dejar los data de test ocultos para hacer las pruebas


# In[39]:


df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[40]:


split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["income_cat"]):
  strat_train_set = df.loc[train_index]
  strat_test_set = df.loc[test_index]

df_train = strat_train_set
df_test = strat_test_set


# ## Matrix de correlación

# In[41]:


corr_matrix = df_train.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[42]:


# Visualización de la matrix de correlación
plt.figure(figsize = (10,6))
sns.heatmap(corr_matrix, annot = True)
plt.show()


# ## Agregar nuevas variables

# In[43]:


df_train["rooms_per_household"] = df_train["total_rooms"]/df_train["households"]
df_train["bedrooms_per_room"] = df_train["total_bedrooms"]/df_train["total_rooms"]
df_train["population_per_household"]=df_train["population"]/df_train["households"]


# In[44]:


corr_matrix = df_train.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[45]:


# Visualización de la matrix de correlación
plt.figure(figsize = (10,6))
sns.heatmap(corr_matrix, annot = True)
plt.show()


# # Limpieza de datos

# In[46]:


df_train.isnull().sum()


# In[47]:


#df_train.dropna(subset=["total_bedrooms"]) #Eliminar los nan
#df_train.drop("total_bedrooms", axis=1)  # Eliminar la columna
median = df_train["total_bedrooms"].median()
q=df_train["total_bedrooms"].fillna(median).copy()


# In[48]:


q=pd.DataFrame(q)


# In[49]:


q.isnull().sum()


# ## Imputer

# In[50]:


df_train_num = df_train.drop("ocean_proximity", axis=1)


# In[51]:


from sklearn.impute import SimpleImputer
#imputer = Imputer(strategy="median")
imp_mean = SimpleImputer( strategy='mean')


# In[52]:


imp_mean.fit(df_train_num)


# In[53]:


imp_mean.statistics_


# In[54]:


df_train_num.median().values


# In[55]:


X = imp_mean.transform(df_train_num)


# In[56]:


X


# In[57]:


housing_tr = pd.DataFrame(X, columns=df_train_num.columns)


# In[58]:


housing_tr


# # Manejo de texto y atributos categoricos

# In[59]:


#from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# In[60]:


df_train["ocean_proximity"].unique()


# In[61]:


housing_cat=df_train[["ocean_proximity"]]
housing_cat


# In[62]:


cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(cat_encoder.categories_)


# # Resumen de las transformaciones paso a paso
# 
# 
# 

# In[63]:


# Read Data
data_link = "https://raw.githubusercontent.com/hernansalinas/Curso_aprendizaje_estadistico/main/datasets/Sesion_07_housing.csv"
df = pd.read_csv(data_link)


# In[64]:


# 1. Clasification 
df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

df_ = df.copy()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df_, df_["income_cat"]):
 strat_train_set = df_.loc[train_index]
 strat_test_set = df_.loc[test_index]


# In[65]:


for set in (strat_train_set, strat_test_set):
 set.drop(["income_cat"], axis=1, inplace=True)


# In[66]:


strat_train_set


# In[67]:


# 2 Add New variables to data
housing = strat_train_set.copy()
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[68]:


housing


# In[69]:


housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)


# In[70]:


housing.isnull().sum()


# In[71]:


housing_num = housing.drop("ocean_proximity", axis=1)
housing_num


# In[72]:


# Imputer
imp_mean = SimpleImputer( strategy='mean')
imp_mean.fit(housing_num)
imp_mean.statistics_


# In[73]:


X = imp_mean.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns)


# In[74]:


housing_tr.isnull().sum()


# In[75]:


# 3. One hot encoder
housing_cat = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(cat_encoder.categories_)


# In[76]:


housing_cat_1hot


# In[77]:


cat_encoder.categories_[0]


# In[78]:


df_cat_1hot = pd.DataFrame(housing_cat_1hot, columns = cat_encoder.categories_[0])


# In[79]:


df_cat_1hot


# In[80]:


housing_tr_ = housing_tr.join(df_cat_1hot)


# ## Escalamiento de variables

# In[81]:


cols=["longitude", "latitude",	"housing_median_age",	"total_rooms",      "total_bedrooms",	"population",	"households",	"median_income",      "<1H OCEAN",	"INLAND",	"ISLAND",	"NEAR BAY", "NEAR OCEAN"]


# In[82]:


housing_scale=housing_tr_[cols]


# In[83]:


housing_scale


# In[84]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(housing_scale)


# In[85]:


X = scaler.transform(housing_scale)
X


# In[86]:


housing_prepared = pd.DataFrame(X, columns = housing_scale.columns)


# In[87]:


len(housing_prepared)


# In[88]:


len(housing_labels)


# In[89]:


# Consideremos las siguientes columnas: 
cols=["longitude", "latitude",	"housing_median_age",	"total_rooms",      "total_bedrooms",	"population",	"households",	"median_income"]
housing_prep = housing_prepared[cols]


# In[90]:


housing_prep


# In[91]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(housing_prep, housing_labels)
housing_predictions = model.predict(housing_prep)


# In[92]:


model.score(housing_prep, housing_labels)


# In[93]:


#¿Como autmatizar todo el proceso?
#¿El modelo de regresion lineal es valido para lo construido, 
#que informacion nos da el score?
#¿Se puede mejorar agregando nuevas caracteristicas?
# ¿Puede ser ajustado a otro modelo?

