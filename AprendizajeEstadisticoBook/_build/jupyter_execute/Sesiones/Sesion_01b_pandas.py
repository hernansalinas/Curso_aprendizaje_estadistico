#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hernansalinas/Curso_aprendizaje_estadistico/blob/main/Sesiones/Sesion_01b_pandas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# 
# 
# 

# # Construyendo Data Frame desde diccionarios  y cargando datos del un data frame
# 

# In[1]:


import numpy as np
import pandas as pd
import os


# In[2]:


x = np.linspace(0, 10, 10)
y = np.linspace(0, 10, 10)

d = {"x": x, "y": y}


# In[3]:


df = pd.DataFrame(d)


# In[4]:


df.y


# In[5]:


path = os.getcwd()
# https://www.kaggle.com/gpreda/covid-world-vaccination-progress?select=country_vaccinations
path="https://github.com/hernansalinas/Curso_aprendizaje_estadistico/blob/main/datasets/sesion_01b_country_vaccinations.xlsx?raw=true"
df = pd.read_excel(f"{path}") 


# In[ ]:


df.head()


# In[ ]:


# https://www.kaggle.com/sakhawat18/asteroid-dataset 
# https://ssd.jpl.nasa.gov/tools/sbdb_query.html

path_git = "https://raw.githubusercontent.com/hernansalinas/Curso_aprendizaje_estadistico/main/datasets/sesion_01b_dataset.csv"
df = pd.read_csv(f"{path_git}")
df


# Lectura de un dataset ubicado en el drive

# In[ ]:


url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSHCOR8_Ha6TvBQwIcpjvJ0bzHYel1S8DXl4NHnMhVvdbibrgL_SP6rffuESpaJvPwLuUizXblQtHox/pub?output=csv"
df = pd.read_csv(url)
df


# Asignacion a la columna index  la columna date

# In[ ]:


df = pd.read_csv(url, index_col="date")


# In[ ]:


df


# Renombrar columnas

# In[ ]:


df1 = df.rename(columns={"location": "Location", 
                         "vaccine":" Vaccine ", 
                         "total_vaccinations":"Total Vaccinations"} ).copy()  #inplace=True
df1
#Note el espacio en Vaccine


# Convirtiendo a minusculas todas las columnas

# In[ ]:


new_df = df1.rename(mapper = str.lower, axis="columns")
new_df


# In[ ]:


new_df.columns


# Elimnando el espacio inicial de todas las columnas

# In[ ]:


new_df = new_df.rename(mapper = str.strip, axis="columns")
new_df


# inicializando el indice a valores enteros

# In[ ]:


new_df = new_df.reset_index()
new_df


# Otra forma de generar el cambio

# In[ ]:


df1.columns


# In[ ]:


cols = [c.lower().strip() for c in df1.columns]


# In[ ]:


df1.columns = cols


# In[ ]:


df1.columns


# ![img](https://github.com/hernansalinas/Curso_aprendizaje_estadistico/blob/main/Sesiones/imagenes/codeCase.png?raw=true "CodeCase")

# # Pascal Case notation

# In[ ]:


#https://www.kaggle.com/saliblue/country-vaccinations-by-manufacturer
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSHCOR8_Ha6TvBQwIcpjvJ0bzHYel1S8DXl4NHnMhVvdbibrgL_SP6rffuESpaJvPwLuUizXblQtHox/pub?output=csv"
df = pd.read_csv(url)
df


# In[ ]:


#df.columns
a = "hello world "
col = [c.capitalize() for c in a.split()]
col


# In[ ]:


df.columns


# In[ ]:


df.columns = [c.replace("_"," ") for c in df.columns]


# In[ ]:


df.columns


# Paso a paso para una expresion más compacta, ejemplo de PascaCase

# In[ ]:


a = [ cols  for cols in df.columns ]
a


# In[ ]:


a = [ [c  for c in cols.split()]   for cols in df.columns ]
a


# In[ ]:


a = [ [c.capitalize()  for c in cols.split()]   for cols in df.columns ]
a


# In[ ]:


a = ["adfads","Bsdfadf"]
" ".join(a)


# In[ ]:


a =[ "".join([c.capitalize() for c in cols.split()])  for cols in df.columns ]
a


# In[ ]:


cols=a


# In[ ]:


df.columns=cols
df


# # Mascaras en columnas

# In[ ]:


tf = df["TotalVaccinations"] > 2157500


# In[ ]:


df[tf] #Mascara, nuevo data frame con un numero diferentes de lineas


# In[ ]:


df.where(tf)  #asigna NAN a todo el data frame donde no se cumple la condicion establecida


# In[ ]:


n_df = df.where(tf).copy()
n_df.dropna()  #Volvemos a obtener el data frame generado con la mascara y #filtrado como un array


# Volviendo a la mascara

# In[ ]:


df = df[tf]


# In[ ]:


df


# Comparacion para valores de una misma columna

# In[ ]:


df[ (df["TotalVaccinations"]>2273457) & (df["TotalVaccinations"]<61206560 ) ]


# In[ ]:


df.reset_index()


# In[ ]:


q=df.set_index("Location")
q.reset_index()


# unique()

# In[ ]:


df.Location.unique()


# In[ ]:


df.Vaccine.unique()


# Definir columnas

# In[ ]:


cols=['Location', 'Date', 'Vaccine']
df[cols]


# In[ ]:


g=df.set_index(['Location','Vaccine'])
g


# In[ ]:


g.loc["Austria"]


# Operacion groupby 

# In[ ]:


df.groupby(["Location"])


# In[ ]:


df.groupby(["Location"]).count()


# In[ ]:


df.groupby(["Vaccine"]).count()


# In[ ]:


part_df = df[df.Location=="Austria"].reset_index()


# In[ ]:


mask = df.Location.isnull()


# In[ ]:


df.fillna(0)


# In[ ]:





# 

# 

# Series de tiempo 
# https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv

# Algunos repositorios y paginas de interes
# 
# 
# 1. https://www.nature.com/sdata/policies/repositories
# 
# 2. https://paperswithcode.com/
# 
# 3. https://towardsdatascience.com/31-datasets-for-your-next-data-science-project-6ef9a6f8cac6
# 
# 4. https://www.data.gov/
# 
# 5. https://archive.ics.uci.edu/ml/index.php
# 
# 6. https://data.world/datasets/geodata
# 
# 7. https://matmatch.com/advanced-search?categories=ceramic
# 
# 8. https://github.com/sedaoturak/data-resources-for-materials-science
# 
# 
# 9. https://guides.library.cmu.edu/machine-learning/datasets

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




