#!/usr/bin/env python
# coding: utf-8

# 
# <a href="https://colab.research.google.com/github/hernansalinas/Curso_aprendizaje_estadistico/blob/main/Sesiones/Sesion_01a_pandas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Regresion multivariada
# 
# Supongamos que tenemos un conjunto de caracteristicas $X = X_1,X_2...X_j...X_n$ para realizar una  predicción $y$ con valores esperados $\hat{y}$.  
# 
# Cada X, puede ser escrito como:
#  $X_1 = x_1^{(1)},x_1^{(2)}, x_1^{(3)}...x_1^{(m)}$, 
# 
#  $X_2 = x_2^{(1)},x_2^{(2)}, x_2^{(3)}...x_2^{(m)}$, 
#  
#  .
#  
#  .
#  
#  .
#  
#  $X_n = x_n^{(1)},x_n^{(2)}, x_n^{(3)}...x_n^{(m)}$. 
#  
# 
# Siendo n el número de caracteristicas y m el número de datos de datos, 
# $\hat{y} = \hat{y}_1^{(1)}, \hat{y}_1^{(2)}...\hat{y}_1^{(m)} $, el conjunto de datos etiquetados  y $y = y_1^{(1)}, y_1^{(2)}...y_1^{(m)} $ los valores predichos por un modelo
# 
# 
# 
# 
# Lo anterior puede ser resumido  como:
# 
# 
# 
# |Training|$\hat{y}$      | X_1  | X_2  |  .  | .|. |. | X_n|
# |--------|-------|------|------|-----|--|--|--|----|
# |1|$\hat{y}_1^{1}$ | $x_1^{1}$|$x_2^{1}$| .  | .|. |. | $x_n^{1}$|
# |2|$\hat{y}_1^{2}$ | $x_1^{2}$|$x_2^{2}$| .  | .|. |. | $x_n^{2}$|
# |.|.         | .        |.| .  | .|. |. | |
# |.|.         | .        |.| .  | .|. |. | |
# |.|.         | .        |.| .  | .|. |. | |
# |m|$\hat{y}_1^{m}$ | $x_1^{m}$  |$x_2^{m}$| .  | .|. |. | $x_n^{m}$|
# 
# 
# y el el modelo puede ser ajustado como sigue: 
# 
# Para un solo conjunto de datos de entrenamiento tentemos que:
# 
# $y = h(\theta_0,\theta_1,\theta_2,...,\theta_n ) = \theta_0 + \theta_1 x_1+\theta_2 x_2 + \theta_3 x_3 +...+ \theta_n x_n $.
# 
# \begin{equation}
# h_{\Theta}(x) = [\theta_0,\theta_1,...,\theta_n ]\begin{bmatrix}
# 1^{(1)}\\
# x_1^{(1)}\\
# x_2^{(1)}\\
# .\\
# .\\
# .\\
# x_n^{(1)}\\
# \end{bmatrix} = \Theta^T X^{(1)}
# \end{equation}
# 
# 
# 
# Para todo el conjunto de datos, tenemos que:
# 
# Sea $\Theta^T = [\theta_0,\theta_1,\theta_2,...,\theta_n]$ una matrix $1 \times (n+1)$ y  
# 
# 
# \begin{equation}
# X =
# \begin{bmatrix}
# 1& 1 & 1 & .&.&.&1\\
# x_1^{(1)}&x_1^{(2)} & x_1^{(3)} & .&.&.&x_1^{(m)}\\
# .&. & . &.&.&.& .\\
# .&. & . & .&.&.&.\\
# .&. & . & .&.&.&.\\
# x_n^{(1)}&x_n^{(2)} & x^{(3)} & .&.&.&x_n^{(m)}\\
# \end{bmatrix}_{(n+1) \times m}
# \end{equation}
# 
# 
# 
# 
# luego $h = \Theta^{T} X $ con dimension $1\times m$
# 
# 
# 
# 
# La anterior ecuación, es un hiperplano en $\mathbb{R}^n$. Notese que en caso de tener una sola característica, la ecuación puede ser análizada según lo visto en la sesión de regresion lineal.
# 
# 
# Para la optimización, vamos a definir la función de coste **$J(\theta_1,\theta_2,\theta_3, ...,\theta_n )$** , como la función  asociada a la minima distancia entre dos puntos, según la metrica euclidiana. 
# 
# - Metrica Eculidiana
# 
# \begin{equation}
# J(\theta_1,\theta_2,\theta_3, ...,\theta_n )=\frac{1}{2m} \sum_{i=1}^m ( h_{\Theta} (X)-\hat{y}^{(i)})^2 =\frac{1}{2m} \sum_{i = 1}^m (\Theta^{T} X - \hat{y}^{(i)})^2
# \end{equation}
# 
# Otras métricas pueden ser definidas como sigue en la siguiente referencia.  [Metricas](https://jmlb.github.io/flashcards/2018/04/21/list_cost_functions_fo_neuralnets/).
# 
# Nuestro objetivo será encontrar los valores mínimos 
# $\Theta = \theta_0,\theta_1,\theta_2,...,\theta_n$ que minimizan el error, respecto a los valores etiquetados y esperados $\hat{y}$ 
# 
# 
# Para encontrar $\Theta$ opmitimo, se necesita  minimizar la función de coste, que permite obtener los valores más cercanos,  esta minimización podrá ser realizada a través de diferentes metodos, el más conocido es el gradiente descendente.
# 
# 
# 
# 
# 
# 

# 
# ## Gradiente descendente
# 
# Consideremos la función de coste sin realizar el promedio esima de funcion de coste:
# \begin{equation}
# \Lambda =
# \begin{bmatrix}
# (\theta_0 1 + \theta_1 x_1^1+\theta_2 x_2^2 + \theta_3 x_3^3 +...+ \theta_n x_n^n - \hat{y}^{1})^2 \\
# (\theta_0 1+ \theta_1 x_1^1+\theta_2 x_2^2 + \theta_3 x_3^3 +...+ \theta_n x_n^n - \hat{y}^{2})^2\\
# .\\
# .\\
# .\\
# (\theta_0 1 + \theta_1 x_1^m+\theta_2 x_2^m + \theta_3 x_3^m +...+ \theta_n x_n^m - \hat{y}^{m})^2\\
# \end{bmatrix}
# \end{equation}
# 
# $\Lambda= [\Lambda_1,\Lambda_2, ...,\Lambda_m]$
# 
# $J = \frac{1}{2m} \sum_{i}^m \Lambda_i $
# 
# El gradiente descente, puede ser escrito como:
# 
# \begin{equation}
# \Delta \vec{\Theta} =  - \alpha \nabla J(\theta_0, \theta_1,...,\theta_n)
# \end{equation}
# 
# escogiendo el valor j-esimo tenemos que:
# 
# \begin{equation}
# \theta_j :=  - \alpha \frac{\partial J(\theta_0, \theta_1,...\theta_j...,\theta_n)}{\partial \theta_j}
# \end{equation}
# 
# Aplicando lo anterior a a funcion de coste asociada a la metrica ecuclidiana, tenemos que:
# 
# Para $j = 0$, 
# 
# 
# \begin{equation}
# \theta_0 :=  - \alpha \frac{\partial J(\theta_0, \theta_1,...\theta_j...,\theta_n)}{\partial \theta_0} = \frac{1}{m}\alpha \sum_{i=1}^m (\theta_j X_{ji} - \hat{y}^{(i)}) 1
# \end{equation}
# 
# 
# 
# Para $0<j<n $
# 
# \begin{equation}
# \theta_j :=  - \alpha \frac{\partial J(\theta_0, \theta_1,...\theta_j...,\theta_n)}{\partial \theta_j} = \frac{1}{m} \alpha\sum_{i=1}^m (\theta_{j} X_{ji} - \hat{y}^{(i)}) X_j
# \end{equation}
# 
# donde X_j es el vector de entrenamiento j-esimo.
# 
# Lo  anterior puede ser generalizado como siguem, teniendo presente que $X_0 = \vec{1}$
# 
# 
# Para $0\leq j<n$, 
# 
# \begin{equation}
# \theta_j :=  - \alpha \frac{\partial J(\theta_0, \theta_1,...\theta_j...,\theta_n)}{\partial \theta_j} = \frac{1}{m} \alpha\sum_{i=1}^m (\theta_j X_{ji} - \hat{y}^{(i)}) X_j 
# \end{equation}
# 
# 
# 

# # Modelos polinomiales
# 
# Otros modelos pueden ser ajustado. Consideremos una sola caracteristica,
# $y=h(\theta_0,\theta_1,\theta_2,...,\theta_n ) = \theta_0 + \theta_1 X_1$
# En este caso el exponente de la variable X habla de la complejidad del modelo, para un solo dato entrenamiento podemos lo siguiente:
# 
# - Lineal
# 
#   $h_{\theta} = \theta_0 + \theta_1 x_1$
# 
# - Orden 2
# 
#   $h_{\theta} = \theta_0 + \theta_1 x_1 + \theta_2 x_1 ^2$
# 
#   
# - Orden 3
# 
#   $h_{\theta} = \theta_0 + \theta_1 x_1 + \theta_2 x_1 ^2 +  \theta_2 x_1 ^3$
# 
# 
# 

# In[1]:


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

#boston_dataset = load_boston()


# In[2]:


#data_url = "http://lib.stat.cmu.edu/datasets/boston"
#raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#target = raw_df.values[1::2, 2]


# In[3]:


N=10
x1 = np.linspace(-1, 1, N) 
x2 = np.linspace(-2, 2, N)
y = 2*x1 - 3*x2 + 0.0 #+ 4*np.random.random(100) 


# In[4]:


N = 10
X1,X2 = np.meshgrid(x1,x2)
Y = 2*X1 - 3*X2 + 0.0
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X1, X2, Y)
#scatter = ax.scatter(x1, x2, y,"-")
                       


# In[5]:


# Construccion del modelo para cualquier valor de Theta y X
# Generalizacion
def linearRegresion(Theta_, _X_, m_training, n_features) : 
  m = m_training
  n = n_features
  shape_t = (n+1,1)
  shape_x = (n+1,m)

  if(shape_x != np.shape(_X_)):
    print(f"Revisar valores dimensiones Theta_ {_X_}")
    return 0  
 
  if(shape_t != np.shape(Theta_)):
    print(f"Revisar valores dimensiones Theta_ {Theta_}")
    return 0
  else :
    return (Theta_.T@_X_)

def cost_function(h, y):
  J = (h-y)**2
  
  return J.mean()/2

def gradiente_D(h,Theta_, _X_, y,alpha, m_training, n_features):  
  grad=((h-y)*_X_.T).mean(axis=1)
  theta=Theta_.T-alpha*grad
  return theta


# In[6]:


N = 10
m_training=10
x1 = np.linspace(-1, 1, N) 
x2 = np.linspace(-1, 1, N)
np.random.seed(30)
y = 2*x1 - 3*x2  + 4*np.random.random(N) 

df = pd.DataFrame({"Y":y,"X1":x1, "X2":x2})
df["ones"] = np.ones(N)
X = df[["ones","X1","X2"]].values
#y = np.reshape(df.Y.values, (1,N))

#_X_ = np.matrix(X.T)
Theta = np.array([2.0, 4, 5])
Theta_ = np.reshape(Theta, (3, 1))


# In[7]:


N = 10
X1,X2 = np.meshgrid(x1,x2)
Y = 2*X1 - 3*X2  + 4*np.random.random(N) 
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X1, X2, Y)
#scatter = ax.scatter(x1, x2, y,"-")


# In[8]:


cost=[]
t_=[]
for i in range(0, 1000):
  h = linearRegresion(Theta_, X.T, N, 2)
  J = cost_function(h,y)  
  theta = gradD(h,Theta_, X,y, 0.1, N,2)  
  Theta_ = theta.T
  t_.append(theta[:1])
  cost.append(J)


# In[ ]:


plt.plot(cost)
plt.ylabel("J")
plt.xlabel("iter")


# In[ ]:


Theta_=np.array(Theta_)
ymodelo = Theta_[0] + Theta_[1]*x1 + Theta_[2]*x2
ymodelo


# In[ ]:


error = abs((y-ymodelo))/y
plt.plot(error)


# ¿Qué sucede si las caracteristicas no estan escaladas?
# 
# Es recomendable escalar en función de :
# 
# \begin{equation}
# x_i = \frac{x_i-\mu_i}{s_i}
# \end{equation}
# 
# 
# 
# $\mu_i$:  es el promedio de los datos de la caracteristica i
# 
# $s_i$:  es el rango de valores maximos o minimos o la desviacion estandar.
# 
# Si el rango de valores de la caracteristica xi esta entre  [200,3000] y la media de valores es 2000, tenemos que 
# 
# \begin{equation}
# x_i = \frac{x_i-2000}{2800}
# \end{equation}
# 
# 
# 
# 
# 
# 

# In[ ]:


N = 10
m_training=10
x1 = np.linspace(-1, 1, N) 
x2 = np.linspace(-2000, 2000, N)
np.random.seed(30)
y = 2*x1 - 3*x2  + 4*np.random.random(N) 

df = pd.DataFrame({"Y":y,"X1":x1, "X2":x2})
df["ones"] = np.ones(N)
X = df[["ones","X1","X2"]].values
#y = np.reshape(df.Y.values, (1,N))

#_X_ = np.matrix(X.T)
Theta = np.array([2.0, 4, 5])
Theta_ = np.reshape(Theta, (3, 1))


# In[ ]:


cost=[]
t_=[]
for i in range(0, 1000):
  h = linearRegresion(Theta_, X.T, N, 2)
  J = cost_function(h,y)  
  theta = gradD(h,Theta_, X,y, 0.1, N,2)  
  Theta_ = theta.T
  t_.append(theta[:1])
  cost.append(J)
# NO hay convergencia 


# In[ ]:


plt.plot(cost)


# In[ ]:


N = 10
m_training=10
x1 = np.linspace(-1, 1, N) 
x2 = np.linspace(-2000, 2000, N)
x2 =  (x2-np.mean(x2))/(max(x2)-min(x2))

np.random.seed(30)
y = 2*x1 - 3*x2  + 4*np.random.random(N) 

df = pd.DataFrame({"Y":y,"X1":x1, "X2":x2})
df["ones"] = np.ones(N)
X = df[["ones","X1","X2"]].values
#y = np.reshape(df.Y.values, (1,N))

#_X_ = np.matrix(X.T)
Theta = np.array([2.0, 4, 5])
Theta_ = np.reshape(Theta, (3, 1))


# In[ ]:


cost=[]
t_=[]
for i in range(0, 1000):
  h = linearRegresion(Theta_, X.T, N, 2)
  J = cost_function(h,y)  
  theta = gradD(h,Theta_, X,y, 0.1, N,2)  
  Theta_ = theta.T
  t_.append(theta[:1])
  cost.append(J)

plt.plot(cost)

