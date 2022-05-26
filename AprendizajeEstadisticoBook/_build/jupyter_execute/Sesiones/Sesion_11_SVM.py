#!/usr/bin/env python
# coding: utf-8

# # Maquinas de soporte vectorial, (*Support vector machine SVM*)
# 
# 
# 
# 
# 
# Las maquinas de soporte vectorial se pueden definir como:
# 
# - Clasfiicar lineal de orden maximo 
# - Clasificador lineal en un espacio N-dimensional
# 
# Entendamos la intuición detras de estos dos conceptos. 
# 
# Nuestra funcion de coste esta definda como sigue:
# 
# \begin{equation}
# J(\theta)=\frac{1}{m}\sum_{i=1}^{m} [-y^{(i)}\log(h^{(i)}) + (1-y^{(i)})\log(1-h^{(i)})]  +  \frac{\lambda}{2m} \sum_{j=1}^{n}\theta_j^2
# \end{equation}
# 
# 
# Si suponemos que $y=1$ y $y=0$ tenemos para estos dos casos que:
# 
# 
# 
# 
# 

# In[1]:


#Cost function
h = lambda z: 1/(1+np.exp(-z))
J1 = lambda z,y: -y*np.log(h(z))
J2 = lambda z,y: -(1-y)*np.log(1-h(z))

# Ref metricas
m1 = lambda z: z-1 



#===================
z1 = np.linspace(-2, 10)
z2 = np.linspace(-10, 2)
plt.plot(z1,J1(z1,1), label="y=1")

plt.plot(z2,J2(z2,0), label="y=0")
plt.legend()
plt.xlabel("Z")
plt.ylabel("Cost function")


# - Para $y = 1$ se cumple que para $Z=\theta^T X\geq 0$ la clasificacion será tipo 1 
# 
# - Para $y = 0$ se cumple que para $Z=\theta^T X < 0 $ la clasificación será tipo 0
# 
# Las métricas anteriores pueden ser definidas en términos de métricas que permitan clasificar en los siguiente intervalos, segun la curva roja y negra definida en la gráfica:
# 
# - Para $y = 1$ se cumple que para $Z=\theta^T X\geq 1$ la clasificacion será tipo 1 
# 
# - Para $y = 0$ se cumple que para $Z=\theta^T X < 1 $ la clasificación será tipo 0
# 

# Si interpretamos la funcion $J(\theta)$ como:
# 
# $J(\theta)=A+\lambda B$, podemos rescribir la anterior expresión como:
# 
# $J(\theta)=C A'+B'$ 
# 
# Siendo $C=\frac{1}{\lambda}$, el inverso del parametros de regularación descrito en las sesiones anteriores.
# 
# 
# Nuestro objetivo sera mínimizar la función  $\min [J(\theta))] =\min[ C A'+B']$, 
# 
# El termino B' de la anterior expresión puede ser expresado como sigue:
#  
#  $B' = \frac{1}{2}\sum_{j=1}^{n}\theta_j^2=\frac{1}{2} (\theta_1^2 + \theta_2^2+...+\theta_n^2)=\frac{1}{2}||\theta||^2$
# 
# ## Interpretacion geométrica
# 
# Supongamos que tenemos dos caracteristicas, en nuestro sistema, de esta manera tenemos que:
# 
# $\theta^T X= [\theta_1, \theta_2]  [x1,x2]$
# 
# Continuar con los aspectos teóricos...
# 
# 
# 
# References
# [1] https://www.cienciadedatos.net/documentos/34_maquinas_de_vector_soporte_support_vector_machines
# 

# In[ ]:


import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Libraries for draw contours
def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_contoursExact(ax, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# In[ ]:


# Dataset Toys References
# https://scikit-learn.org/stable/datasets/toy_dataset.html
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# Dataset sinteticos
X0, y0 = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, 
    n_clusters_per_class=1
)
X1, y1 = make_moons(n_samples=100, noise=0.15, shuffle=True,  random_state=1)
X2, y2 = make_circles(n_samples=100, noise=0.05, shuffle=True,  random_state=1)
X3, y3 = make_blobs(n_samples=500, centers=3, n_features=2,shuffle=True, 
                    random_state=10)

fig, axs = plt.subplots(2,2)

axs[0, 0].plot(X0[:,0][y0==0],X0[:,1][y0==0],"ro", alpha=0.5)
axs[0, 0].plot(X0[:,0][y0==1],X0[:,1][y0==1],"bo", alpha=0.5)

# Dataset a moons
axs[0, 1].plot(X1[:,0][y1==0],X1[:,1][y1==0],"ro", alpha=0.5)
axs[0, 1].plot(X1[:,0][y1==1],X1[:,1][y1==1],"bo", alpha=0.5)

# Dataset circles
axs[1, 0].plot(X2[:,0][y2==0],X2[:,1][y2==0],"ro", alpha=0.5)
axs[1, 0].plot(X2[:,0][y2==1],X2[:,1][y2==1],"bo", alpha=0.5)

# Dataset circles
axs[1, 1].plot(X3[:,0][y3==0],X3[:,1][y3==0],"ro", alpha=0.5)
axs[1, 1].plot(X3[:,0][y3==1],X3[:,1][y3==1],"bo", alpha=0.5)
axs[1, 1].plot(X3[:,0][y3==2],X3[:,1][y3==2],"go", alpha=0.5)


# In[ ]:


# Based on :
# https://rramosp.github.io/ai4eng.v1.20211.udea/content/NOTES%2003.03%20-%20SVM%20AND%20FEATURE%20TRANSFORMATION.html


# In[ ]:


X, y = make_circles(n_samples=200, noise=0.1, shuffle=True,  random_state=1)

plt.plot(X[:,0][y==0],X[:,1][y==0],"ro", alpha=0.5)
plt.plot(X[:,0][y==1],X[:,1][y==1],"bo", alpha=0.5)


# In[ ]:


clf = SVC(gamma = 0.10) #Complexity of algorithm 
clf.fit(X, y)

#Countour plot
fig, ax = plt.subplots()
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.plot(X[y==0][:,0],X[y==0][:,1],"bo", alpha=1)
plt.plot(X[y==1][:,0],X[y==1][:,1],"ro", alpha=1)
print(f"Training error:{clf.score(X, y):.3f}")


# # Grid Search of parameter in SVM
# 
# Grid-search is used to find the optimal hyperparameters of a model which results in the most ‘accurate’ predictions.
# https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e

# In[ ]:


#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import GridSearchCV


X, y = make_circles(n_samples=200, noise=0.1, shuffle=True,  random_state=1)

plt.plot(X[:,0][y==0],X[:,1][y==0],"ro", alpha=0.5)
plt.plot(X[:,0][y==1],X[:,1][y==1],"bo", alpha=0.5)


# In[ ]:


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
parameters = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

clf = GridSearchCV(estimator=SVC(),
             param_grid = parameters)

clf.fit(X, y)
sorted(clf.cv_results_.keys())


# In[ ]:


clf.cv_results_


# In[ ]:


clf.best_estimator_


# In[ ]:


clf.best_params_


# In[ ]:


clf.best_score_


# In[ ]:


#Countour plot
fig, ax = plt.subplots()
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.plot(X[y==0][:,0],X[y==0][:,1],"bo", alpha=1)
plt.plot(X[y==1][:,0],X[y==1][:,1],"ro", alpha=1)
print(f"Training error:{clf.score(X, y):.3f}")


# Tarea 11.1 
# 1. Implementar un SVM para clasificar los siguientes datasets, para ello se deberá crear un grid search. 
# 2. Con los mejores párametros dibujar  las fronteras de clasificación. 
# ```
# X0, y0 = make_classification(
#     n_features=2, n_redundant=0, n_informative=2, random_state=1, 
#     n_clusters_per_class=1
# )
# X1, y1 = make_moons(n_samples=100, noise=0.15, shuffle=True,  random_state=1)
# X3, y3 = make_blobs(n_samples=500, centers=3, n_features=2,shuffle=True, random_state=10)
# ```

# In[ ]:


# Dataset Toys References
# https://scikit-learn.org/stable/datasets/toy_dataset.html
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# Dataset sinteticos
X0, y0 = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, 
    n_clusters_per_class=1
)
X1, y1 = make_moons(n_samples=100, noise=0.15, shuffle=True,  random_state=1)
X2, y2 = make_circles(n_samples=100, noise=0.05, shuffle=True,  random_state=1)
X3, y3 = make_blobs(n_samples=500, centers=3, n_features=2,shuffle=True, 
                    random_state=10)

fig, axs = plt.subplots(2,2)

axs[0, 0].plot(X0[:,0][y0==0],X0[:,1][y0==0],"ro", alpha=0.5)
axs[0, 0].plot(X0[:,0][y0==1],X0[:,1][y0==1],"bo", alpha=0.5)

# Dataset a moons
axs[0, 1].plot(X1[:,0][y1==0],X1[:,1][y1==0],"ro", alpha=0.5)
axs[0, 1].plot(X1[:,0][y1==1],X1[:,1][y1==1],"bo", alpha=0.5)

# Dataset circles
axs[1, 1].plot(X3[:,0][y3==0],X3[:,1][y3==0],"ro", alpha=0.5)
axs[1, 1].plot(X3[:,0][y3==1],X3[:,1][y3==1],"bo", alpha=0.5)
axs[1, 1].plot(X3[:,0][y3==2],X3[:,1][y3==2],"go", alpha=0.5)


# In[ ]:




