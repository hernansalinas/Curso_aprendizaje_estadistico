#!/usr/bin/env python
# coding: utf-8

# 
# <a href="https://colab.research.google.com/github/hernansalinas/Curso_aprendizaje_estadistico/blob/main/Sesiones/Sesion_01a_pandas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# # Aprendizaje Estadístico vs. Aprendizaje Automático
# 
# Tomado de : https://rubenfcasal.github.io/aprendizaje_estadistico/aprendizaje-estad%C3%ADstico-vs.-aprendizaje-autom%C3%A1tico.html
# 
# - El término Machine Learning (ML; Aprendizaje Automático) se utiliza en el campo de la Intelingencia Artificial desde 1959 para hacer referencia, fundamentalmente, a algoritmos de predicción (inicialmente para reconocimiento de patrones). Muchas de las herramientas que utilizan provienen del campo de la Estadística y, en cualquier caso, la Estadística (y por tanto las Matemáticas) es la base de todos estos enfoques para analizar datos (y no conviene perder la base formal). Por este motivo desde la Estadística Computacional se introdujo el término Statistical Learning (Aprendizaje Estadístico) para hacer referencia a este tipo de herramientas, pero desde el punto de vista estadístico (teniendo en cuenta la incertidumbre debida a no disponer de toda la información).
# 
# Tradicionalmente ML no se preocupa del origen de los datos e incluso es habitual que se considere que un conjunto enorme de datos es equivalente a disponer de toda la información (i.e. a la población).
# 
# “The sheer volume of data would obviate the need of theory and even scientific method” --- Chris Anderson, físico y periodista, 2008
# 
# - Por el contrario en el caso del AE se trata de comprender, si es posible, el proceso subyacente del que provienen los datos y si estos son representativos de la población de interés (i.e. si tienen algún tipo de sesgo). No obstante, en este libro se considerará en general ambos términos como sinónimos.
# 
# ML/AE hacen un importante uso de la programación matemática, ya que muchos de sus problemas se plantean en términos de la optimización de funciones bajo restricciones. Recíprocamente, en optimización también se utilizan algoritmos de ML/AE.
# 
# ## Machine Learning vs. Data Mining
# Mucha gente utiliza indistintamente los nombres ML y Data Mining (DM). Sin embargo, aunque tienen mucho solapamiento, lo cierto es que hacen referencia a conceptos ligeramente distintos.
# 
# - ML es un conjunto de algoritmos principalmente dedicados a hacer predicciones y que son esencialmente automáticos minimizando la intervención humana.
# 
# - DM intenta entender conjuntos de datos (en el sentido de encontrar sus patrones), requiere de una intervención humana activa (al igual que la Inferencia Estadística tradicional), pero utiliza entre otras las técnicas automáticas de ML. Por tanto podríamos pensar que es más parecido al AE.
# 
# ## Las dos culturas (Breiman, 2001b)
# Breiman diferencia dos objetivos en el análisis de datos, que él llama información (en el sentido de inferencia) y predicción. Cada uno de estos objetivos da lugar a una cultura:
# 
# - **Modelización de datos:** desarrollo de modelos (estocásticos) que permitan ajustar los datos y hacer inferencia. Es el trabajo habitual de los estadísticos académicos.
# 
# 
# 
# - **Modelización algorítmica:** (en el sentido de predictiva): esta cultura no está interesada en los mecanismos que generan los datos, sólo en los algoritmos de predicción. Es el trabajo habitual de muchos estadísticos industriales y de muchos ingenieros informáticos. El ML es el núcleo de esta cultura que pone todo el énfasis en la precisión predictiva (así, un importante elemento dinamizador son las competiciones entre algoritmos predictivos, al estilo del Netflix Challenge).
# 
# 
# 
# Breiman, L. (2001b). Statistical modeling: The two cultures (with comments and a rejoinder by the author). Statistical Science, 16(3), 199-231. https://doi.org/10.1214/ss/1009213726
# 
# ## Machine Learning vs. Estadística (Dunson, 2018)
# 
# 
# - **Machine learning (ML) community**: tends to have its roots in engineering, computer science, and to a certain extent neuroscience – growing out of artificial intelligence (AI). The main publication outlets tend to be peer-reviewed conference proceedings, such as Neural Information Processing Systems (NIPS), and the style of research is very fast paced, trendy, and driven by performance metrics in prediction and related tasks. One
# measure of “trendiness” is the fact that there is a strong auto-correlation in the main focus areas that are represented in the papers accepted to NIPS and other top conferences. For example, in the past several years much of the focus has been on deep neural network methods. The ML community also has a tendency towards marketing and salesmanship, posting talks and papers on social media and attempting to sell their ideas to the broader public. This feature of the research seems to reflect a desire or tendency to want to monetize the algorithms in the near term, perhaps leading to a focus on industry problems
# over scientific problems, where the road to monetization is often much longer and less assured. ML marketing has been quite successful in recent years, and there is abundant interest and discussion in the general public about ML/AI, along with increasing success in start-ups and industrial sector high paying jobs partly fueled by the hype.
# 
# 
# - **Statistical (Stats) community**: made up predominantly of researchers who received their initial degree(s) in mathematics followed by graduate training in statistics. The main publication outlets are peer-reviewed journals, most of which have a long drawn  out review process, and the style of research tends to be careful, slower paced, intellectual as opposed to primarily performance driven, emphasizing theoretical support (e.g., through asymptotic properties), under-stated, and conservative. Statisticians tend to be reluctant to market their research, and their training tends to differ dramatically from that for most ML researchers. Statisticians usually have a mathematics base including multivariate calculus, linear algebra, differential equations, and real analysis. They then take several years of probability and statistics, including coverage of asymptotic theory, statistical sampling theory, hypothesis testing, experimental design, and many other areas. ML researchers coming out of Computer Science and Engineering have much less background in many of these areas, but have a stronger background in signal processing, computing (including not just programming but also an understanding of computer
# engineering and hardware), optimization, and computational complexity.
# 
# 
# Dunson, D. B. (2018). Statistics in the big data era: Failures of the machine. Statistics and Probability Letters, 136, 4-9. https://doi.org/10.1016/j.spl.2018.02.028

# # Métodos de Aprendizaje Estadístico
# 
# Dado un conjunto de predictores(caracteristicas) $X_1,X_2,X_3 ... X_p $ existe una predicion $Y$ relacionada con X , que puede ser escrita como modelada como :
# 
# $Y=f(X) + \epsilon$, 
# 
# Asi, el aprendizaje estadistico hace referencia al conjunto de enfoques sobre los cuales se puede estimar $f$, teniendo presente la naturaleza de los datos. 
# 
# ¿Cual es la mejor forma de estimar f ?
# De acuerdo con el teorema,  [No free lunch theorem](https://ti.arc.nasa.gov/m/profile/dhw/papers/78.pdf), existen diferentes algoritmos que permiten estimar f, y por cada par de algoritmos de optimizacion, hay tantos en el que el primer algoritmo es mejor que el segundo, como problemas en el que el segundo es mejor que el primero. 
# Todos los algoritmos de optimizacion se compartan igual de bien o de mal en prmedio frente a diferentes tipos de problemas. No se ha evidenciado un algoritmo de optmizacion general para cualquier problema. Es necesario incorporar cierto conocmiento especifico del problema.
# 
# En otras palabras:
# 
# "Si tu unica herramienta es un martillo", tratas cada problema como si fuera un clavo(Abraaham Maslow, 1966). Cada algoritmos de clasificacion tiene sus sesgos inherentes y ninguna clasificacion individual es superior si no hacemos suposicion sobre la tarea. En la practica se hace necesario comparar un puñado de algoritmos distintos para entrenar y seleccionar el mejor modelo de rendimiento. Se debe definir una unidad para definir el rendimiendo de cada algoritmo.
# 
# 
# 
# Algunos lecturas de interés: 
# - [Comentarios de no-free-lunch](https://victoryepes.blogs.upv.es/2020/10/14/no-free-lunch/)
# 
# - [The Lack of A Priori Distinctions Between Learning Algorithms](https://ieeexplore.ieee.org/document/6795940)
# 
# 
# 
# 
# # Aprendizaje supervisado :
#   
# Este forma de aprendizaje puede estar clasificada en dos tipos:
# 
#  - *Algoritmos de regresion*: Regresion para predecir resultados
#  - *Algoritmos de clasificacion*: Clasificar para predecir etiquetas
# 
# 
# 
# 
# ## Taxonomia:
#   - Regresion lineal 
#   - arboles de desicion
#   - Naive-Bayes
#   - Random-Forest
#   - SVM
# 
# El aprendizaje supervisado parte de  datos etiquetados para encontrar una funcion f tal que cuando se ingresado una nueva muestra de datos(predictores) el modelo logre predecir que etiqueta deben de tener los datos. 
# 
# 
# Incluir imagen del modelo
# Imagen de Coursera
# 
# # Aprendizaje No supervisado 
#   - Clustering: Kmean, DBSCAN, Hierarchical Cluster Analysis(HCA)  
#   - Anomaly detection and novelty detection (One-calss SVM), Isolation forest
#   - Visualization and dimensionality reduction( PCA, kernel PCA, Locally linear embedding LLE)
#   - t-Distributed Stochastic Neighbor Embedding 
#   - Association rule Learning 
#   
# # Aprendizaje Reforzado
# Incluir imagen 

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# # Herramientas del curso
# 1. Git hub
# 2. Anaconda
# 3. Virtual env
# 4. Principales librerias: Matplotlib, pandas, numpy, seaborn.
# 
# 
# 
# #Evalución del curso#
# 1. Tareas 70%  (Conjunto de tareas, entregadas a través del git.)
# 2. Proyecto Final. 30% (Tiempo son los últimos 2 meses)
#     - Exposición, Artículo
#     - Planificación. 
#     - Desarrollo.
#     - Análisis.
# 
# 
# 
# # Preeliminares.
# 1. Herramientas del curso.
# 2. Crear guia de instación 
# 3. Crear guia del github para subir los archivos.
# 
# 
# 
# 
# 
# 

# 
