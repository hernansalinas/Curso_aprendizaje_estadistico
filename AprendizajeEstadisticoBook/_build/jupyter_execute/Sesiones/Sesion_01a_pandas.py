#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hernansalinas/Curso_aprendizaje_estadistico/blob/main/Sesiones/Sesion_01b_pandas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# 
# 
# 

# # Repaso de python. 
# Objetivo: El objetivo de esta sesion es estudiar conceptos basicos de python necesarios para realizar modelos de maching learning
# 
# 0. Python Básico
# 1. Objetos 
# 2. Operación filter, map, reduce 
# 3. Pandas,  
# 4. Matplotlib , Sea born
# 5. Aplicaciones
# 
# 
# 

# # Python Philosophy..
# 
# https://www.python.org/dev/peps/pep-0020/#abstract
# 

# In[1]:


import this


# # Functions, Loops, Conditionals, and list comprehensions
# 
# 

# In[2]:


my_list = []
for number in range(0, 10):
  if number % 2 == 0:
    my_list.append(number)
my_list


# In[3]:


my_list = [number for number in range(0, 10)  if number %2 ==0]


# In[4]:


my_list


# In[5]:


def times_tables():
    """    
    Params:
      --  

    Return:
      -- lst: List

    """
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i*j)
    return lst


# In[6]:


times_tables() == [j*i for i in range(10) for j in range(10)]


# # Built-in Functions
# https://docs.python.org/3/library/functions.html
# 
# ## Function map

# In[7]:


import numpy as np
b = map(lambda x: x**2, range(10))


# In[8]:


type(b)


# In[9]:


list(b)


# In[10]:


sum(b)


# In[11]:


def squares_0():
  squares = []
  for x in range(10000):
    squares.append(x**2)
  return squares


# Operation use in big data files
def squares_1():
  return list(map(lambda x: x**2, range(10000)))


def squares_2():
  return [x**2 for x in range(10000)]


# In[12]:


squares_0()


# In[13]:


import numpy as np
import time


# In[14]:


times = [] 
tmax = 10000
for i in range(0,tmax):
  # Common for
  tini0 = time.time()
  squares_0()
  tend0 = time.time()
  # Operation map
  tini1 = time.time()
  squares_1()
  tend1 = time.time()
  # Comprhension expresion 
  tini2 = time.time()
  squares_2()
  tend2 = time.time()    
  times.append([tend0-tini0, tend1-tini1, tend2-tini2] )

t=np.array(times)


# In[39]:


mean = np.mean(t, axis = 0)
print(mean)


# In[41]:


# Magic command
# https://ipython.readthedocs.io/en/stable/interactive/magics.html


# In[54]:


get_ipython().run_cell_magic('timeit', '-n 100 ', 'squares_0()')


# In[55]:


get_ipython().run_cell_magic('timeit', '-n 1000 ', 'squares_1()')


# In[56]:


get_ipython().run_cell_magic('timeit', '-n 1000  ', 'squares_2()')


# # Lambda function
# 

# In[267]:


f = lambda x: (x+2) 


# In[268]:


f(2)


# # Filter function
# 

# Filtros avanzados en python a través de programacion funcional
# 
# Ref = https://docs.hektorprofe.net/python/funcionalidades-avanzadas/funcion-filter/, 
# 
# https://docs.python.org/3/library/functions.html#filter

# In[282]:


def multiple(numero):    # Primero declaramos una función condicional
    if numero % 5 == 0:  # Comprobamos si un numero es múltiple de cinco
        return True      # Sólo devolvemos True si lo es

numeros = [2, 5, 10, 23, 50, 33, 5000]
a = filter(multiple, numeros)


# In[283]:


a


# In[284]:


list(a)


# In[285]:


list(a)


# # FUNCIÓN MAP

# In[286]:


a = [10.00, 11.00, 12.34, 2.0 ]
b = [9.8, 11.10, 12.34, 2.01 ]


# In[287]:


var = map(min, a, b)
var


# In[288]:


list(var)


# In[289]:


var


# In[290]:


list(var)


# In[294]:


# Este resultado no muestra nada, ¿porqué?. 
#La variable var ya fue evaluada a través de elementos funcionales
for item in var:
  print(item)


# Otro ejemplo con la operación map: Dejar el apellido de las siguientes personas

# In[295]:


people = ['Dr. Simon Einstein', 'Dr. Pedro Euler  ', 'Dr. Juan Tesla', 'Dr. Daniel Maxwell']
people[0].split()


# In[296]:


def split_title_and_name(person):  
    title = person.split()[0]
    lastname = person.split()[-1]
    return f'{title} {lastname}'

last_names = map(split_title_and_name, people)
list(last_names)


# In[297]:


a = []
for p in people:
  a.append(split_title_and_name(p))
a


# # Tarea 0.1 
# Determinar los primeros 100 numeros impares empleando la funcion map
# 

# In[298]:


def impar(x):
  if(x%2!=0):
    return x

q = map(lambda x: 2*x+1 if (2*x+1)<=100 else 0 , range(50))


# In[299]:


#list(q)


# # Objetos

# In[128]:


class auto:
  """
  Esta clase asigna un color y un tipo 
  a un clase tipo carro
  """
  var = "taller de carros"

  def set_name_tipo(self, new_tipo):
    self.tipo = new_tipo

  def set_name_color(self, new_color ):
    self.color = new_color


# In[129]:


carro = auto()
carro.set_name_color="rojo"
carro.set_name_tipo="bus "
print(f"El carro es {carro.set_name_color} y es un  {carro.set_name_tipo} " )


# In[211]:


class circulo(object):
  def __init__(self, R, posx, posy ):
    self.R1 = R
    self.posx = posx
    self.posy= posy
  
  def Area(self):
    A = np.pi*(self.R1)**2
    return A  
  def perimetro(self):
    return 2*np.pi*self.R1


class circulo_(object):
  def __init__(self):
    self.R1 = None
    self.posx = None
    self.posy= None
  
  def Area(self):
    A = np.pi*(self.R1)**2
    return A
  
  def perimetro(self):
    return 2*np.pi*self.R1
   
   


# In[212]:


c = circulo(1, 0, 0)


# In[213]:


c.Area()
c.perimetro()


# In[214]:


cc=circulo_()


# In[223]:


cc.posx=1
cc.posy=1
cc.R1=1


# In[224]:


cc.R1


# In[225]:


cc.perimetro()


# In[260]:





# ## Tarea 0.2
# Given a sentence,you task is build a iterator the words 
# Ref = https://www.youtube.com/watch?v=C3Z9lJXI6Qw&ab_channel=CoreySchafer

# In[353]:


class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence
        self.index = 0
        self.words = self.sentence.split()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.words):
            raise StopIteration
        index = self.index
        self.index += 1
        return self.words[index]

my_sentence = sentence('This is a test')

print(next(my_sentence))
print(next(my_sentence))
print(next(my_sentence))
print(next(my_sentence))
#print(next(my_sentence))


# In[263]:


a = Sentence("hola mundo esta es una prueba")


# In[265]:


a.__next__()


# # Diccionarios
# 
# Elementos basicos con diccionarios

# In[ ]:


students_class = { "Bob": "Physics","Alice": "Physics","Ana": "Biology" }


# In[ ]:


for i, s in enumerate(students_class):
  print(i, s, students_class[s])


# In[ ]:


students_class.items()


# Otra forma de iteraciones para los diccionarios a través del metodo items()

# In[ ]:


for key, val in students_class.items():
  print(key, val)


# Accediendo a los valores del diccionario
# Accediendo a las claves
# - metodo keys()
# - metodo values()

# In[ ]:


print(students_class.values())
print(students_class.keys())


# # Pandas.
# 
# ## Series
# ## Data Frame

# In[300]:


import pandas as pd
students_class = { "Bob": "Physics",
                  "Alice": "Chemistry",
                  "Ana": "Biology" }


# In[301]:


# Ndarray unidimensional con ejes etiquetados
s = pd.Series(students_class)
s


# In[307]:


# https://pandas.pydata.org/docs/reference/series.html
print(type(s.index))
s.index


# In[308]:


#Forma de acceder a los elementos con el número del indice
s.iloc[2]


# In[309]:


#Forma de acceder a los indices
s.loc["Alice"]


# In[310]:


s.Bob


# ### Definición clave valor con enteros como clave. 

# In[311]:


class_code = {99:"Physics", 
              100:"Chemistry", 
              101:"English" }


# In[312]:


s = pd.Series(class_code)


# In[313]:


s


# In[314]:


s.iloc[2]


# In[315]:


s.loc[99]


# Tambien podemos definir el objeto Serie  a partir de una lista

# In[316]:


grades = pd.Series([8,7,10,1])


# In[317]:


grades


# In[318]:


for i, g in enumerate(grades):
  print(i,g)


# In[319]:


grades.mean()


# In[320]:


grades.describe()


# Definicion a través de un  numpy array 
# 
# 

# In[321]:


x = np.random.randint(0,20, 100)
random_s = pd.Series(x)


# In[322]:


random_s


# In[323]:


random_s.head()


# In[324]:


#Recorrido por las claves y valores, el metodo head es considerado para mostrar pocos valores

for index, values in random_s.head().iteritems():
  print(index, values)


# In[325]:


get_ipython().run_cell_magic('timeit', '-n 100', 'x = np.random.randint(0,20, 100)\nrandom_s = pd.Series(x)\nrandom_s+=2 # OPeraciones vectoriales a todo el data frame, más eficiente.\n\n#Comparar cuando se tiene un ciclo para realizar la suma, ¿cuál es mas eficiente?')


# Agregando nuevos valores con indices diferentes
# 
# 

# In[326]:


s = pd.Series([1,2,3,4,9])


# In[327]:


s.loc["nuevo"]=2


# In[328]:


s


# In[329]:


s.loc["nuevo"]


# In[330]:


s["nuevo"]


# In[331]:


s.iloc[-1]


# Otra *forma* de definir una serie es a través de :

# In[332]:


juan_class = pd.Series(["a", "b","c"], index=["0","1","2"])


# In[333]:


juan_class


# # Data Frame 
# 
# 
# 
# Un DataFrame es una lista  de series
# 
# 

# In[334]:


d1 = { "Name":"Juan", "Topic":"Quantum Mechanics", "Score" : 10}
d2 = { "Name":"Pedro", "Topic":"statistical", "Score" : 10}
d3 = { "Name":"Ana", "Topic":"Clasical Mechanics", "Score" : 10}

record1 = pd.Series(d1)
record2 = pd.Series(d2)
record3 = pd.Series(d3)


# In[335]:


# indices con números enteros
df1 = pd.DataFrame( [record1, record2, record3] )
df1


# In[337]:


# Asignando nombre a los indices
df2 = pd.DataFrame( [record1, record2, record3] , index = ["UdeA","Unal", "ITM"] )
df2


# In[338]:


#Accediendo a los indices por el nombre
df2.loc["UdeA"]


# In[339]:


#Accediendo a los indices por el numero
df2.iloc[0]


# In[340]:


#Accediendo a un elemento en particular
df2.loc["UdeA", "Name"]


# In[341]:


#Accediendo a algunas columnas del data frame
df2.loc[:, ["Name", "Topic"]]


# Se recomienda crear copias del data frame cuando se esta trabajando con pandas a traves del metodo copy() y no con el operador =, dado que se comparte el mismo espacio de memoria

# In[342]:


df2


# In[343]:


a = df2


# In[344]:


a


# In[345]:


a.loc["UdeA", "Name"] = "JuanB"


# In[346]:


a


# In[347]:


df2


# In[348]:


b = df2.copy()


# In[348]:





# Eliminacion de columnas

# In[349]:


del b["Topic"]


# In[350]:


b


#  Agregando nuevas columnas al data frame

# In[351]:


b["Nueva"] = [10, 8, 3]


# In[352]:


b


# # Tarea 0.3
# 
# Empleando  los siguientes tiempos:
# ```
# t = np.linspace(0, 2, 1000)
# ```
# 1. Crear un data frame de pandas para la posicion $y=ho-0.5gt^2$, $g=9.8m/s$, $h = 100 m$ 
# 2. Adicione una nueva columna para la velocidad y la aceleración.
# 
# Construya un nuevo data frame desde el tiempo t=0.5s a 1.5s, solo con las posición como funcion del tiempo.

# In[354]:


# Exercise
#https://github.com/ajcr/100-pandas-puzzles/blob/master/100-pandas-puzzles.ipynb


# In[ ]:




