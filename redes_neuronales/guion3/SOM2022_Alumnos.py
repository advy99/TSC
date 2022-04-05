#!/usr/bin/env python
# coding: utf-8

# In[219]:


### Los datos son del repositorio de UCI
### Aquí detectamos los valores atípicos que serán justamente los clientes fraudulentos
## Necesitamos calcular la MID ((Mean Inter-Neuron Distance(Distancia media 
## entre neuronas)), que es la distancia euclidia media de la neurona principal con su vecin
### Los valores atípicos estarán lejos de la media
### Clase 1 son los clientes cuyas solicitudes fueron aprobadas
### Pueden ser fraudulentos


# In[1]:



# Importando las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importando el conjunto de datos

dataset = pd.read_csv('Credit_Card_Applications.csv')


# In[3]:


dataset.head()


# In[5]:


# iloc obtiene índices de observación, todas las líneas que usamos: y todas las columnas excepto
# por último: -1 y valores

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[6]:


dataset.shape


# In[7]:


# dividir el conjunto de datos en X e Y, sin intentar hacer un aprendizaje supervisado o una clasificación 0 o 1
# haciendo distinción de clientes aprobados y no aprobados, solo usaremos el SOM con
# X y no hay variable dependiente

# Escalado de características: obligatorio para el aprendizaje profundo porque estamos comenzando con un alto
# conjunto de datos dimensionales con muchas relaciones no lineales y será mucho más fácil
# para que nuestros modelos de aprendizaje profundo  si las características se escalan.
# utiliza la normalización todas las funciones de 0 a 1

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))


# In[8]:


# ajusta el objeto sc a X para que sc obtenga toda la información (min y max) y toda la información para la normalización
# aplica la normalización a X, el método de ajuste devuelve la versión normalizada de X


X = sc.fit_transform(X)


# In[16]:




# In[18]:


# Entrenando el SOM

### x e y son las dimensiones del SOM (no debe ser pequeño para que se puedan detectar valores atípicos)
### input_len son el número de características de X (la identificación del cliente se incluye para encontrar la identificación de los infractores)
### sigma es el radio
# Aprendizaje no supervisado,
# sigma es el radio de los diferentes vecindades
# razon de aprendizaje, el hiperparámetro decide cuánto se añade de cada incremento de peso, 
# mayores tasas de aprendizaje, pueden hacer más rápida la convergencia
# Tasa de aprendizaje bajas hacen que el entrenamiento pueda ir mas lento,

from minisom import MiniSom    

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
#som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
#som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.05)


# In[228]:


# inicializa aleatoriamente los vectores de pesos a números pequeños cercanos a 0

som.random_weights_init(X)


# In[229]:


# entrenar SOM con X, matriz de características y patrones reconocidos

# som.train_random(data = X, num_iteration = 5000)
som.train_random(data = X, num_iteration = 5000, verbose=1)


# In[230]:


# Visualizando los resultados
# cuadrícula bidimensional de los nodos ganadores


### Cuanto mayor sea la distancia media entre neuronas MID, más lejos estará el nodo ganador de su vecindario
### por lo tanto, será más probable que sea un valor atípico
### La BMU lejos de su vecindario generalmente está lejos de los clústeres
### los colores más cercanos al blanco son valores atípicos



# In[231]:



from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


# In[232]:


mappings=som.win_map(X)


# In[ ]:


## Recuerda que las coordenadas de matriz que tienes que poner abajo
## para presentar los individuos candidatos a fraude son aquellas 
## coordenadas cuya neurona haya agrupado individuos con distancia 
## maxima a la promedio representados por color blanco


# In[233]:


fraudes=np.array(mappings[5,1])


# In[234]:


fraudes=sc.inverse_transform(fraudes)


# In[ ]:


fraudesInt=fraudes.astype(int)
fraudesInt


# In[ ]:


### A continuación pintaremos la matriz de distancias del SOM y
### contaremos aquellos clientes que superen la distancia de 0.5 
### puesto que serán los posibles clientes fraudulentos
### esta es otra forma de ver posibles candidatos


# In[ ]:


distance_map = som.distance_map().round(1)


# In[ ]:


distance_map


# In[ ]:


index = []
for i in range(10):
    for j in range(10):
        if(distance_map[i,j]>=0.5):
            index.append([i,j])
len(index)


# In[ ]:




