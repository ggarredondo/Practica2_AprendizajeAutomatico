# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Guillermo García Arredondo
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
seed = 1
np.random.seed(seed)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
print("-Ejercicio 1-\n")
print("-1.1.a-\n\nSe muestra gráfica...")

# 1.1.a. - Dibujar la gráfica para N=50, d=2 y rango=[-50,50] para una distribución uniforme.
x = simula_unif(50, 2, [-50,50]) # Extraemos 50 datos aleatorios de 2 dimensiones de una distribución uniforme
                                 # en el intervalo [-50,50].
plt.scatter(x[:,0], x[:,1]) # Los mostramos en una gráfica como nube de puntos.
plt.title("Ejercicio 1.1.a.")
plt.show()

input("--- Pulsar tecla para continuar al ejercicio 1.1.b ---\n")
print("Se muestra gráfica...")

# 1.1.b. - Dibujar la gráfica para N=50, d=2 y sigma=[5,7] para una distribución normal/gaussiana.
x = simula_gaus(50, 2, np.array([5,7])) # Extraemos 50 datos aleatorios de 2 dimensiones de una distribución normal
                                        # en el intervalo [-50,50].
plt.scatter(x[:,0], x[:,1]) # Los mostarmos en una gráfica como nube de puntos.
plt.title("Ejercicio 1.1.b.")
plt.show()

input("--- Pulsar tecla para continuar al ejercicio 1.2 ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.2: Valoración de la influencia del ruido en la selección de la complejidad de la clase.
print("\n-1.2.a-\n")

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

# Función f para comprobar si un punto está por encima y por debajo de una recta
# dada por a y b (y = a*x + b)
def f(x, y, a, b):
	return signo(y - a*x - b)

# 1.2.a. - Dibujar un gráfico 2D donde los puntos muestren el resultado de su etiqueta.
# Dibuje también la recta usada para etiquetar.

print("Se muestra gráfica...")

x = simula_unif(100, 2, [-50,50]); # Obtenemos 100 puntos aleatorios de una distribución uniforme en el intervalo anterior.
a, b = simula_recta([-50,50]); # Obtenemos una recta dada por dos puntos aleatorios en el mismo intervalo.
y = np.array([f(x, y, a, b) for x, y in x], dtype=np.float64) # Etiquetamos los puntos según su posición respecto a la recta.

# Para mostrar los datos y la recta en una gráfica...
recta_x = np.linspace(-50, 50, 2) # Obtenemos dos puntos equidistantes en el intervalo anterior.
recta_y = a*recta_x + b # Calculamos las coordenadas y para los dos puntos aleatorios antes obtenidos (y = a*x + b).
plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], c="purple") # Los puntos con signo positivo se dibujan en la
                                                                        # gráfica con color morado.
plt.scatter(x[np.where(y == -1), 0], x[np.where(y == -1), 1], c="orange") # Los puntos con signo negativo se dibujan en la
                                                                          # gráfica con color naranja.
plt.plot(recta_x, recta_y, c="red") # Usando los puntos calculados antes para la recta mostramos la frontera en la gráfica
                                    # con color rojo.
plt.legend(("Frontera","+1","-1"))
plt.title("Ejercicio 1.2.a.")
plt.show()

input("--- Pulsar tecla para continuar al ejercicio 1.2.b ---\n")

# 1.2.b. - Modifique de forma aleatoria el 10% de las etiquetas positivas y otro 10% de las
# negativas y guarde los puntos con sus nuevas etiquetas. Dibuje de nuevo la gráfica anterior.

# Función para generar ruido en un vector de etiquetas 'y'.
def generar_ruido(y):
    y_positivo = np.where(y == 1)[0] # Obtenemos los índices de las etiquetas que son de signo positivo.
    y_negativo = np.where(y == -1)[0] # Obtenemos los índices de las etiquetas son de signo negativo.
    # Cogemos un 10% de índices aleatorios de las etiquetas positivas y otro 10% de las negativas, 
    # asegurándonos que los índices no se pueden repetir especificando 'replace=False'.
    index_positivo = np.random.choice(y_positivo, y_positivo.size//10, replace=False)
    index_negativo = np.random.choice(y_negativo, y_negativo.size//10, replace=False)
    # Multiplicamos por -1 el 10% aleatorio de etiquetas positivas y el 10% aleatorio de etiquetas negativas.
    y[index_positivo] *= -1
    y[index_negativo] *= 1
    
print("Se muestra gráfica...")
y_ruido = y.copy() # Hago una copia en memoria de 'y' porque sino los cambios en 'y_ruido' afectarían tambień a 'y'
                   # por ser arrays de numpy.
# Se introduce el ruido y se visualiza con la misma recta frontera anterior.
generar_ruido(y_ruido)
plt.scatter(x[np.where(y_ruido == 1), 0], x[np.where(y_ruido == 1), 1], c="purple")
plt.scatter(x[np.where(y_ruido == -1), 0], x[np.where(y_ruido == -1), 1], c="orange")
plt.plot(recta_x, recta_y, c="red")
plt.legend(("Frontera","+1","-1"))
plt.title("Ejercicio 1.2.b.")
plt.show()

input("--- Pulsar tecla para continuar al ejercicio 1.2.c ---\n")

# 1.2.c. - Supongamos ahora que las siguientes funciones definen la 
# frontera de clasificación de los puntos de la muestra en lugar de una recta.
# Visualizar el etiquetado generado en 2b junto con cada una de las gráficas de cada
# una de las funciones.

def f1(x):
    return (x[:,0]-10)**2 + (x[:,1]-20)**2 - 400

def f2(x):
    return 0.5*(x[:,0]+10)**2 + (x[:,1]-20)**2 - 400

def f3(x):
    return 0.5*(x[:,0]-10)**2 - (x[:,1]+20)**2 - 400

def f4(x):
    return x[:,1] - 20*x[:,0]**2 - 5*x[:,0] + 3

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
# Se visualizan las funciones y los puntos etiquetados por 'y_ruido' utilizando la función plot_datos_cuad.
print("Se muestra gŕafica de la primera función...")
    
plot_datos_cuad(x, y_ruido, f1, "f(x,y) = (x-10)^2 + (y-20)^2 - 400", "Eje x", "Eje y")
input("--- Pulsar tecla para visualizar la segunda función ---")
plot_datos_cuad(x, y_ruido, f2, "f(x,y) = 0.5*(x+10)^2 + (y-20)^2 - 400", "Eje x", "Eje y")
input("--- Pulsar tecla para visualizar la tercera función ---")
plot_datos_cuad(x, y_ruido, f3, "f(x,y) = 0.5*(x-10)^2 - (y+20)^2 - 400", "Eje x", "Eje y")
input("--- Pulsar tecla para visualizar la cuarta función ---")
plot_datos_cuad(x, y_ruido, f4, "f(x,y) = y - 20*x^2 - 5*x + 3", "Eje x", "Eje y")

input("--- Pulsar tecla para continuar al ejercicio 2 ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON
print("\n-Ejercicio 2-\n")

# 2.a. - Implementar la función que calcula el hiperpalo solución a un problema de clasificación
# binaria usando el algoritmo PLA.

# Perceptron Learning Algorithm (PLA).
# Como entrada tenemos el conjunto de datos 'datos' con sus respectivas características,
# el conjunto de etiquetas 'y' para cada dato, el número máximo de iteraciones 'max_iter' 
# y el vector de pesos inicial 'vini'.
def ajusta_PLA(datos, label, max_iter, vini):
    w = vini # Inicializamos w a vini.
    it = 0 # Inicializamos el número de iteraciones a 0 en una variable aparte para luego poder devolverla.
    for it in range(0, max_iter): # Mientras no superemos el máximo de iteraciones...
        w_ant = w # Guardamos el w previo a recorrer los datos.
        for i in range(0, datos.shape[0]): # Recorremos los datos.
            if signo(np.matmul(w, datos[i])) != label[i]: # Por cada dato donde la estimación dada por la w
                                                          # no coincide con la etiqueta...
                w = w + label[i]*datos[i] # Actualizamos w dada la ecuación w_new = w_old + y_i*x_i
        if (w_ant == w).all(): # Si después de recorrer los datos no ha habido cambio, para el algoritmo.
            break
    return w, it

# 2.a.1. - Ejecutar el algoritmo PLA con los datos simulados en el apartado 1.2.a.

x = np.hstack((np.ones((x.shape[0], 1)), x)) # Se añade una columna de unos a la muestra de datos 'x'.
print("-2.a.1-\n")
w, it = ajusta_PLA(x, y, 1000, np.zeros(x.shape[1])) # Y se ejecuta PLA para esos datos, etiquetas sin ruido, vector inicial 0
                                                     # y 1000 iteraciones para asegurarnos que encuentra el óptimo si puede.

# Para visualizar los resultados en una gráfica...
w_x = np.linspace(-50, 50) # Obtenemos dos puntos equidistantes en el intervalo en el que se generaron los datos.
w_y = -(w[0]+w[1]*w_x)/w[2] # Despejamos x2 de la ecuación w0 + w1*x1 + w2*x2 = 0 para obtener las coordenadas en el eje y.
plt.scatter(x[np.where(y == 1), 1], x[np.where(y == 1), 2], c="purple")
plt.scatter(x[np.where(y == -1), 1], x[np.where(y == -1), 2], c="orange")
plt.plot(w_x, w_y, c="red")
plt.legend(("PLA","+1","-1"))
plt.title("Ejercicio 2.a.1. Vector 0")
plt.show()

print("a) el vector 0")
print("Valor de iteraciones necesario para converger: ", it)

print("\nb) vectores de números aleatorios en [0,1] (10 veces)")
# Se ejecuta PLA para vectores iniciales aleatorios en [0,1] 10 veces y se guarda el número de iteraciones para
# luego obtener el promedio.
iterations = []
for i in range(0,10):
    w, it = ajusta_PLA(x, y, 1000, np.random.rand(x.shape[1]))
    print("Valor de iteraciones necesario para converger: ", it)
    iterations.append(it)
    
print("Valor medio de iteraciones necesario para converger: {}".format(np.mean(np.asarray(iterations))))

input("--- Pulsar tecla para continuar al ejercicio 2.a.2 ---\n")

# 2.a.2. - Ejecutar el algoritmo PLA con los datos simulados en el aparado 1.2.b.

w, it = ajusta_PLA(x, y_ruido, 1000, np.zeros(x.shape[1])) # Se ejecuta PLA para las etiquetas con ruido y vector inicial 0.

# Se muestran los resultados en una gráfica.
w_x = np.linspace(-50, 50)
w_y = -(w[0]+w[1]*w_x)/w[2]
plt.scatter(x[np.where(y_ruido == 1), 1], x[np.where(y_ruido == 1), 2], c="purple")
plt.scatter(x[np.where(y_ruido == -1), 1], x[np.where(y_ruido == -1), 2], c="orange")
plt.plot(w_x, w_y, c="red")
plt.legend(("PLA","+1","-1"))
plt.title("Ejercicio 2.a.2. Vector 0")
plt.show()

print("a) el vector 0")
print("Valor de iteraciones necesario para converger: ", it)

print("\nb) vectores de números aleatorios en [0,1] (10 veces)")
# Se ejecuta PLA para vectores iniciales aleatorios en [0,1] 10 veces
iterations = []
for i in range(0,10):
    w, it = ajusta_PLA(x, y_ruido, 1000, np.random.rand(x.shape[1]))
    print("Valor de iteraciones necesario para converger: ", it)
    iterations.append(it)
    
print("Valor medio de iteraciones necesario para converger: {}".format(np.mean(np.asarray(iterations))))

input("--- Pulsar tecla para continuar al ejercicio 2.b. ---\n")

###############################################################################
###############################################################################
###############################################################################

# 2.b. - Regresión logística con Gradiente Descendente Estocástico

from sklearn.utils import shuffle # Para barajar los minibatches en SGD.
from scipy.special import expit # Función sigmoide para el error en regresión logística.

# Función de error para Regresión Logística. Ein = 1/N*Σ(ln(1 + e^(-yi*w*xi)))
def Err(x, y, w):
    return np.log(1+np.exp(-y * np.dot(w, x.T))).mean()

# Función gradiente del error de Regresión Logística. ▽Ein(w) = 1/N*Σ(-yi*xi*σ(-yi*w*xi))
def gradErr(x, y, w):
    return np.dot(np.dot(-y, x), expit(np.dot(-y, np.dot(w, x.T))))

# Gradiente Descendente Estocástico (SGD) para Regresión Logística.
# Como argumentos de entrada tenemos el punto de inicio para empezar a minimizar,
# los datos x e y, la tasa de aprendizaje η (eta), el número máximo de iteraciones y
# el tamaño de cada minibatch.
def sgdRL(initial_point, x, y, eta, minibatch_size):
    w = initial_point # Asignamos el punto de inicio a w.
    epocas = 0 # Inicializamos el numero de épocas a 0.
    
    parar = False 
    while not parar: # Mientras no parar...
        # Barajamos x e y en el mismo orden usando 'shuffle' de sklearn y dividmos
        # el conjunto de datos en minibatches.
        x,y = shuffle(x, y, random_state=seed)
        minibatches_x = np.array_split(x, len(x)//minibatch_size)
        minibatches_y = np.array_split(y, len(y)//minibatch_size)
        
        w_ant = w # Guardamos el w previo a recorrer los minibatches.
        for i in range(0, len(minibatches_x)): # Para cada minibatch...
            w = w - eta*gradErr(minibatches_x[i], minibatches_y[i], w) # Obtenemos el nuevo w dado wj = wj-η*dEin(w)/dwj .
        parar = np.linalg.norm(w_ant - w) < 0.01 # Comprobamos ||w(t-1) - w(t)|| < 0.01, en cuyo caso pararía el algoritmo.
        epocas += 1 # Y se suma una época.
    return w, epocas # Se devuelve el w resultante y el número de épocas total.

# Consideramos d = 2, χ = [0,2]x[0,2] con probabilidad uniforme de elegir cada x ∈ χ.
# Seleccionar 100 puntos aleatorios χ.
x_train = simula_unif(100, 2, [0,2])
# Elegir una línea en el plano que pase por χ como la frontera entre f(x) = 1 (donde y toma valores +1) y
# f(x) = 0 (donde y toma valores -1), para ello seleccionar dos puntos aleatorios de χ y calcular la línea que pasa por ambos.
a, b = simula_recta([0,2])
# Etiquetar χ respecto a la frontera elegida.
y_train = np.array([f(x, y, a, b) for x, y in x_train], dtype=np.float64)
# Estimar la función para la muestra de entrenamiento usando SGD.
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
w, epocas = sgdRL(np.zeros(x_train.shape[1]), x_train, y_train, 0.01, 1)

# Evaluar el error de salida Eout para una muestra grande de datos (> 999).
x_test = simula_unif(1000, 2, [0,2])
y_test = np.array([f(x, y, a, b) for x, y in x_test], dtype=np.float64)
x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

# Se visualizan los resultados obtenidos para el primer experimento para la
# muestra de entrenamiento, incluyendo gráfica, error de entrada Ein y número de épocas.
print("-Primer experimento-\n")
print("Épocas: ", epocas)
print("Ein: ", Err(x_train, y_train, w))

recta_x = np.linspace(0,2,2)
recta_y = a*recta_x + b
sgd_y = -(w[0]+w[1]*recta_x)/w[2]
plt.scatter(x_train[np.where(y_train == 1), 1], x_train[np.where(y_train == 1), 2], c="purple")
plt.scatter(x_train[np.where(y_train == -1), 1], x_train[np.where(y_train == -1), 2], c="orange")
plt.plot(recta_x, recta_y, c="red")
plt.plot(recta_x, sgd_y, c="blue")
plt.legend(("Frontera","SGD","+1","-1"), loc="upper left")
plt.ylim(0,2)
plt.title("2.b. Primer experimento / Muestra de entrenamiento")
plt.show()

# Se visualizan los resultados obtenidos para el primer experimento para la
# muestra de test, incluyendo gráfica, error de salida Eout y número de épocas.
input("--- Pulsar tecla para visualizar la muestra de prueba en el primer experimento ---\n")
print("Eout: ", Err(x_test, y_test, w))

sgd_y = -(w[0]+w[1]*recta_x)/w[2]
plt.scatter(x_test[np.where(y_test == 1), 1], x_test[np.where(y_test == 1), 2], c="purple")
plt.scatter(x_test[np.where(y_test == -1), 1], x_test[np.where(y_test == -1), 2], c="orange")
plt.plot(recta_x, recta_y, c="red")
plt.plot(recta_x, sgd_y, c="blue")
plt.legend(("Frontera","SGD","+1","-1"), loc="upper left")
plt.ylim(0,2)
plt.title("2.b. Primer experimento / Muestra de prueba")
plt.show()

input("--- Pulsar tecla para continuar con los 100 experimentos ---\n")

# Repetir el experimento 100 veces.
n = 100
Ein_promedio = 0
Eout_promedio = 0
Epocas_promedio = 0
for i in range(0, n):
    # Generar muestra de entrenamiento, calcular w y evaluar Ein.
    x_train = simula_unif(100, 2, [0,2])
    y_train = np.array([f(x, y, a, b) for x, y in x_train], dtype=np.float64)
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    w, epocas = sgdRL(np.zeros(x_train.shape[1]), x_train, y_train, 0.01, 1)
    Ein_promedio += Err(x_train, y_train, w)
    Epocas_promedio += epocas
    
    # Generar muestra de prueba y evaluar Eout.
    x_test = simula_unif(1000, 2, [0,2])
    y_test = np.array([f(x, y, a, b) for x, y in x_test], dtype=np.float64)
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
    Eout_promedio += Err(x_test, y_test, w)

# Se imprime por pantalla los resultados promedios de repetir el experimento 100 veces.
print("Número de épocas promedio: ", Epocas_promedio/n)
print("Ein promedio: ", Ein_promedio/n)
print("Eout promedio: ", Eout_promedio/n)

input("--- Pulsar tecla para continuar al ejercicio 3 ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3. BONUS: Clasificación de dígitos.
print("\n-Ejercicio 3-\n")

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


# EJERCICIO 3.2: Usar Regresión Lineal y aplicar PLA-Pocket como mejora.

#LINEAR REGRESSION FOR CLASSIFICATION 

# Pseudoinversa. w = X†*y
# Para ello simplemente utilizamos matmul para multiplicar la pseudoinversa dada por
# la función pinv de numpy con y.
def pseudoinverse(x, y):
    return np.matmul(np.linalg.pinv(x), y)

#POCKET ALGORITHM

# Error de clasificación para Pocket. 1/N*Σ(sign(w*xn) ≠ yn)
# El sumatorio de todas las etiquetas estimadas que no coinciden con las etiquetas de 'y'.
def ErrorPLA(datos, label, w):
    total = 0
    for i in range(0, datos.shape[0]):
        if signo(np.matmul(w, datos[i])) != label[i]:
            total += 1
    return total/datos.shape[0]

# Algoritmo Pocket.
# Como entrada tenemos el conjunto de datos 'datos' con sus respectivas características,
# el conjunto de etiquetas 'y' para cada dato, el número máximo de iteraciones 'max_iter'
# y el vector inicial 'vini'.
def PLA_pocket(datos, label, max_iter, vini):
    w = vini # Inicializamos w al vector inicial.
    error_min = ErrorPLA(datos, label, w) # Calculamos y guardamos el error inicial.
    for it in range(0, max_iter): # Mientras no superemos el máximo de iteraciones...
        w1, one_update = ajusta_PLA(datos, label, 1, w) # Ejecutamos PLA una sola época para obtener w(t+1).
        error = ErrorPLA(datos, label, w1) # Calculamos el error de clasificación de w(t+1).
        if error_min > error: # Si el error de w(t) es mayor que el de w(t+1).
            w = w1 # Actualizamos w.
            error_min = error # Guardamos el nuevo error mínimo.
    return w # Devolvemos el w final.

# 3.2.a. y 3.2.b - Generar gráficos separados de los datos de entrenamiento y test junt con la función
# estimada y calcular los Ein y Etest.
w_pinv = pseudoinverse(x, y) # Obtenemos w usando la pseudoinversa para la muestra de entrenamiento.
w_pocket = PLA_pocket(x, y, 1000, w_pinv) # Obtenemos w usando Pocket para la misma muestra, con 1000 iteraciones y
                                          # usando la estimación de la pseudoinversa como vector inicial.

# Se calculan los puntos para cada recta.
recta_x = np.linspace(0, 1, 2)
pinv_y = -(w_pinv[0]+w_pinv[1]*recta_x)/w_pinv[2]
pocket_y = -(w_pocket[0]+w_pocket[1]*recta_x)/w_pocket[2]

# Se imprime el error de entrada para la pseudoinversa.
print("-PSEUDOINVERSA-")
print("Ein: ", ErrorPLA(x, y, w_pinv))

# Mostramos las estimaciones obtenidas junto a los datos de entrenamiento para la pseudoinversa.
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.plot(recta_x, pinv_y, color='purple', label='Pinv')
ax.set(xlabel='Intensidad promedio', ylabel='Simetría', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
ax.set_ylim((-7, -1))
plt.legend()
plt.show()

input("--- Pulsar tecla para mostrar los resultados de Pocket en la muestra de entrenamiento ---\n")

# Se imprime el error de entrenamiento para Pocket.
print("-POCKET-")
print("Ein: ", ErrorPLA(x, y, w_pocket))

# Mostramos las estimaciones obtenidas junto a los datos de entrenamiento para Pocket.
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.plot(recta_x, pocket_y, color='orange', label='Pocket')
ax.set(xlabel='Intensidad promedio', ylabel='Simetría', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
ax.set_ylim((-7, -1))
plt.legend()
plt.show()

input("--- Pulsar tecla para continuar con los resultados en la muestra de test ---\n")

# Se imprime el error de test para la pseudoinversa.
print("-PSEUDOINVERSA-")
print("Etest: ", ErrorPLA(x_test, y_test, w_pinv))

# Mostramos las estimaciones obtenidas junto a los datos de prueba para la pseudoinversa.
fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.plot(recta_x, pinv_y, color='purple', label='Pinv')
ax.set(xlabel='Intensidad promedio', ylabel='Simetría', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
ax.set_ylim((-7, -1))
plt.legend()
plt.show()

input("--- Pulsar tecla para mostrar los resultados de Pocket en la muestra de test ---\n")

# Se imprime el error de test para Pocket.
print("-POCKET-")
print("Etest: ", ErrorPLA(x_test, y_test, w_pocket))

# Mostramos las estimaciones obtenidas junto a los datos de prueba para Pocket.
fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.plot(recta_x, pocket_y, color='orange', label='Pocket')
ax.set(xlabel='Intensidad promedio', ylabel='Simetría', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
ax.set_ylim((-7, -1))
plt.legend()
plt.show()

input("--- Pulsar tecla para continuar al ejercicio 3.2.c ---\n")
  
# 3.2.c. - Obtener cotas sobre el verdadero valor de Eout para Ein y Etest.

# Cota basada en Ein.
# Eout(h) ≤ Ein(h) + √(1/2N * log(2/delta))
delta = 0.05
cota_sup = ErrorPLA(x, y, w_pocket) + np.sqrt((1/(2*x.shape[0]))*np.log(2/delta))
print("Eout será menor o igual que", cota_sup, "(cota basada en Ein)")

# Cota basada en Etest.
# Eout(h) ≤ Etest(h) + √(1/2N * log(2/delta))
cota_sup = ErrorPLA(x_test, y_test, w_pocket) + np.sqrt((1/(2*x.shape[0]))*np.log(2/delta))
print("Eout será menor o igual que", cota_sup, "(cota basada en Etest)")










