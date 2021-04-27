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
print("-1.1.-\n\nSe muestra gráfica...")

# 1.1.a. - Dibujar la gráfica para N=50, d=2 y rango=[-50,50] para una distribución uniforme.
x = simula_unif(50, 2, [-50,50])
plt.scatter(x[:,0], x[:,1])
plt.title("Ejercicio 1.1.a.")
plt.show()

input("--- Pulsar tecla para continuar al ejercicio 1.1.b ---\n")
print("Se muestra gráfica...")

# 1.1.b. - Dibujar la gráfica para N=50, d=2 y sigma=[5,7] para una distribución normal.
x = simula_gaus(50, 2, np.array([5,7]))
plt.scatter(x[:,0], x[:,1])
plt.title("Ejercicio 1.1.b.")
plt.show()

input("--- Pulsar tecla para continuar al ejercicio 1.2 ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente
print("\n-1.2-\n")

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

# 1.2.a. - Dibujar un gráfico 2D donde los puntos muestren el resultado de su etiqueta.
# Dibuje también la recta usada para etiquetar.

print("Se muestra gráfica...")

x = simula_unif(100, 2, [-50,50]);
a, b = simula_recta([-50,50]);
y = np.array([f(x, y, a, b) for x, y in x], dtype=np.float64)
recta_x = np.linspace(-50, 50, 2)
recta_y = a*recta_x + b

plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], c="purple")
plt.scatter(x[np.where(y == -1), 0], x[np.where(y == -1), 1], c="orange")
plt.legend(("+1","-1"))
plt.plot(recta_x, recta_y, c="red")
plt.title("Ejercicio 1.2.a.")
plt.show()

input("--- Pulsar tecla para continuar al ejercicio 1.2.b ---\n")

# 1.2.b. - Modifique de forma aleatoria el 10% de las etiquetas positivas y otro 10% de las
# negativas y guarde los puntos con sus nuevas etiquetas. Dibuje de nuevo la gráfica anterior.

def generar_ruido(y):
    y_positivo = np.where(y == 1)[0]
    y_negativo = np.where(y == -1)[0]
    index_positivo = np.random.choice(y_positivo, y_positivo.size//10, replace=False)
    index_negativo = np.random.choice(y_negativo, y_negativo.size//10, replace=False)
    y[index_positivo] *= -1
    y[index_negativo] *= 1
    
print("Se muestra gráfica...")
y_ruido = y.copy()
generar_ruido(y_ruido)
plt.scatter(x[np.where(y_ruido == 1), 0], x[np.where(y_ruido == 1), 1], c="purple")
plt.scatter(x[np.where(y_ruido == -1), 0], x[np.where(y_ruido == -1), 1], c="orange")
plt.legend(("+1","-1"))
plt.plot(recta_x, recta_y, c="red")
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
def ajusta_PLA(datos, label, max_iter, vini):
    w = vini
    it = 0
    for it in range(0, max_iter):
        w_ant = w
        for i in range(0, datos.shape[0]):
            if signo(np.matmul(w, datos[i])) != label[i]:
                w = w + label[i]*datos[i]
        if (w_ant == w).all():
            break
    return w, it

# 2.a.1. - Ejecutar el algoritmo PLA con los datos simulados en el aparado 1.2.a.

x = np.hstack((np.ones((x.shape[0], 1)), x))
print("-2.a.1-\n")
w, it = ajusta_PLA(x, y, 1000, np.zeros(x.shape[1]))

w_x = np.linspace(-50, 50)
w_y = -(w[0]+w[1]*w_x)/w[2]
plt.scatter(x[np.where(y == 1), 1], x[np.where(y == 1), 2], c="purple")
plt.scatter(x[np.where(y == -1), 1], x[np.where(y == -1), 2], c="orange")
plt.plot(w_x, w_y, c="red")
plt.legend(("PLA","+1","-1"))
plt.title("Ejercicio 2.a.1. Vector 0")
plt.show()

print("a) el vector 0")
print("Valor de iteraciones necesario para converger: ", it)

print("\nb) vectores de números aleatorios en [0,1] (10 veces)")
# Random initializations
iterations = []
for i in range(0,10):
    w, it = ajusta_PLA(x, y, 1000, np.random.rand(x.shape[1]))
    print("Valor de iteraciones necesario para converger: ", it)
    iterations.append(it)
    
print("Valor medio de iteraciones necesario para converger: {}".format(np.mean(np.asarray(iterations))))

input("--- Pulsar tecla para continuar al ejercicio 2.a.2 ---\n")

# 2.a.2. - Ejecutar el algoritmo PLA con los datos simulados en el aparado 1.2.b.

w, it = ajusta_PLA(x, y_ruido, 1000, np.zeros(x.shape[1]))

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
# Random initializations
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

from sklearn.utils import shuffle
from scipy.special import expit

def Err(x, y, w):
    return np.log(1+np.exp(np.matmul(np.matmul(-y, w), x))).mean()

def gradErr(x, y, w):
    return np.matmul(np.matmul(-y, w), expit(np.matmul(np.matmul(-y, w), x)))

# Gradiente Descendente Estocástico (SGD) para Regresión Logística.
def sgdRL(initial_point, x, y, eta, maxIter, minibatch_size):
    w = initial_point
    iterations = 0 
    
    parar = False 
    while not parar and iterations < maxIter:
        x,y = shuffle(x, y, random_state=seed)
        minibatches_x = np.array_split(x, len(x)//minibatch_size)
        minibatches_y = np.array_split(y, len(y)//minibatch_size)
        
        for i in range(0, len(minibatches_x)):
            w_ant = w
            w = w - eta*gradErr(minibatches_x[i], minibatches_y[i], w) 
            parar = np.linalg.norm(w_ant - w) < 0.01
            if parar:
                break
        iterations += 1
    return w # Devolvemos el w final.

# Consideramos d = 2, χ = [0,2]x[0,2] con probabilidad uniforme de elegir cada x ∈ χ.
# Seleccionar 100 puntos aleatorios χ.
x = simula_unif(100, 2, [0,2])
# Elegir una línea en el plano que pase por χ como la frontera entre f(x) = 1 (donde y toma valores +1) y
# f(x) = 0 (donde y toma valores -1), para ello seleccionar dos puntos aleatorios de χ y calcular la línea que pasa por ambos.
a, b = simula_recta([0,2])
# Etiquetar χ respecto a la frontera elegida.
y = np.array([f(x, y, a, b) for x, y in x], dtype=np.float64)

print("Se muestra gŕafica...")
recta_x = np.linspace(0, 2, 2)
recta_y = a*recta_x + b
plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], c="purple")
plt.scatter(x[np.where(y == -1), 0], x[np.where(y == -1), 1], c="orange")
plt.legend(("+1","-1"))
plt.plot(recta_x, recta_y, c="red")
plt.ylim(0,2)
plt.title("2.b. Muestra de entrenamiento")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
# x = np.hstack((np.ones((x.shape[0], 1)), x))
# sgd

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

# EJERCICIO 3. BONUS: Clasificación de dígitos

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


# Mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetría', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("--- Pulsar tecla para visualizar la muestra de prueba ---")

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetría', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
