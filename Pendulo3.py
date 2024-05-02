import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import copy
from scipy.signal import convolve

from funciones import H
from funciones import posicion
from funciones import posicion_k
from funciones import H_k
from scipy.optimize import least_squares

import Algoritmos as alg
import Graficador as graf
import funciones as f

import time
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


############################## GENERO ANGULOS ###############################################

############ THETA 1 ############
# Parámetros
a = 1  # Amplitud
b = 0  # Desplazamiento vertical
omega = 1*np.pi  # Frecuencia angular (en este caso, una vuelta completa cada unidad de tiempo)
# Vector de tiempo
t = np.linspace(0, 4, 400)  # Desde 0 hasta 10 con 400 puntos
phi = np.pi / 4  # Desplazamiento en omega t (en radianes)
# Función de ángulo
theta1 = a * np.cos(omega * t + phi) + b
theta1_copia=copy.copy(theta1)

############ THETA 2 ############
# Parámetros
a = 1 # Amplitud
b = 0  # Desplazamiento vertical
omega = 1*np.pi  # Frecuencia angular (en este caso, una vuelta completa cada unidad de tiempo)
# Vector de tiempo
t = np.linspace(0, 4, 400)  # Desde 0 hasta 10 con 400 puntos
# Función de ángulo
phi = np.pi / 2  # Desplazamiento en omega t (en radianes)
theta2 = a * a * np.cos(omega * t + phi) + b
theta2_copia=copy.copy(theta2)

############ THETA 3 ############
# Parámetros
a = 1  # Amplitud
b = 0 # Desplazamiento vertical
omega = 1*np.pi  # Frecuencia angular (en este caso, una vuelta completa cada unidad de tiempo)
# Vector de tiempo
t = np.linspace(0, 4, 400)  # Desde 0 hasta 10 con 400 puntos
# Función de ángulo
phi = np.pi / 4  # Desplazamiento en omega t (en radianes)
theta3 = a * np.cos(omega * t + phi) + b
theta3_copia=copy.copy(theta3)

################################## DEFINICION DE VALORES #############################

l_pendulo = 1
#distancia entre pendulo y marcador
l_m=0.05

############################## CONFIGURACIONES DE LA GRAFICA #############################

# Crear la figura y los ejes
fig, ax = plt.subplots()
ax.invert_yaxis()  # Invertir el eje y para que los valores positivos apunten hacia abajo

# Cambiar la posición del origen del gráfico a (0, 0)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

# Definir límites de los ejes
ax.set_xlim(-1, 3)  # Límites del eje x
ax.set_ylim(-1, 1)  # Límites del eje y invertido

ax.set_yticks([-0.25, -0.5, -0.75, -1])  # Definir las ubicaciones de las marcas en el eje x
ax.set_yticklabels(['-0.25', '-0.5', '-0.75', '-1'])  # Etiquetas correspondientes a las ubicaciones


ax.set_aspect('equal')
ax.grid()

fps = 100
frames = np.arange(0, 400, 1)  # Inicio, fin (no incluido), paso


# Convertir frames a tiempo en segundos
tiempo = [frame / fps for frame in frames]

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###################### FUNCIONES #############################################################


def complex_step_derivative(f, x, h=1e-10):
    """
    Calcula la derivada de una función f en un punto x utilizando el método de paso complejo.
    
    Args:
    - f: La función para la cual se calculará la derivada.
    - x: El punto en el cual se calculará la derivada.
    - h: El tamaño del paso complejo.
    
    Returns:
    - La derivada de f en x.
    """
    # Evaluamos la función en el punto x + h*i (parte imaginaria)
    f_complex = f(x + 1j * h)
    # Obtenemos la parte imaginaria de la función evaluada
    derivative_imag = np.imag(f_complex)
    # Calculamos la derivada dividiendo por h
    derivative = derivative_imag / h
    return derivative

# Ejemplo de uso
def f(x):
    return x**2 + 2*x + 1

x = 2.0
derivative = complex_step_derivative(f, x)
print("La derivada de f en x =", x, "es:", derivative)


def leer_txt(nombre):
    
    with open(nombre, 'r') as archivo:
        # Lee las líneas del archivo
        lineas = archivo.readlines()
        x_y = []
        posiciones=[]

        for linea in lineas:
            
            linea = linea.strip('[]')
            linea = linea.replace("]", "")
            elementos = linea.split()
            valores_float = [float(elemento) for elemento in elementos]
            posiciones.extend(valores_float)
            
            if (len(posiciones)==400):
                x_y.append(posiciones[:])
                posiciones.clear()
                
    
    return x_y[0:5], x_y[5:10]
                

        

def escribir_txt(nombre, posiciones_x, posiciones_y):
    
    with open(nombre +'.txt', 'w') as archivo:
        
        for x in posiciones_x:
            archivo.write(str(x))
            archivo.write("\n")
        
        archivo.write("\n")
        
        for y in posiciones_y:
            archivo.write(str(y))
            archivo.write("\n")
            
            


def graficar_posicion(t,posicion):
        
    fig, axs = plt.subplots(5, 1, figsize=(6, 12))  # 5 filas, 1 columna

    # Graficar los datos en cada subplot
    for i, ax in enumerate(axs):
        ax.plot(t, posicion[i])
        ax.set_title(f'Fila {i+1}')
    
    # Ajustar el espaciado entre los subplots
    plt.tight_layout()


def filtro_media_movil(datos, ventana):
    # Crear el kernel del filtro de media móvil
    kernel = np.ones(ventana) / ventana
    
    # Aplicar el filtro de media móvil
    datos_filtrados = convolve(datos, kernel, mode='same')
    
    return datos_filtrados

def actualizar_pendulo(theta, theta2, theta3, posiciones_x, posiciones_y, i):
    
     
    ######################################### PENDULO 1 ######################################
    r0_1=[0,0]
    A=np.array([[np.sin((theta)), np.cos((theta))], [-np.cos((theta)), np.sin((theta))]])     
    rp_l=np.array([l_pendulo,0])
    rp_1=np.dot(A, rp_l)
    pendulo1.set_data([0, r0_1[0]+rp_1[0]],[0, r0_1[1]+ rp_1[1]])
    
    # M1 
    
    rp_l_m1_base=np.array([0.3,0])
    rp_l_m1_punta=np.array([0.3,-l_m])
    
    rp_base=np.dot(A, rp_l_m1_base)
    rp_punta=np.dot(A, rp_l_m1_punta)
    
    marcador1.set_data((r0_1[0]+rp_base[0], r0_1[0]+rp_punta[0]), (r0_1[0]+rp_base[1], r0_1[1]+rp_punta[1]))
    punto_final_m1.set_data(r0_1[0]+rp_punta[0], r0_1[1]+rp_punta[1] )
    
    posiciones_x[0][i]=r0_1[0]+rp_punta[0]
    posiciones_y[0][i]=r0_1[1]+rp_punta[1]

    # M2
    
    rp_l_m2_base=np.array([0.7,0])
    rp_l_m2_punta=np.array([0.7,-l_m])
                           
    rp_base=np.dot(A, rp_l_m2_base)
    rp_punta=np.dot(A, rp_l_m2_punta)
                           
    marcador2.set_data((r0_1[0]+rp_base[0], r0_1[0]+rp_punta[0]), (r0_1[1]+rp_base[1], r0_1[1]+rp_punta[1]))
    punto_final_m2.set_data(r0_1[0]+rp_punta[0], r0_1[1]+rp_punta[1] )

    posiciones_x[1][i]=r0_1[0]+rp_punta[0]
    posiciones_y[1][i]=r0_1[1]+rp_punta[1]
    
    ######################################### PENDULO 2 ######################################
    r0_2=[np.sin((theta)),-np.cos((theta))]
    B=np.array([[-np.sin((theta2)), np.cos((theta2))], [np.cos((theta2)), np.sin((theta2))]])     
    rp_l=np.array([0,1])
    rp_2=np.dot(np.dot(B,A), rp_l)
    pendulo2.set_data([r0_2[0],r0_2[0]+rp_2[0]],[r0_2[1],r0_2[1]+rp_2[1]])
    
    # M3 
    
    rp_l_m3_base=np.array([0,0.5])
    rp_l_m3_punta=np.array([-l_m,0.5])
                           
    rp_base=np.dot(np.dot(B,A), rp_l_m3_base)
    rp_punta=np.dot(np.dot(B,A), rp_l_m3_punta)

    marcador3.set_data((r0_2[0]+rp_base[0], r0_2[0]+rp_punta[0]), (r0_2[1]+rp_base[1], r0_2[1]+rp_punta[1]))
    punto_final_m3.set_data(r0_2[0]+rp_punta[0],r0_2[1]+rp_punta[1] )
    
    posiciones_x[2][i]=r0_2[0]+rp_punta[0]
    posiciones_y[2][i]=r0_2[1]+rp_punta[1]
        
    # M4
    
    rp_l_m4_base=np.array([0,0.9])
    rp_l_m4_punta=np.array([-l_m,0.9])
                           
    rp_base=np.dot(np.dot(B,A), rp_l_m4_base)
    rp_punta=np.dot(np.dot(B,A), rp_l_m4_punta)
                           
    marcador4.set_data((r0_2[0]+rp_base[0], r0_2[0]+rp_punta[0]), (r0_2[1]+rp_base[1], r0_2[1]+rp_punta[1]))
    punto_final_m4.set_data(r0_2[0]+rp_punta[0], r0_2[1]+rp_punta[1] )
    
    posiciones_x[3][i]=r0_2[0]+rp_punta[0]
    posiciones_y[3][i]=r0_2[1]+rp_punta[1]
    
    # ############ PENDULO 3 ##############
    r0_3=[r0_2[0]+rp_2[0],r0_2[1]+rp_2[1]]
    C=np.array([[-np.sin((theta3)), np.cos((theta3))], [np.cos((theta3)), np.sin((theta3))]])     
    rp_l=np.array([1,0])
    rp_3=np.dot(np.dot(np.dot(C,B),A), rp_l)
    pendulo3.set_data([r0_3[0],r0_3[0]+rp_3[0]],[r0_3[1],r0_3[1]+rp_3[1]])
    
    #M5
    
    rp_l_m5_base=np.array([0.8,0])
    rp_l_m5_punta=np.array([0.8,-l_m])
    
    rp_base=np.dot(np.dot(np.dot(C,B),A), rp_l_m5_base)
    rp_punta=np.dot(np.dot(np.dot(C,B),A), rp_l_m5_punta)
    
    punto_final_m5.set_data(r0_3[0]+rp_punta[0], r0_3[1]++rp_punta[1])
    marcador5.set_data((r0_3[0]+rp_base[0], r0_3[0]+rp_punta[0]), (r0_3[1]+rp_base[1], r0_3[1]+rp_punta[1]))
    
    posiciones_x[4][i]=r0_3[0]+rp_punta[0]
    posiciones_y[4][i]=r0_3[1]+rp_punta[1]
    
################ GRAFICOS INICIALES EN COORDENADAS GLOBALES ##########################

#Pendulos
pendulo1, = ax.plot([], [], 'b-')  # Graficar el vector inicial
pendulo2, = ax.plot([], [], 'b-')  # Graficar el vector inicial
pendulo3, = ax.plot([], [], 'b-')  # Graficar el vector inicial


#Marcador 1
marcador1, = ax.plot([], [], 'b-')  # Graficar el vector inicial
punto_final_m1, =ax.plot([], 'ro') 
#Marcador 2
marcador2, = ax.plot([], [], 'b-')  # Graficar el vector inicial
punto_final_m2, =ax.plot([], 'ro') 
#Marcador 3
marcador3, = ax.plot([], [], 'b-')  # Graficar el vector inicial
punto_final_m3, =ax.plot([], 'ro') 
#Marcador 4
marcador4, = ax.plot([], [], 'b-')  # Graficar el vector inicial
punto_final_m4, =ax.plot([], 'ro') 
#Marcador 5
marcador5, = ax.plot([], [], 'b-')  # Graficar el vector inicial
punto_final_m5, =ax.plot([], 'ro') 

###########################################################################################

posiciones_x = np.zeros((5, 400))
posiciones_y = np.zeros((5, 400))

i=0
for theta_1, theta_2, theta_3 in zip(theta1, theta2, theta3):
    actualizar_pendulo(theta_1,theta_2, theta_3, posiciones_x, posiciones_y,i) # Actualizar el vector
    i=i+1
    plt.pause(0.001)
    

plt.show()

################################################# PARTE 2 #############################################


# Generar ruido aleatorio distribuido normalmente
ruido_posicion_x_camara = np.random.normal(loc=0, scale=0.00001, size=(5, 400))
ruido_posicion_y_camara = np.random.normal(loc=0, scale=0.00001, size=(5, 400))

ruido_posicion_x_piel = np.random.normal(loc=0, scale=0.01, size=(5, 400))
ruido_posicion_y_piel = np.random.normal(loc=0, scale=0.01, size=(5, 400))


i = 0
for ruido_x, ruido_y in zip(ruido_posicion_x_piel, ruido_posicion_y_piel):
    ruido_posicion_x_piel[i] = filtro_media_movil(ruido_x, ventana=50)
    ruido_posicion_y_piel[i] = filtro_media_movil(ruido_y, ventana=50)
    i += 1

ruido_x = ruido_posicion_x_camara + ruido_posicion_x_piel
ruido_y = ruido_posicion_y_camara + ruido_posicion_y_piel


x_con_ruido=ruido_x+posiciones_x
y_con_ruido=ruido_y+posiciones_y



##################################################################################################
theta1_p=np.diff(theta1_copia) / 10E-2
theta1_p = np.insert(theta1_p, 0, theta1_p[0])

theta2_p=np.diff(theta2_copia) / 10E-2
theta2_p = np.insert(theta2_p, 0, theta2_p[0])


theta3_p=np.diff(theta3_copia) / 10E-2
theta3_p = np.insert(theta3_p, 0, theta3_p[0])

####################################################################################################################

#ECUACIONES DE POSICION DEL MARCADOR

######################################### PENDULO 1 ######################################


from sympy import symbols, sin, cos, Matrix

θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3 = symbols('θ1 θ2 θ3 k1x k1y k2x k2y k3')


A = [[sin(θ1), cos(θ1)], 
     [-cos(θ1), sin(θ1)]]

B = [[-sin(θ2 ), cos(θ2)], 
     [cos(θ2), sin(θ2)]]

C = [[-sin(θ3), cos(θ3)], 
     [cos(θ3), sin(θ3)]]

######################################### PENDULO 1 ######################################

r0_1=[0,0]
rp_l=Matrix([k1x*l_pendulo,k1y*0])
rp_1=np.dot(A,rp_l)
r_final_pendulo1=Matrix([k1x*sin(θ1),-k1y*cos(θ1)])


# M1 
rp_l_m1_punta=Matrix([k1x*0.3,k1y*-l_m])
rp_punta=np.dot(A, rp_l_m1_punta)
m1_posicion=Matrix([r0_1[0]+rp_punta[0], r0_1[1]+rp_punta[1]])


# M2
rp_l_m2_punta=Matrix([k1x*0.7,k1y*(-l_m)])                       
rp_punta=np.dot(A, rp_l_m2_punta)
m2_posicion=Matrix([r0_1[0]+rp_punta[0], r0_1[1]+rp_punta[1]])


######################################### PENDULO 2 ######################################

rp_l=Matrix([k2x*0,k2y*l_pendulo])
rp_2=np.dot(np.dot(B,A), rp_l)
r_final_pendulo2=Matrix([r_final_pendulo1[0]+rp_2[0],r_final_pendulo1[1]+rp_2[1]])

# M3 
rp_l_m3_punta=Matrix([k2x*(-l_m),k2y*0.5])                    
rp_punta=np.dot(np.dot(B,A), rp_l_m3_punta)
m3_posicion=Matrix([r_final_pendulo1[0]+rp_punta[0],r_final_pendulo1[1]+rp_punta[1]])

# M4
rp_l_m4_punta=Matrix([k2x*(-l_m),k2y*0.9])                      
rp_punta=np.dot(np.dot(B,A), rp_l_m4_punta)
m4_posicion=Matrix([r_final_pendulo1[0]+rp_punta[0], r_final_pendulo1[1]+rp_punta[1]])

# ############ PENDULO 3 ##############   
rp_l=Matrix([k3*1,k3*0])
rp_3=np.dot(np.dot(np.dot(C,B),A), rp_l)

#M5
rp_l_m5_punta=Matrix([k3*0.8,k3*(-l_m)])
rp_punta=np.dot(np.dot(np.dot(C,B),A), rp_l_m5_punta)
m5_posicion=Matrix([r_final_pendulo2[0]+rp_punta[0], r_final_pendulo2[1]+rp_punta[1]])

matriz_posicion=Matrix([m1_posicion,m2_posicion,m3_posicion, m4_posicion, m5_posicion])


# h_k=np.array([
# [                                                                                                                                                                                                                                  0.3*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1)],
# [                                                                                                                                                                                                                                 -0.3*k1x*np.cos(θ1) - 0.05*k1y*np.sin(θ1)],
# [                                                                                                                                                                                                                                  0.7*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1)],
# [                                                                                                                                                                                                                                 -0.7*k1x*np.cos(θ1) - 0.05*k1y*np.sin(θ1)],
# [                                                                                                                                                           k1x*np.sin(θ1) - 0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1))],
# [                                                                                                                                                           -k1y*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2))],
# [                                                                                                                                                           k1x*np.sin(θ1) - 0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1))],
# [                                                                                                                                                           -k1y*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2))],
# [    k1x*np.sin(θ1) + k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.8*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1))],
# [-k1y*np.cos(θ1) + k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) - 0.05*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) + 0.8*k3*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1))]])


#################################################################################################


# h_k = Matrix([
        
#         #m1
#         [0.3*k1x*sin(θ1) - 0.05*k1y*cos(θ1)],
#         [-0.3*k1x*cos(θ1) - 0.05*k1y*sin(θ1)],
#         #m2
#         [0.7*k1x*sin(θ1) - 0.05*k1y*cos(θ1)],
#         [-0.7*k1x*cos(θ1) - 0.05*k1y*sin(θ1)],        
#         #m3
#         [k1x*sin(θ1) - 0.05*k2x*(-sin(θ1)*sin(θ2) - cos(θ1)*cos(θ2)) + 0.5*k2y*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1))],
#         [-k1y*cos(θ1) - 0.05*k2x*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1)) + 0.5*k2y*(sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2))],         
#         #m4
#         [k1x*sin(θ1) - 0.05*k2x*(-sin(θ1)*sin(θ2) - cos(θ1)*cos(θ2)) + 0.9*k2y*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1))],
#         [-k1y*cos(θ1) - 0.05*k2x*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1)) + 0.9*k2y*(sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2))],
#         #m5
#         [k1x*sin(θ1) + k2y*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1)) + 0.8*k3x*((sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*sin(θ1) - (sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*cos(θ1)) - 0.05*k3y*((sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*cos(θ1) + (sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*sin(θ1))],
#         [-k1y*cos(θ1) + k2y*(sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2)) + 0.8*k3x*(-(sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*cos(θ1) + (-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*sin(θ1)) - 0.05*k3y*((sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*sin(θ1) + (-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*cos(θ1))]
#         ])


# jacobian_matrix = matriz_posicion.jacobian([θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3])

# H_k = Matrix([
        
#         #m1
#         [0.3*k1x*cos(θ1) + 0.05*k1y*sin(θ1), 0, 0, 0.3*sin(θ1), -0.05*cos(θ1), 0, 0, 0, 0],
#         [0.3*k1x*sin(θ1) - 0.05*k1y*cos(θ1), 0, 0, -0.3*cos(θ1), -0.05*sin(θ1), 0, 0, 0, 0],
#         #m2
#         [0.7*k1x*cos(θ1) + 0.05*k1y*sin(θ1), 0, 0, 0.7*sin(θ1), -0.05*cos(θ1), 0, 0, 0, 0],
#         [0.7*k1x*sin(θ1) - 0.05*k1y*cos(θ1), 0, 0, -0.7*cos(θ1), -0.05*sin(θ1), 0, 0, 0, 0],        
#         #m3
#         [k1x*cos(θ1) - 0.05*k2x*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1)) + 0.5*k2y*(sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2)), -0.05*k2x*(-sin(θ1)*cos(θ2) + sin(θ2)*cos(θ1)) + 0.5*k2y*(-sin(θ1)*sin(θ2) - cos(θ1)*cos(θ2)), 0, sin(θ1), 0, 0.05*sin(θ1)*sin(θ2) + 0.05*cos(θ1)*cos(θ2), 0.5*sin(θ1)*cos(θ2) - 0.5*sin(θ2)*cos(θ1), 0, 0],
#         [k1y*sin(θ1) - 0.05*k2x*(sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2)) + 0.5*k2y*(-sin(θ1)*cos(θ2) + sin(θ2)*cos(θ1)), -0.05*k2x*(-sin(θ1)*sin(θ2) - cos(θ1)*cos(θ2)) + 0.5*k2y*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1)), 0, 0, -cos(θ1), -0.05*sin(θ1)*cos(θ2) + 0.05*sin(θ2)*cos(θ1), 0.5*sin(θ1)*sin(θ2) + 0.5*cos(θ1)*cos(θ2), 0, 0],         
#         #m4
#         [k1x*cos(θ1) - 0.05*k2x*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1)) + 0.9*k2y*(sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2)), -0.05*k2x*(-sin(θ1)*cos(θ2) + sin(θ2)*cos(θ1)) + 0.9*k2y*(-sin(θ1)*sin(θ2) - cos(θ1)*cos(θ2)), 0, sin(θ1), 0, 0.05*sin(θ1)*sin(θ2) + 0.05*cos(θ1)*cos(θ2), 0.9*sin(θ1)*cos(θ2) - 0.9*sin(θ2)*cos(θ1), 0, 0],
#         [k1y*sin(θ1) - 0.05*k2x*(sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2)) + 0.9*k2y*(-sin(θ1)*cos(θ2) + sin(θ2)*cos(θ1)), -0.05*k2x*(-sin(θ1)*sin(θ2) - cos(θ1)*cos(θ2)) + 0.9*k2y*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1)), 0, 0, -cos(θ1), -0.05*sin(θ1)*cos(θ2) + 0.05*sin(θ2)*cos(θ1), 0.9*sin(θ1)*sin(θ2) + 0.9*cos(θ1)*cos(θ2), 0, 0],
#         #m5
#         [k1x*cos(θ1) + k2y*(sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2)) + 0.8*k3x*((sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*cos(θ1) + (sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*sin(θ1)) - 0.05*k3y*(-(sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*sin(θ1) + (sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*cos(θ1)), k2y*(-sin(θ1)*sin(θ2) - cos(θ1)*cos(θ2)) + 0.8*k3x*(-(sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*cos(θ1) + (-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*sin(θ1)) - 0.05*k3y*((sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*sin(θ1) + (-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*cos(θ1)), 0.8*k3x*(-(-sin(θ2)*sin(θ3) - cos(θ2)*cos(θ3))*cos(θ1) + (sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*sin(θ1)) - 0.05*k3y*((-sin(θ2)*sin(θ3) - cos(θ2)*cos(θ3))*sin(θ1) + (sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*cos(θ1)), sin(θ1), 0, 0, sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1), 0.8*(sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*sin(θ1) - 0.8*(sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*cos(θ1), -0.05*(sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*cos(θ1) - 0.05*(sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*sin(θ1)],
#         [k1y*sin(θ1) + k2y*(-sin(θ1)*cos(θ2) + sin(θ2)*cos(θ1)) + 0.8*k3x*(-(-sin(θ2)*sin(θ3) - cos(θ2)*cos(θ3))*sin(θ1) + (-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*cos(θ1)) - 0.05*k3y*((sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*cos(θ1) - (-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*sin(θ1)), k2y*(sin(θ1)*cos(θ2) - sin(θ2)*cos(θ1)) + 0.8*k3x*((-sin(θ2)*sin(θ3) - cos(θ2)*cos(θ3))*sin(θ1) + (sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*cos(θ1)) - 0.05*k3y*((-sin(θ2)*sin(θ3) - cos(θ2)*cos(θ3))*cos(θ1) + (-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*sin(θ1)), 0.8*k3x*((sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*sin(θ1) + (-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*cos(θ1)) - 0.05*k3y*((sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*cos(θ1) + (sin(θ2)*cos(θ3) - sin(θ3)*cos(θ2))*sin(θ1)), 0, -cos(θ1), 0, sin(θ1)*sin(θ2) + cos(θ1)*cos(θ2), 0.8*(-sin(θ2)*sin(θ3) - cos(θ2)*cos(θ3))*cos(θ1) + 0.8*(-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*sin(θ1), -0.05*(sin(θ2)*sin(θ3) + cos(θ2)*cos(θ3))*sin(θ1) - 0.05*(-sin(θ2)*cos(θ3) + sin(θ3)*cos(θ2))*cos(θ1)]
#         ])


# H_k=np.array([
# [                                                                                                                                                                                                                                 0.3*k1x*np.cos(θ1) + 0.05*k1y*np.sin(θ1),                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                            0,  0.3*np.sin(θ1), -0.05*np.cos(θ1),                                            0,                                         0,                                                                                                                                                                                                          0],
# [                                                                                                                                                                                                                                 0.3*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1),                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                            0, -0.3*np.cos(θ1), -0.05*np.sin(θ1),                                            0,                                         0,                                                                                                                                                                                                          0],
# [                                                                                                                                                                                                                                 0.7*k1x*np.cos(θ1) + 0.05*k1y*np.sin(θ1),                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                            0,  0.7*np.sin(θ1), -0.05*np.cos(θ1),                                            0,                                         0,                                                                                                                                                                                                          0],
# [                                                                                                                                                                                                                                 0.7*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1),                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                            0, -0.7*np.cos(θ1), -0.05*np.sin(θ1),                                            0,                                         0,                                                                                                                                                                                                          0],
# [                                                                                                                                                           k1x*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)),                                                                                                                                                          -0.05*k2x*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)),                                                                                                                                                                                                            0,      np.sin(θ1),             0,  0.05*np.sin(θ1)*np.sin(θ2) + 0.05*np.cos(θ1)*np.cos(θ2), 0.5*np.sin(θ1)*np.cos(θ2) - 0.5*np.sin(θ2)*np.cos(θ1),                                                                                                                                                                                                          0],
# [                                                                                                                                                          k1y*np.sin(θ1) - 0.05*k2x*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)),                                                                                                                                                           -0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)),                                                                                                                                                                                                            0,            0,      -np.cos(θ1), -0.05*np.sin(θ1)*np.cos(θ2) + 0.05*np.sin(θ2)*np.cos(θ1), 0.5*np.sin(θ1)*np.sin(θ2) + 0.5*np.cos(θ1)*np.cos(θ2),                                                                                                                                                                                                          0],
# [                                                                                                                                                           k1x*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)),                                                                                                                                                          -0.05*k2x*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)),                                                                                                                                                                                                            0,      np.sin(θ1),             0,  0.05*np.sin(θ1)*np.sin(θ2) + 0.05*np.cos(θ1)*np.cos(θ2), 0.9*np.sin(θ1)*np.cos(θ2) - 0.9*np.sin(θ2)*np.cos(θ1),                                                                                                                                                                                                          0],
# [                                                                                                                                                          k1y*np.sin(θ1) - 0.05*k2x*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)),                                                                                                                                                           -0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)),                                                                                                                                                                                                            0,            0,      -np.cos(θ1), -0.05*np.sin(θ1)*np.cos(θ2) + 0.05*np.sin(θ2)*np.cos(θ1), 0.9*np.sin(θ1)*np.sin(θ2) + 0.9*np.cos(θ1)*np.cos(θ2),                                                                                                                                                                                                          0],
# [  k1x*np.cos(θ1) + k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) - 0.05*k3*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) + 0.8*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) - 0.05*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) + 0.8*k3*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), -0.05*k3*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) + 0.8*k3*(-(-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)),      np.sin(θ1),             0,                                            0,         np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1),    0.8*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - 0.05*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) - 0.05*(np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1) - 0.8*(np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)],
# [k1y*np.sin(θ1) + k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.8*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) - (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.8*k3*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)),     0.8*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)),            0,      -np.cos(θ1),                                            0,         np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2), -0.05*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - 0.8*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + 0.8*(-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1) - 0.05*(-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)]])


# ################################################################################################

#Evaluo la derivada numerica

H_e = sp.zeros(10, 8)

valores = np.array([1.2, 1.5, 0.75,1,1,1,1,1], dtype=np.float64)

delta=1E-8

for i in range (8):
    valores[i]+=delta
    h_2 = posicion_k(*valores)
    valores[i]+=(-2*delta)
    h_1 = posicion_k(*valores)
    H_e[:,i]=(h_2-h_1)/(2*delta)
    valores[i]+=delta
    
H_evaluada = H_k(1.2, 1.5, 0.75,1,1,1,1,1)

dif = np.array(H_evaluada - H_e, dtype=float)

#########################################################################•

##################################### FILTRO DE KALMAN ##############################################
from funciones import H
from funciones import posicion

delta_t=10E-2
sigma_s = 10E-4
sigma_p = 10

phi =     np.array([[1, 0, 0,    delta_t, 0, 0],
                    [0, 1, 0,    0, delta_t, 0],
                    [0, 0, 1,    0, 0, delta_t],
                    [0, 0, 0,    1, 0, 0],
                    [0, 0, 0,    0, 1, 0],
                    [0, 0, 0,    0, 0, 1]])

phi_t = phi.transpose() 

P = np.identity(6)

covarianza_ruido_planta= np.array([[0.25*pow(delta_t,4), 0, 0, 0.50*pow(delta_t,3), 0, 0],
                                    [0, 0.25*pow(delta_t,4), 0, 0, 0.50*pow(delta_t,3), 0],
                                    [0, 0, 0.25*pow(delta_t,4), 0, 0,  0.50*pow(delta_t,3)],
                                    [0.50*pow(delta_t,3), 0, 0, pow(delta_t,2), 0, 0],
                                    [0, 0.50*pow(delta_t,3), 0, 0, pow(delta_t,2), 0],
                                    [0, 0, 0.50*pow(delta_t,3), 0, 0, pow(delta_t,2)]]) * sigma_p


covarianza_sensores = np.identity(10)* sigma_s

x_filtro=np.zeros((6, 400))
import numpy as np

x_correc = np.array ([0,0,0,0,0,0]).reshape(-1, 1)

for i in range (400):
    
    ## Etapa de predicción
    x_predic = phi @ x_correc
    P = (phi @ P @ phi_t +  covarianza_ruido_planta).astype(np.float64)
        
    # Etapa de corrección
    θ1=float(x_predic[0][0])
    θ2=float(x_predic[1][0])
    θ3=float(x_predic[2][0])
    
    H_kalman = H(θ1,θ2,θ3)
    #Matriz de ganancia de Kalman
    K = P @ H_kalman.transpose() @ (np.linalg.inv ((H_kalman @ P @ H_kalman.transpose()) + covarianza_sensores).astype(np.float64))
    h_kalman = posicion(θ1,θ2,θ3)
    y=np.array ([[x_con_ruido[0][i], y_con_ruido[0][i], x_con_ruido[1][i], y_con_ruido[1][i], x_con_ruido[2][i], y_con_ruido[2][i], x_con_ruido[3][i], y_con_ruido[3][i], x_con_ruido[4][i], y_con_ruido[4][i]]])
    x_correc = np.array (x_predic + K @ (y.reshape(-1,1) - h_kalman))
    
    P = (np.identity(6)- K @ H_kalman) @ P
    
    x_filtro[0][i]=x_correc[0]
    x_filtro[1][i]=x_correc[1]
    x_filtro[2][i]=x_correc[2]
    x_filtro[3][i]=x_correc[3]
    x_filtro[4][i]=x_correc[4]
    x_filtro[5][i]=x_correc[5]
    

valores_originales = np.vstack((theta1_copia, theta2_copia, theta3_copia, theta1_p, theta2_p, theta3_p))
graf.graficar_thetas_kalman(valores_originales, x_filtro )

# ############################################# OPTIMIZACION ######################################################################

theta1_opt,theta2_opt,theta3_opt=alg.optimizar_thetas(x_con_ruido, y_con_ruido)
graf.graficar_thetas_optimizacion(theta1_opt, theta2_opt, theta3_opt, theta1_copia, theta2_copia, theta3_copia)

########################### CALCULO EL VALOR RMS #######################################

def rms_difference(x, y):
    # Calcula la diferencia entre las dos señales
    diff = x - y
    
    # Calcula el valor RMS de la diferencia
    rms = np.sqrt(np.sum(diff**2))
    
    return rms

# Ejemplo de uso
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

theta1_filtro=x_filtro[0] # angulos segun Kalmna
theta2_filtro=x_filtro[1] # angulos segun Kalmna
theta3_filtro=x_filtro[2] # angulos segun Kalmna


#THETA1
rms_diff_filtro = rms_difference(theta1_filtro[20:], theta1_copia[20:])
rms_diff_opt = rms_difference(theta1_opt, theta1_copia)
rms_diff_filtro_opt = rms_difference(theta1_filtro[20:], theta1_opt[20:])


print("THETA 1")
print("Valor RMS de la diferencia entre la señal original y la obtenida por Kalman:", rms_diff_filtro)
print("Valor RMS de la diferencia entre la señal original y la obtenida por Optimización:", rms_diff_opt)
print("Valor RMS de la diferencia entre las señales obtenidas por Kalman y Optimización:", rms_diff_filtro_opt)

#THETA2
rms_diff_filtro = rms_difference(theta2_filtro[20:], theta2_copia[20:])
rms_diff_opt = rms_difference(theta2_opt, theta2_copia)
rms_diff_filtro_opt = rms_difference(theta2_filtro[20:], theta2_opt[20:])

print("THETA 2")
print("Valor RMS de la diferencia entre la señal original y la obtenida por Kalman:", rms_diff_filtro)
print("Valor RMS de la diferencia entre la señal original y la obtenida por Optimización:", rms_diff_opt)
print("Valor RMS de la diferencia entre las señales obtenidas por Kalman y Optimización:", rms_diff_filtro_opt)

#THETA3
rms_diff_filtro = rms_difference(theta3_filtro[20:], theta3_copia[20:])
rms_diff_opt = rms_difference(theta3_opt, theta3_copia)
rms_diff_filtro_opt = rms_difference(theta3_filtro[20:], theta3_opt[20:])

print("THETA 3")
print("Valor RMS de la diferencia entre la señal original y la obtenida por Kalman:", rms_diff_filtro)
print("Valor RMS de la diferencia entre la señal original y la obtenida por Optimización:", rms_diff_opt)
print("Valor RMS de la diferencia entre las señales obtenidas por Kalman y Optimización:", rms_diff_filtro_opt)

#########################################################################################################################

#Optimizacion con factores de escala
from funciones import H
from funciones import posicion
from funciones import posicion_k
from funciones import H_k

x = np.linspace(0, 400, 400)

def position_function(θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3):

    h_k = posicion_k(θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3)
        
    return h_k

def jacobian(params, y):
    
    θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3 = params

    jac = H_k(θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3)
    
    return jac


# Definimos la función de error
def error_function(params, y_observed):
    
    θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3 = params    
    y_predicted = np.squeeze(position_function(θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3))
        
    return y_predicted - y_observed

initial_guess = [0.707107,0,0.707107,1,1,1,1,1]

theta_opt= []
k_opt = []

for i in range (400):
    y_observed=np.array([x_con_ruido [0,i], y_con_ruido [0,i], x_con_ruido [1,i], y_con_ruido [1,i], x_con_ruido [2,i], y_con_ruido [2,i], x_con_ruido [3,i], y_con_ruido [3,i],  x_con_ruido [4, i], y_con_ruido [4,i]])
    result = least_squares(error_function, initial_guess, method='trf', args=(y_observed,))
    initial_guess=result.x
    theta_opt.append(result.x[:3])
    k_opt.append(result.x[3:])
    print (i)


promedios = []

# Itera sobre el rango de las columnas
for i in range(len(k_opt[0])):
    # Extrae la columna i
    columna_i = [fila[i] for fila in k_opt]
    # Calcula el promedio de la columna i
    promedio_i = sum(columna_i) / len(columna_i)
    # Agrega el promedio a la lista de promedios
    promedios.append(promedio_i)

# Imprime los promedios
for i, promedio in enumerate(promedios):
    print(f"Promedio de k{i+1}_opt:", promedio)

######################################################################################################################



#Concatenar el jacobiano

    H_k=np.array([
    [                                                                                                                                                                                                                                 0.3*k1x*np.cos(θ1) + 0.05*k1y*np.sin(θ1),                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                            0,  0.3*np.sin(θ1), -0.05*np.cos(θ1),                                            0,                                         0,                                                                                                                                                                                                          0],
    [                                                                                                                                                                                                                                 0.3*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1),                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                            0, -0.3*np.cos(θ1), -0.05*np.sin(θ1),                                            0,                                         0,                                                                                                                                                                                                          0],
    [                                                                                                                                                                                                                                 0.7*k1x*np.cos(θ1) + 0.05*k1y*np.sin(θ1),                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                            0,  0.7*np.sin(θ1), -0.05*np.cos(θ1),                                            0,                                         0,                                                                                                                                                                                                          0],
    [                                                                                                                                                                                                                                 0.7*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1),                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                            0, -0.7*np.cos(θ1), -0.05*np.sin(θ1),                                            0,                                         0,                                                                                                                                                                                                          0],
    [                                                                                                                                                           k1x*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)),                                                                                                                                                          -0.05*k2x*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)),                                                                                                                                                                                                            0,      np.sin(θ1),             0,  0.05*np.sin(θ1)*np.sin(θ2) + 0.05*np.cos(θ1)*np.cos(θ2), 0.5*np.sin(θ1)*np.cos(θ2) - 0.5*np.sin(θ2)*np.cos(θ1),                                                                                                                                                                                                          0],
    [                                                                                                                                                          k1y*np.sin(θ1) - 0.05*k2x*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)),                                                                                                                                                           -0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)),                                                                                                                                                                                                            0,            0,      -np.cos(θ1), -0.05*np.sin(θ1)*np.cos(θ2) + 0.05*np.sin(θ2)*np.cos(θ1), 0.5*np.sin(θ1)*np.sin(θ2) + 0.5*np.cos(θ1)*np.cos(θ2),                                                                                                                                                                                                          0],
    [                                                                                                                                                           k1x*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)),                                                                                                                                                          -0.05*k2x*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)),                                                                                                                                                                                                            0,      np.sin(θ1),             0,  0.05*np.sin(θ1)*np.sin(θ2) + 0.05*np.cos(θ1)*np.cos(θ2), 0.9*np.sin(θ1)*np.cos(θ2) - 0.9*np.sin(θ2)*np.cos(θ1),                                                                                                                                                                                                          0],
    [                                                                                                                                                          k1y*np.sin(θ1) - 0.05*k2x*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)),                                                                                                                                                           -0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)),                                                                                                                                                                                                            0,            0,      -np.cos(θ1), -0.05*np.sin(θ1)*np.cos(θ2) + 0.05*np.sin(θ2)*np.cos(θ1), 0.9*np.sin(θ1)*np.sin(θ2) + 0.9*np.cos(θ1)*np.cos(θ2),                                                                                                                                                                                                          0],
    [  k1x*np.cos(θ1) + k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) - 0.05*k3*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) + 0.8*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) - 0.05*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) + 0.8*k3*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), -0.05*k3*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) + 0.8*k3*(-(-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)),      np.sin(θ1),             0,                                            0,         np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1),    0.8*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - 0.05*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) - 0.05*(np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1) - 0.8*(np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)],
    [k1y*np.sin(θ1) + k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.8*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) - (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.8*k3*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)),     0.8*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)),            0,      -np.cos(θ1),                                            0,         np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2), -0.05*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - 0.8*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + 0.8*(-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1) - 0.05*(-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)]])



   

    

#######################################################################################################################################################



#Optimizacion global con factores de escala
from funciones import H
from funciones import posicion
from funciones import posicion_k
from funciones import H_k

x = np.linspace(0, 400, 400)
j=np.zeros((4000, 1205))
casero=np.zeros((4000, 1205))

# def jacobian(params, y_observed):
   
    # global j, casero
    # H_total = np.zeros((4000, 1205))
    
    # for i in range (400):
        
    #     xi = np.concatenate((params[3*i:3*i+3], params[-5:]))
    #     H_total[10*i:10*(i+1),3*i:3*(i+1)]=H_k(*xi)[:,:3]
    #     H_total[10*i:10*(i+1),-5:] =H_k(*xi)[:,-5:]

    # j=H_total
    
def jacobian(params, y_observed):

    fila=[]
    columna=[]
    valor=[]
    
    for i in range (400):
        
        xi = np.concatenate((params[3*i:3*i+3], params[-5:]))
        jac=H_k(*xi)
        angulos=jac[:,:3]
        factores=jac[:,-5:]
 
        for f in range (10):
            for c in range (3): 
                if angulos[f,c]!=0:
                    fila.append(f+(10*i))
                    columna.append(c+(3*i))
                    valor.append(angulos[f,c])
                    
            for k in range (5):
                if factores[f,k]!=0:
                    fila.append(f+(10*i))
                    columna.append(1200+k)
                    valor.append(factores[f,k])
                    
    columna = np.array(columna)
    fila = np.array(fila)
    valor = np.array(valor)
    matriz_dispersa = coo_matrix((valor, (fila, columna)))
        
    return matriz_dispersa



# Definimos la función de error
def calcular_y_predicted(params):
    
    y_predicted = np.zeros(4000)
    
    for i in range(400):
      xi = np.concatenate((params[3*i:3*i+3], params[-5:]))
      y_predicted[10*i:10*i+10] = np.squeeze(posicion_k(*xi))
     
    return y_predicted
        
y_observed = np.zeros(4000)
for i in range(400):   # Recorremos los frames
    for j in range(5): # Recorremos los markers
        y_observed[10*i+2*j] = x_con_ruido[j, i]
        y_observed[10*i+2*j+1] = y_con_ruido[j, i]

    
#Calculo el vector initial guess
initial_guess=np.zeros(1205)
# for i in range (400):
#     initial_guess[i*3]=theta1[i]
#     initial_guess[i*3+1]=theta2[i]
#     initial_guess[i*3+2]=theta3[i]
for i in range (5):
    #lleno de 1 el lugar correspondiente a las constantes
    initial_guess[1200+i]=1


def error_function(params, y_observed):
    
    global iteration_count
    iteration_count += 1
    #print("Iteración en error_function:", iteration_count)
    y_predicted=calcular_y_predicted(params)
    res=y_predicted-y_observed
    print (np.sqrt(np.vdot(res, res)/res.size))
    return (y_predicted - y_observed)

iteration_count=0

inicio = time.time()

result = least_squares(error_function, initial_guess, jac=jacobian, method='dogbox', args=(y_observed,),max_nfev=100)

fin = time.time()
tiempo_transcurrido = fin - inicio
print("Tiempo transcurrido:", tiempo_transcurrido, "segundos")


# print (result.x)
p=result.x

# matriz_dispersa = csr_matrix(j)
# ver = matriz_dispersa.toarray()
# print("Matriz dispersa CSR:")
# print(matriz_dispersa.nnz)

