import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import copy
from scipy.signal import convolve


############################## GENERO ANGULOS ###############################################

############ THETA 1 ############
# Parámetros
a = 0.5  # Amplitud
b = 1  # Desplazamiento vertical
omega = 0.5*np.pi  # Frecuencia angular (en este caso, una vuelta completa cada unidad de tiempo)
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
omega = 2*np.pi  # Frecuencia angular (en este caso, una vuelta completa cada unidad de tiempo)
# Vector de tiempo
t = np.linspace(0, 4, 400)  # Desde 0 hasta 10 con 400 puntos
# Función de ángulo
phi = np.pi / 2  # Desplazamiento en omega t (en radianes)
theta2 = a * a * np.cos(omega * t + phi) + b
theta2_copia=copy.copy(theta2)

############ THETA 3 ############
# Parámetros
a = 0.5  # Amplitud
b = 0 # Desplazamiento vertical
omega = 2*np.pi  # Frecuencia angular (en este caso, una vuelta completa cada unidad de tiempo)
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



from sympy import symbols, sin, cos, rad, pretty, pprint, Matrix
import sympy as sp

theta_1, theta_2, theta_3 = symbols('θ1 θ2 θ3', real=True)



A = [[sin(theta_1), cos(theta_1)], 
     [-cos(theta_1), sin(theta_1)]]

B = [[-sin(theta_2 ), cos(theta_2)], 
     [cos(theta_2), sin(theta_2)]]

C = [[-sin(theta_3), cos(theta_3)], 
     [cos(theta_3), sin(theta_3)]]


################## PENDULO 1 #######################

r0_p1=[0,0]
rp_l=np.array([1,0])

#MARCADOR 1
rp_l_m1=np.array([0.3,-l_m])
rp_m1=np.dot(A, rp_l_m1)
[x_m1, y_m1] = [r0_p1[0]+rp_m1[0], r0_p1[1]+rp_m1[1]] 
############

#MARCADOR 2

rp_l_m2=np.array([0.7,-l_m])
rp_m2=np.dot(A, rp_l_m2)
[x_m2, y_m2] = [r0_p1[0]+rp_m2[0], r0_p1[1]+rp_m2[1]] 
############

##Calculo mi r0 del pendulo 2
rp_1=np.dot(A, rp_l)
r_p1 = [r0_p1[0]+rp_1[0], r0_p1[1]+ rp_1[1]]

################## PENDULO 2 #######################

rp_l=np.array([0,1])

#MARCADOR 3
rp_l_m3=np.array([-l_m, 0.5])
rp_m3=np.dot(np.dot(B,A), rp_l_m3)
[x_m3, y_m3] = [r_p1[0]+rp_m3[0], r_p1[1]+rp_m3[1]] 
###########

#MARCADOR 4
rp_l_m4=np.array([-l_m,0.9])
rp_m4=np.dot(np.dot(B,A), rp_l_m4)
[x_m4, y_m4] = [r_p1[0]+rp_m4[0], r_p1[1]+rp_m4[1]] 
###########

##Calculo mi r0 del pendulo 3
rp_2=np.dot(np.dot(B,A), rp_l)
r_p2 = [r_p1[0]+rp_2[0], r_p1[1]+ rp_2[1]]

################## PENDULO 3 #######################

rp_l=np.array([1,0])

#MARCADOR 5
rp_l_m5=np.array([0.8,-l_m])
rp_m5=np.dot(np.dot(np.dot(C,B),A), rp_l_m5)
[x_m5, y_m5] = [r_p2[0]+rp_m5[0], r_p2[1]+rp_m5[1]] 

#####################################################

θ1, θ2, θ3 = symbols('θ1 θ2 θ3', real=True)

h = Matrix([
    [0.3*sin(θ1) - 0.05*cos( θ1)],
    [-0.05*sin(θ1) - 0.3*cos( θ1)],
    [0.7*sin(θ1) - 0.05*cos( θ1)],
    [-0.05*sin(θ1) - 0.7*cos( θ1)],
    [0.05*sin(θ1)*sin( θ2) + 0.5*sin( θ1)*cos( θ2) + sin( θ1) - 0.5*sin( θ2)*cos( θ1) + 0.05*cos( θ1)*cos( θ2)],
    [0.5*sin(θ1)*sin( θ2) - 0.05*sin( θ1)*cos( θ2) + 0.05*sin( θ2)*cos( θ1) + 0.5*cos( θ1)*cos( θ2) - cos( θ1)],
    [0.05*sin(θ1)*sin( θ2) + 0.9*sin( θ1)*cos( θ2) + sin( θ1) - 0.9*sin( θ2)*cos( θ1) + 0.05*cos( θ1)*cos( θ2)],
    [0.9*sin( θ1)*sin( θ2) - 0.05*sin( θ1)*cos( θ2) + 0.05*sin( θ2)*cos( θ1) + 0.9*cos( θ1)*cos( θ2) - cos( θ1)],
    [0.8*(sin( θ2)*sin( θ3) + cos( θ2)*cos( θ3))*sin( θ1) - 0.05*(sin( θ2)*sin( θ3) + cos( θ2)*cos( θ3))*cos( θ1) - 0.05*(sin( θ2)*cos( θ3) - sin( θ3)*cos( θ2))*sin( θ1) - 0.8*(sin( θ2)*cos( θ3) - sin( θ3)*cos( θ2))*cos( θ1) + sin( θ1)*cos( θ2) + sin( θ1) - sin( θ2)*cos( θ1)],
    [-0.05*(sin( θ2)*sin( θ3) + cos( θ2)*cos( θ3))*sin( θ1) - 0.8*(sin( θ2)*sin( θ3) + cos( θ2)*cos( θ3))*cos( θ1) + 0.8*(-sin( θ2)*cos( θ3) + sin( θ3)*cos( θ2))*sin( θ1) - 0.05*(-sin( θ2)*cos( θ3) + sin( θ3)*cos( θ2))*cos( θ1) + sin( θ1)*sin( θ2) + cos( θ1)*cos( θ2) - cos( θ1)]
])



H = sp.Matrix([h]).jacobian([θ1, θ2, θ3])

################################################################################################

H_e = sp.zeros(10, 3)

valores_theta = np.array([20, 50, 30], dtype=np.float64)
delta=1E-8

for i in range (3):
    valores_theta[i]+=delta
    h_2 = np.vectorize(lambda x: x.evalf(n=20, subs={θ1: valores_theta[0], θ2: valores_theta[1], θ3: valores_theta[2]}))(h)
    valores_theta[i]+=(-2*delta)
    h_1 = np.vectorize(lambda x: x.evalf(n=20, subs={θ1: valores_theta[0], θ2: valores_theta[1], θ3: valores_theta[2]}))(h)
    H_e[:,i]=(h_2-h_1)/(2*delta)
    valores_theta[i]+=delta
    
H_evaluada = np.vectorize(lambda x: x.evalf(n=20, subs={θ1: valores_theta[0], θ2: valores_theta[1], θ3: valores_theta[2]}))(H)

dif = H_evaluada - H_e

                            
###################################### FILTRO DE KALMAN ##############################################


# theta1_p = np.gradient(theta1_copia)
# theta2_p = np.gradient(theta2_copia)
# theta3_p = np.gradient(theta3_copia)

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
    
    H = np.array([[0.05*np.sin(θ1) + 0.3* np.cos(θ1), 0, 0, 0, 0, 0],
                        [0.3* np.sin(θ1) - 0.05* np.cos( θ1), 0, 0, 0, 0, 0],
                        [0.05* np.sin( θ1) + 0.7* np.cos( θ1), 0, 0, 0, 0, 0],
                        [0.7* np.sin(θ1) - 0.05* np.cos(θ1), 0, 0, 0, 0, 0],
                        [0.5* np.sin(θ1)* np.sin(θ2) - 0.05* np.sin(θ1)* np.cos(θ2) + 0.05* np.sin(θ2)* np.cos(θ1) + 0.5* np.cos(θ1)* np.cos(θ2) +  np.cos(θ1), -0.5* np.sin(θ1)* np.sin(θ2) + 0.05* np.sin(θ1)* np.cos(θ2) - 0.05* np.sin(θ2)* np.cos(θ1) - 0.5* np.cos(θ1)* np.cos(θ2), 0, 0, 0, 0],                                
                        [-0.05* np.sin(θ1)* np.sin(θ2) - 0.5* np.sin(θ1)* np.cos(θ2) +  np.sin(θ1) + 0.5* np.sin(θ2)* np.cos(θ1) - 0.05* np.cos(θ1)* np.cos(θ2), 0.05* np.sin(θ1)* np.sin(θ2) + 0.5* np.sin(θ1)* np.cos(θ2) - 0.5* np.sin(θ2)* np.cos(θ1) + 0.05* np.cos(θ1)* np.cos(θ2), 0,0,0,0],                   
                        [0.9* np.sin(θ1)* np.sin(θ2) - 0.05* np.sin(θ1)* np.cos(θ2) + 0.05* np.sin(θ2)* np.cos(θ1) + 0.9* np.cos(θ1)* np.cos(θ2) +  np.cos(θ1), -0.9* np.sin(θ1)* np.sin(θ2) + 0.05* np.sin(θ1)* np.cos(θ2) - 0.05* np.sin(θ2)* np.cos(θ1) - 0.9* np.cos(θ1)* np.cos(θ2), 0,0,0,0],                                    
                        [-0.05* np.sin(θ1)* np.sin(θ2) - 0.9* np.sin(θ1)* np.cos(θ2) +  np.sin(θ1) + 0.9* np.sin(θ2)* np.cos(θ1) - 0.05* np.cos(θ1)* np.cos(θ2), 0.05* np.sin(θ1)* np.sin(θ2) + 0.9* np.sin(θ1)* np.cos(θ2) - 0.9* np.sin(θ2)* np.cos(θ1) + 0.05* np.cos(θ1)* np.cos(θ2), 0, 0, 0, 0],
                        [ (0.05* np.sin(θ2)* np.sin(θ3) + 0.05* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) + (0.8* np.sin(θ2)* np.sin(θ3) + 0.8* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) - (0.05* np.sin(θ2)* np.cos(θ3) - 0.05* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) + (0.8* np.sin(θ2)* np.cos(θ3) - 0.8* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) +  np.sin(θ1)* np.sin(θ2) +  np.cos(θ1)* np.cos(θ2) +  np.cos(θ1), -(0.05* np.sin(θ2)* np.sin(θ3) + 0.05* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) - (0.8* np.sin(θ2)* np.sin(θ3) + 0.8* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) + (-0.8* np.sin(θ2)* np.cos(θ3) + 0.8* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) - (-0.05* np.sin(θ2)* np.cos(θ3) + 0.05* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) -  np.sin(θ1)* np.sin(θ2) -  np.cos(θ1)* np.cos(θ2), -(-0.8* np.sin(θ2)* np.sin(θ3) - 0.8* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) - (-0.05* np.sin(θ2)* np.sin(θ3) - 0.05* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) - (0.05* np.sin(θ2)* np.cos(θ3) - 0.05* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) + (0.8* np.sin(θ2)* np.cos(θ3) - 0.8* np.sin(θ3)* np.cos(θ2))* np.sin(θ1), 0, 0, 0],
                        [ (-0.05* np.sin(θ2)* np.sin(θ3) - 0.05* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) + (0.8* np.sin(θ2)* np.sin(θ3) + 0.8* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) + (-0.8* np.sin(θ2)* np.cos(θ3) + 0.8* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) + (-0.05* np.sin(θ2)* np.cos(θ3) + 0.05* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) -  np.sin(θ1)* np.cos(θ2) +  np.sin(θ1) +  np.sin(θ2)* np.cos(θ1), (-0.8* np.sin(θ2)* np.sin(θ3) - 0.8* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) - (-0.05* np.sin(θ2)* np.sin(θ3) - 0.05* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) - (-0.8* np.sin(θ2)* np.cos(θ3) + 0.8* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) + (0.05* np.sin(θ2)* np.cos(θ3) - 0.05* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) +  np.sin(θ1)* np.cos(θ2) -  np.sin(θ2)* np.cos(θ1), -(0.05* np.sin(θ2)* np.sin(θ3) + 0.05* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) + (0.8* np.sin(θ2)* np.sin(θ3) + 0.8* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) + (-0.05* np.sin(θ2)* np.cos(θ3) + 0.05* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) - (0.8* np.sin(θ2)* np.cos(θ3) - 0.8* np.sin(θ3)* np.cos(θ2))* np.cos(θ1), 0, 0, 0]], dtype=float)

    #Matriz de ganancia de Kalman
        
    K = P @ H.transpose() @ (np.linalg.inv ((H @ P @ H.transpose()) + covarianza_sensores).astype(np.float64))
    
    h = Matrix([
        [0.3* np.sin( θ1) - 0.05* np.cos( θ1)],
        [-0.05* np.sin( θ1) - 0.3* np.cos( θ1)],
        [0.7* np.sin( θ1) - 0.05* np.cos( θ1)],
        [-0.05* np.sin( θ1) - 0.7* np.cos( θ1)],
        [0.05* np.sin( θ1)* np.sin( θ2) + 0.5* np.sin( θ1)* np.cos( θ2) +  np.sin( θ1) - 0.5* np.sin( θ2)* np.cos( θ1) + 0.05* np.cos( θ1)* np.cos( θ2)],
        [0.5* np.sin( θ1)* np.sin( θ2) - 0.05* np.sin( θ1)* np.cos( θ2) + 0.05* np.sin( θ2)* np.cos( θ1) + 0.5* np.cos( θ1)* np.cos( θ2) -  np.cos( θ1)],
        [0.05* np.sin( θ1)* np.sin( θ2) + 0.9* np.sin( θ1)* np.cos( θ2) +  np.sin( θ1) - 0.9* np.sin( θ2)* np.cos( θ1) + 0.05* np.cos( θ1)* np.cos( θ2)],
        [0.9* np.sin( θ1)* np.sin( θ2) - 0.05* np.sin( θ1)* np.cos( θ2) + 0.05* np.sin( θ2)* np.cos( θ1) + 0.9* np.cos( θ1)* np.cos( θ2) -  np.cos( θ1)],
        [0.8*( np.sin( θ2)* np.sin( θ3) +  np.cos( θ2)* np.cos( θ3))* np.sin( θ1) - 0.05*( np.sin( θ2)* np.sin( θ3) +  np.cos( θ2)* np.cos( θ3))* np.cos( θ1) - 0.05*( np.sin( θ2)* np.cos( θ3) -  np.sin( θ3)* np.cos( θ2))* np.sin( θ1) - 0.8*( np.sin( θ2)* np.cos( θ3) -  np.sin( θ3)* np.cos( θ2))* np.cos( θ1) +  np.sin( θ1)* np.cos( θ2) +  np.sin( θ1) -  np.sin( θ2)* np.cos( θ1)],
        [-0.05*( np.sin( θ2)* np.sin( θ3) +  np.cos( θ2)* np.cos( θ3))* np.sin( θ1) - 0.8*( np.sin( θ2)* np.sin( θ3) +  np.cos( θ2)* np.cos( θ3))* np.cos( θ1) + 0.8*(- np.sin( θ2)* np.cos( θ3) +  np.sin( θ3)* np.cos( θ2))* np.sin( θ1) - 0.05*(- np.sin( θ2)* np.cos( θ3) +  np.sin( θ3)* np.cos( θ2))* np.cos( θ1) +  np.sin( θ1)* np.sin( θ2) +  np.cos( θ1)* np.cos( θ2) -  np.cos( θ1)]
    ])
    
    y=np.array ([[x_con_ruido[0][i], y_con_ruido[0][i], x_con_ruido[1][i], y_con_ruido[1][i], x_con_ruido[2][i], y_con_ruido[2][i], x_con_ruido[3][i], y_con_ruido[3][i], x_con_ruido[4][i], y_con_ruido[4][i]]])
    
    x_correc = np.array (x_predic + K @ (y.reshape(-1,1) - h))
    
    P = (np.identity(6)- K @ H) @ P
    
    
    x_filtro[0][i]=x_correc[0]
    x_filtro[1][i]=x_correc[1]
    x_filtro[2][i]=x_correc[2]
    x_filtro[3][i]=x_correc[3]
    x_filtro[4][i]=x_correc[4]
    x_filtro[5][i]=x_correc[5]
    
    print ("Ciclo ", i, "Valores: ", x_correc)
    

valores_originales = np.vstack((theta1_copia, theta2_copia, theta3_copia, theta1_p, theta2_p, theta3_p))

# Crear una cuadrícula de subgráficos con 3 filas y 2 columnas
fig, axs = plt.subplots(3, 2, figsize=(10, 12))

# Graficar las tres primeras gráficas en la columna de la izquierda
for i, ax in enumerate(axs[:3, 0]):
    # Graficar la primera serie de datos (sin)
    ax.plot(np.linspace(0, 400, 400), valores_originales[i,:], label='Valor original', color='blue')
    # Graficar la segunda serie de datos (cos)
    ax.plot(np.linspace(0, 400, 400), x_filtro[i,:], label='valor del filtro', color='red')
    # Agregar leyenda al subplot
    ax.legend()
    # Añadir título
    ax.set_title(f'Theta {i+1}')

# Graficar las tres siguientes gráficas en la columna de la derecha
for i, ax in enumerate(axs[:3, 1]):
    # Graficar la primera serie de datos (sin)
    ax.plot(np.linspace(0, 400, 400), valores_originales[i+3,:], label='valor original', color='blue')
    # Graficar la segunda serie de datos (cos)
    ax.plot(np.linspace(0, 400, 400), x_filtro[i+3,:], label='valor del filtro', color='red')
    # Agregar leyenda al subplot
    ax.legend()
    # Añadir título
    ax.set_title(f'Derivada Theta {i+1}')

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar gráficas
plt.show()



