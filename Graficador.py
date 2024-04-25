import matplotlib.pyplot as plt
import numpy as np

def graficar_thetas_optimizacion(theta1_opt,theta2_opt,theta3_opt, theta1_copia, theta2_copia, theta3_copia ):
    

    # Definir los datos de los arrays para cada conjunto de gráficos
    datos = [
        (theta1_opt, theta1_copia),
        (theta2_opt, theta2_copia),
        (theta3_opt, theta3_copia)
    ]
    
    # Lista de etiquetas para las líneas optimizadas
    etiquetas_optimizadas = ['Theta  1  optimizado', 'Theta  2  optimizado', 'Theta  3  optimizado']
    
    # Crear una figura con tres subgráficos en una columna
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
    # Iterar sobre los datos y las etiquetas y graficar cada par de arrays en un subgráfico
    for (theta_opt, theta_copia), etiqueta_optimizada, ax in zip(datos, etiquetas_optimizadas, axs):
        ax.plot(theta_opt, label=etiqueta_optimizada)
        ax.plot(theta_copia, label='Theta original')
        ax.set_xlabel('Eje X')
        ax.set_ylabel('Eje Y')
        ax.set_title('Optimizacion')
        ax.legend()
    
    # Ajustar el espaciado entre los subgráficos
    plt.tight_layout()
    
    # Mostrar la figura con los subgráficos
    plt.show()


def graficar_thetas_kalman(valores_originales, x_filtro ):
    
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