import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, rad, pretty, pprint, Matrix
from funciones import H_3,posicion


# def optimizar_thetas(x_con_ruido, y_con_ruido):
    
    
    
#     x = np.linspace(0, 400, 400)
    
    
#     ### y observed es m
#     # Definimos la función sinusoidal que queremos ajustar
#     def position_function(θ1, θ2, θ3):
#         h = np.array([
#             [0.3*sin(θ1) - 0.05*cos( θ1)],
#             [-0.05*sin(θ1) - 0.3*cos( θ1)],
#             [0.7*sin(θ1) - 0.05*cos( θ1)],
#             [-0.05*sin(θ1) - 0.7*cos( θ1)],
#             [0.05*sin(θ1)*sin( θ2) + 0.5*sin( θ1)*cos( θ2) + sin( θ1) - 0.5*sin( θ2)*cos( θ1) + 0.05*cos( θ1)*cos( θ2)],
#             [0.5*sin(θ1)*sin( θ2) - 0.05*sin( θ1)*cos( θ2) + 0.05*sin( θ2)*cos( θ1) + 0.5*cos( θ1)*cos( θ2) - cos( θ1)],
#             [0.05*sin(θ1)*sin( θ2) + 0.9*sin( θ1)*cos( θ2) + sin( θ1) - 0.9*sin( θ2)*cos( θ1) + 0.05*cos( θ1)*cos( θ2)],
#             [0.9*sin( θ1)*sin( θ2) - 0.05*sin( θ1)*cos( θ2) + 0.05*sin( θ2)*cos( θ1) + 0.9*cos( θ1)*cos( θ2) - cos( θ1)],
#             [0.8*(sin( θ2)*sin( θ3) + cos( θ2)*cos( θ3))*sin( θ1) - 0.05*(sin( θ2)*sin( θ3) + cos( θ2)*cos( θ3))*cos( θ1) - 0.05*(sin( θ2)*cos( θ3) - sin( θ3)*cos( θ2))*sin( θ1) - 0.8*(sin( θ2)*cos( θ3) - sin( θ3)*cos( θ2))*cos( θ1) + sin( θ1)*cos( θ2) + sin( θ1) - sin( θ2)*cos( θ1)],
#             [-0.05*(sin( θ2)*sin( θ3) + cos( θ2)*cos( θ3))*sin( θ1) - 0.8*(sin( θ2)*sin( θ3) + cos( θ2)*cos( θ3))*cos( θ1) + 0.8*(-sin( θ2)*cos( θ3) + sin( θ3)*cos( θ2))*sin( θ1) - 0.05*(-sin( θ2)*cos( θ3) + sin( θ3)*cos( θ2))*cos( θ1) + sin( θ1)*sin( θ2) + cos( θ1)*cos( θ2) - cos( θ1)]
        
    
#         ],dtype=float)
        
#         return h
    




    
#     # Definimos la función de error
#     def error_function(params, x, y_observed):
#         θ1, θ2, θ3 = params
#         y_predicted = position_function(θ1, θ2, θ3)
#         return np.squeeze(y_predicted - y_observed)
    
#     def jacobian(params, x, y):
#         θ1, θ2, θ3 = params
        
#         H = np.array([[0.05*np.sin(θ1) + 0.3* np.cos(θ1), 0, 0],
#                       [0.3* np.sin(θ1) - 0.05* np.cos( θ1), 0, 0],
#                       [0.05* np.sin( θ1) + 0.7* np.cos( θ1), 0, 0],
#                       [0.7* np.sin(θ1) - 0.05* np.cos(θ1), 0, 0],
#                       [0.5* np.sin(θ1)* np.sin(θ2) - 0.05* np.sin(θ1)* np.cos(θ2) + 0.05* np.sin(θ2)* np.cos(θ1) + 0.5* np.cos(θ1)* np.cos(θ2) +  np.cos(θ1), -0.5* np.sin(θ1)* np.sin(θ2) + 0.05* np.sin(θ1)* np.cos(θ2) - 0.05* np.sin(θ2)* np.cos(θ1) - 0.5* np.cos(θ1)* np.cos(θ2), 0],                                
#                       [-0.05* np.sin(θ1)* np.sin(θ2) - 0.5* np.sin(θ1)* np.cos(θ2) +  np.sin(θ1) + 0.5* np.sin(θ2)* np.cos(θ1) - 0.05* np.cos(θ1)* np.cos(θ2), 0.05* np.sin(θ1)* np.sin(θ2) + 0.5* np.sin(θ1)* np.cos(θ2) - 0.5* np.sin(θ2)* np.cos(θ1) + 0.05* np.cos(θ1)* np.cos(θ2), 0],                   
#                       [0.9* np.sin(θ1)* np.sin(θ2) - 0.05* np.sin(θ1)* np.cos(θ2) + 0.05* np.sin(θ2)* np.cos(θ1) + 0.9* np.cos(θ1)* np.cos(θ2) +  np.cos(θ1), -0.9* np.sin(θ1)* np.sin(θ2) + 0.05* np.sin(θ1)* np.cos(θ2) - 0.05* np.sin(θ2)* np.cos(θ1) - 0.9* np.cos(θ1)* np.cos(θ2), 0],                                    
#                       [-0.05* np.sin(θ1)* np.sin(θ2) - 0.9* np.sin(θ1)* np.cos(θ2) +  np.sin(θ1) + 0.9* np.sin(θ2)* np.cos(θ1) - 0.05* np.cos(θ1)* np.cos(θ2), 0.05* np.sin(θ1)* np.sin(θ2) + 0.9* np.sin(θ1)* np.cos(θ2) - 0.9* np.sin(θ2)* np.cos(θ1) + 0.05* np.cos(θ1)* np.cos(θ2), 0],
#                       [ (0.05* np.sin(θ2)* np.sin(θ3) + 0.05* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) + (0.8* np.sin(θ2)* np.sin(θ3) + 0.8* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) - (0.05* np.sin(θ2)* np.cos(θ3) - 0.05* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) + (0.8* np.sin(θ2)* np.cos(θ3) - 0.8* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) +  np.sin(θ1)* np.sin(θ2) +  np.cos(θ1)* np.cos(θ2) +  np.cos(θ1), -(0.05* np.sin(θ2)* np.sin(θ3) + 0.05* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) - (0.8* np.sin(θ2)* np.sin(θ3) + 0.8* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) + (-0.8* np.sin(θ2)* np.cos(θ3) + 0.8* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) - (-0.05* np.sin(θ2)* np.cos(θ3) + 0.05* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) -  np.sin(θ1)* np.sin(θ2) -  np.cos(θ1)* np.cos(θ2), -(-0.8* np.sin(θ2)* np.sin(θ3) - 0.8* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) - (-0.05* np.sin(θ2)* np.sin(θ3) - 0.05* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) - (0.05* np.sin(θ2)* np.cos(θ3) - 0.05* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) + (0.8* np.sin(θ2)* np.cos(θ3) - 0.8* np.sin(θ3)* np.cos(θ2))* np.sin(θ1)],
#                       [ (-0.05* np.sin(θ2)* np.sin(θ3) - 0.05* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) + (0.8* np.sin(θ2)* np.sin(θ3) + 0.8* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) + (-0.8* np.sin(θ2)* np.cos(θ3) + 0.8* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) + (-0.05* np.sin(θ2)* np.cos(θ3) + 0.05* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) -  np.sin(θ1)* np.cos(θ2) +  np.sin(θ1) +  np.sin(θ2)* np.cos(θ1), (-0.8* np.sin(θ2)* np.sin(θ3) - 0.8* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) - (-0.05* np.sin(θ2)* np.sin(θ3) - 0.05* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) - (-0.8* np.sin(θ2)* np.cos(θ3) + 0.8* np.sin(θ3)* np.cos(θ2))* np.cos(θ1) + (0.05* np.sin(θ2)* np.cos(θ3) - 0.05* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) +  np.sin(θ1)* np.cos(θ2) -  np.sin(θ2)* np.cos(θ1), -(0.05* np.sin(θ2)* np.sin(θ3) + 0.05* np.cos(θ2)* np.cos(θ3))* np.cos(θ1) + (0.8* np.sin(θ2)* np.sin(θ3) + 0.8* np.cos(θ2)* np.cos(θ3))* np.sin(θ1) + (-0.05* np.sin(θ2)* np.cos(θ3) + 0.05* np.sin(θ3)* np.cos(θ2))* np.sin(θ1) - (0.8* np.sin(θ2)* np.cos(θ3) - 0.8* np.sin(θ3)* np.cos(θ2))* np.cos(θ1)]], dtype=float)
    
#         return H
    
    
#     # Supongamos unos valores iniciales para los parámetros del ajuste
#     initial_guess = [0, 0, 0]
#     thetas= []
#     for i in range (400):
#         y_observed=np.array([x_con_ruido [0,i], y_con_ruido [0,i], x_con_ruido [1,i], y_con_ruido [1,i], x_con_ruido [2,i], y_con_ruido [2,i], x_con_ruido [3,i], y_con_ruido [3,i],  x_con_ruido [4, i], y_con_ruido [4,i]])
#         ######ACOMODAR ESTO
#         y_observed = np.reshape(y_observed, (10, 1))
#         result = least_squares(error_function, initial_guess, jac=jacobian, args=(x, y_observed))
#         initial_guess=result.x
#         thetas.append(np.squeeze(result.x))
    
    
#     theta1_opt=[]
#     theta2_opt=[]
#     theta3_opt=[]
    
#     for i in range (400):
#         theta1_opt.append(thetas[i][0])
#         theta2_opt.append(thetas[i][1])
#         theta3_opt.append(thetas[i][2])


#     return theta1_opt,theta2_opt,theta3_opt
    

def optimizar_thetas(x_con_ruido, y_con_ruido):
    
    
    
    x = np.linspace(0, 400, 400)
    
    
    ### y observed es m
    # Definimos la función sinusoidal que queremos ajustar
    def position_function(θ1, θ2, θ3):
        
        h=posicion(θ1, θ2, θ3)
        
        return h
        
    # Definimos la función de error
    def error_function(params, x, y_observed):
        θ1, θ2, θ3 = params
        y_predicted = position_function(θ1, θ2, θ3)
        return np.squeeze(y_predicted - y_observed)
    
    def jacobian(params, x, y):
        θ1, θ2, θ3 = params
                
        return H_3(θ1, θ2, θ3)
    
    
    # Supongamos unos valores iniciales para los parámetros del ajuste
    initial_guess = [0, 0, 0]
    thetas= []
    for i in range (400):
        y_observed=np.array([x_con_ruido [0,i], y_con_ruido [0,i], x_con_ruido [1,i], y_con_ruido [1,i], x_con_ruido [2,i], y_con_ruido [2,i], x_con_ruido [3,i], y_con_ruido [3,i],  x_con_ruido [4, i], y_con_ruido [4,i]])
        ######ACOMODAR ESTO
        y_observed = np.reshape(y_observed, (10, 1))
        result = least_squares(error_function, initial_guess, jac=jacobian, method='lm', args=(x, y_observed))
        initial_guess=result.x
        thetas.append(np.squeeze(result.x))
    
    
    theta1_opt=[]
    theta2_opt=[]
    theta3_opt=[]
    
    for i in range (400):
        theta1_opt.append(thetas[i][0])
        theta2_opt.append(thetas[i][1])
        theta3_opt.append(thetas[i][2])


    return theta1_opt,theta2_opt,theta3_opt
    