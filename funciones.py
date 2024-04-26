import numpy as np
from functools import partial

def posicion_k(θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3x, k3y):


    # h_k = np.array([
        
    #     #m1
    #     [0.3*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1)],
    #     [-0.3*k1x*np.cos(θ1) - 0.05*k1y*np.sin(θ1)],
    #     #m2
    #     [0.7*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1)],
    #     [-0.7*k1x*np.cos(θ1) - 0.05*k1y*np.sin(θ1)],        
    #     #m3
    #     [k1x*np.sin(θ1) - 0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1))],
    #     [-k1y*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2))],         
    #     #m4
    #     [k1x*np.sin(θ1) - 0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1))],
    #     [-k1y*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2))],
    #     #m5
    #     [k1x*np.sin(θ1) + k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.8*k3x*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3y*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1))],
    #     [-k1y*np.cos(θ1) + k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.8*k3x*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)) - 0.05*k3y*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1))]
    #     ],dtype=float)
        
    h_k=np.array([
    [ 0.3*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1)],
    [-0.3*k1x*np.cos(θ1) - 0.05*k1y*np.sin(θ1)],                                                                                                                               
    [ 0.7*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1)],
    [-0.7*k1x*np.cos(θ1) - 0.05*k1y*np.sin(θ1)],                                                                                                       
    [k1x*np.sin(θ1) - 0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1))],
    [-k1y*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2))],                                                                                                                                                               
    [k1x*np.sin(θ1) - 0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1))],
    [-k1y*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2))],                                                                                                                                                                            
    [ k1x*np.sin(θ1) + k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.8*k3x*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3y*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1))],
    [-k1y*np.cos(θ1) + k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.8*k3x*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)) - 0.05*k3y*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1))]
    ], dtype=float)    
    
    
    
    return h_k

def posicion(theta1, theta2, theta3):
        
    posicion_partial = partial(posicion_k, theta1, theta2, theta3, 1, 1, 1, 1, 1, 1)

    return posicion_partial()



def H_k(θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3x, k3y):
    
    
    H = np.array([
            
            #m1
            [0.3*k1x*np.cos(θ1) + 0.05*k1y*np.sin(θ1), 0, 0, 0.3*np.sin(θ1), -0.05*np.cos(θ1), 0, 0, 0, 0],
            [0.3*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1), 0, 0, -0.3*np.cos(θ1), -0.05*np.sin(θ1), 0, 0, 0, 0],
            #m2
            [0.7*k1x*np.cos(θ1) + 0.05*k1y*np.sin(θ1), 0, 0, 0.7*np.sin(θ1), -0.05*np.cos(θ1), 0, 0, 0, 0],
            [0.7*k1x*np.sin(θ1) - 0.05*k1y*np.cos(θ1), 0, 0, -0.7*np.cos(θ1), -0.05*np.sin(θ1), 0, 0, 0, 0],        
            #m3
            [k1x*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)), -0.05*k2x*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.5*k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)), 0, np.sin(θ1), 0, 0.05*np.sin(θ1)*np.sin(θ2) + 0.05*np.cos(θ1)*np.cos(θ2), 0.5*np.sin(θ1)*np.cos(θ2) - 0.5*np.sin(θ2)*np.cos(θ1), 0, 0],
            [k1y*np.sin(θ1) - 0.05*k2x*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)), -0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.5*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)), 0, 0, -np.cos(θ1), -0.05*np.sin(θ1)*np.cos(θ2) + 0.05*np.sin(θ2)*np.cos(θ1), 0.5*np.sin(θ1)*np.sin(θ2) + 0.5*np.cos(θ1)*np.cos(θ2), 0, 0],         
            #m4
            [k1x*np.cos(θ1) - 0.05*k2x*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)), -0.05*k2x*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.9*k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)), 0, np.sin(θ1), 0, 0.05*np.sin(θ1)*np.sin(θ2) + 0.05*np.cos(θ1)*np.cos(θ2), 0.9*np.sin(θ1)*np.cos(θ2) - 0.9*np.sin(θ2)*np.cos(θ1), 0, 0],
            [k1y*np.sin(θ1) - 0.05*k2x*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)), -0.05*k2x*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.9*k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)), 0, 0, -np.cos(θ1), -0.05*np.sin(θ1)*np.cos(θ2) + 0.05*np.sin(θ2)*np.cos(θ1), 0.9*np.sin(θ1)*np.sin(θ2) + 0.9*np.cos(θ1)*np.cos(θ2), 0, 0],
            #m5
            [k1x*np.cos(θ1) + k2y*(np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2)) + 0.8*k3x*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)) - 0.05*k3y*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)), k2y*(-np.sin(θ1)*np.sin(θ2) - np.cos(θ1)*np.cos(θ2)) + 0.8*k3x*(-(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)) - 0.05*k3y*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)), 0.8*k3x*(-(-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)) - 0.05*k3y*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)), np.sin(θ1), 0, 0, np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1), 0.8*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - 0.8*(np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1), -0.05*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) - 0.05*(np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)],
            [k1y*np.sin(θ1) + k2y*(-np.sin(θ1)*np.cos(θ2) + np.sin(θ2)*np.cos(θ1)) + 0.8*k3x*(-(-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3y*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) - (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), k2y*(np.sin(θ1)*np.cos(θ2) - np.sin(θ2)*np.cos(θ1)) + 0.8*k3x*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3y*((-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), 0.8*k3x*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) + (-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)) - 0.05*k3y*((np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + (np.sin(θ2)*np.cos(θ3) - np.sin(θ3)*np.cos(θ2))*np.sin(θ1)), 0, -np.cos(θ1), 0, np.sin(θ1)*np.sin(θ2) + np.cos(θ1)*np.cos(θ2), 0.8*(-np.sin(θ2)*np.sin(θ3) - np.cos(θ2)*np.cos(θ3))*np.cos(θ1) + 0.8*(-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.sin(θ1), -0.05*(np.sin(θ2)*np.sin(θ3) + np.cos(θ2)*np.cos(θ3))*np.sin(θ1) - 0.05*(-np.sin(θ2)*np.cos(θ3) + np.sin(θ3)*np.cos(θ2))*np.cos(θ1)]
            ])


    return H

def H(theta1, theta2, theta3):
        
    H_partial = partial(H_k, theta1, theta2, theta3, 1, 1, 1, 1, 1, 1)
    H_partial_result = H_partial()
    H_partial_columns = H_partial_result[:, :3]
    
    # Creamos una matriz de ceros de dimensiones 3x10
    result = np.zeros((10, 3))

    return np.hstack((H_partial_columns, result)) 

def H_3(theta1, theta2, theta3):
        
    H_partial = partial(H_k, theta1, theta2, theta3, 1, 1, 1, 1, 1, 1)
    H_partial_result = H_partial()
    H_partial_columns = H_partial_result[:, :3]

    return H_partial_columns




def jacobian(params, x, y):
    
    θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3x, k3y = params
    
    return H_k(θ1, θ2, θ3, k1x, k1y, k2x, k2y, k3x, k3y)