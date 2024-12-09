# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:53:41 2024

@author: diego
"""

import streamlit as st
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components

from streamlit_drawable_canvas import st_canvas
import pandas as pd


#from matplotlib.animation import FFMpegWriter
from matplotlib.animation import PillowWriter

import io
import tempfile


# -----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# Funciones auxiliares


def obtener_puntos_interpolar(puntos, distancia = "equidistante"):
    '''
    Usa los puntos  (x, y) para obtener los puntos (t, fx_t) y (t, fy_t) usados en la interpolacion
      
    Es decir, toma una matriz donde cada fila es (x, y)
    devuelve puntos usados para parametrizar fx y fy
    '''
    
    # las evaluaciones fx_t y fy_t son las coordenadas de los puntos en el plano
    fx_t = puntos[:, 0]
    fy_t = puntos[:, 1]
    
    # se agrega el primero al final para que "de vuelta"
    fx_t = np.hstack([fx_t, fx_t[0]])
    fy_t = np.hstack([fy_t, fy_t[0]])
    
    # si se tienen puntos en posiciones
    # 0, 1, ..., n
    # ver cual es n
    n = puntos.shape[0]
    
    # si se quieren nodos equidistantes
    if distancia == "equidistante":
        # nodos equidistantes {0, 1, ..., n}
        valores_t = np.arange(n+1)
    
    # si la distancia entre nodos depende de la distancia en el plano
    elif distancia == "plano":
        # ir haciendo la lista de nodos iterativamente, iniciar con el t_0 = 0
        valores_t = [0]
        
        # por cada punto i (1, ..., n-1)
        for i in range(1, n):
            # ver la distancia del punto i-1 al punto i
            punto_anterior = puntos[i-1, :]
            punto_actual = puntos[i, :]
            distancia_actual = np.linalg.norm(punto_anterior - punto_actual)
    
            # agregar el nodo i a esa distancia del nodo i-1
            valores_t.append(valores_t[-1] + distancia_actual)
        # end for sobre los puntos
    
        # agregar el ultimo nodo segun la distancia del ultimo al primero
        valores_t.append(valores_t[-1] + np.linalg.norm(puntos[-1, :] - puntos[0, :]))
    
    # si se quiere otro metodo
    else:
        raise ValueError("Distancia no reconocida")
    
    # re escalar los puntos para que el maximo sea 2pi
    # y para ambos metodos el primero es 0
    # entonces lo que queda es algo periodico en 2pi (pues inicia donde termina)
    valores_t = valores_t / np.max(valores_t) * 2*np.pi
    
    # ordenar resultados
    res = {
        "t": valores_t,
        "fx_t": fx_t,
        "fy_t": fy_t
        }
    return res
    


class Lagrange_interpolation:
    '''
    Interpolacion usando un polinomio (bases de Lagrange)
    '''
    def __init__(self, valores_t, valores_f_t):
        # valores de t:   t_0,   t_1, ...,    t_n
        # evaluaciones: f(t_0), f(t_1), ..., f(t_n)
    
        # poner como atributos
        self.valores_t = valores_t
        self.valores_f_t = valores_f_t
        assert len(valores_t) == len(valores_f_t)
    
        # ver cual es n
        self.n = len(valores_t) - 1
    
        # delimitar el conjunto de indices
        # 0, 1, ..., n
        self.indices = set(range(self.n+1))
    
        # por cada par de nodos (t_j, t_k) calcular t_j - t_k
        self.resta_nodos = {(j, k): self.valores_t[j] - self.valores_t[k]
                           for j in self.indices for k in self.indices}
    
    # hacer las bases
    def base_lagrange(self, index, t):
        '''
        Calcular la base index evaluada en t
        '''
        res = 1 # ir evaluadno
    
        # por cada k que no es el indice de la base
        for k in self.indices - {index}:
            # añadir el termino correspondiente
            res = res * ((t - self.valores_t[k]) / self.resta_nodos[(index, k)])
        return res
    
    # evaluar el polinomio de interpolacion en un valor
    def evaluar_P_single_t(self, t):
        '''
        Evaluar el polinomio de interpolacion en un valor t
        '''
        res = 0 # ir evaluando
        # por cada indice
        for k in self.indices:
            # poner el termino correspondiente
            res = res + self.valores_f_t[k] * self.base_lagrange(index = k, t = t)
        return res
    
    # evaluar en varios
    def interpolar(self, array_t):
        '''
        Evaluar el polinomio de interpolacion en un array de valores t
        '''
        return np.array([self.evaluar_P_single_t(t) for t in array_t])
    
    # ver la interpolacion
    def ver_interpolacion(self, num_valores = 500, titulo = "Interpolacion", ax = None):
        '''
        Ver la interpolacion
        '''
    
        # dominio para la interpolacion
        dominio = np.linspace(self.valores_t[0], self.valores_t[-1], num_valores)
        # interpolacion
        interpolacion = self.interpolar(dominio)
          
        # graficar
        if ax is None:
            fig, ax = plt.subplots(figsize = (5, 5))
        ax.scatter(self.valores_t, self.valores_f_t, label = "Puntos", color = color_puntos)
        ax.plot(dominio, interpolacion, label = "Interpolacion", color = color_interpolacion)
        ax.set_title(titulo)



class CubicSpline_interpolation:
    '''
    Interpolacion usando splines cubicos
    '''
    def __init__(self, valores_t, valores_f_t, boundary_condition = "natural"):
        # valores de t:   t_0,   t_1, ...,    t_n
        # evaluaciones: f(t_0), f(t_1), ..., f(t_n)
        # boundary_condition: natural, clamped, not-a-knot, periodic
        
        # poner como atributos
        self.valores_t = valores_t
        self.valores_f_t = valores_f_t
        assert len(valores_t) == len(valores_f_t)
        
        # ver cual es n
        self.n = len(valores_t) - 1
        
        # usar la interpolacion de scipy para el cubic spline
        self.cs = CubicSpline(valores_t, valores_f_t,
                              bc_type= boundary_condition, extrapolate=False)
    
    
    # evaluar en varios
    def interpolar(self, array_t):
        '''
        Evaluar el cubic spline de interpolacion en un array de valores t
        '''
        return self.cs(array_t)
    
    # ver la interpolacion
    def ver_interpolacion(self, num_valores = 500, titulo = "Interpolacion", ax = None):
        '''
        Ver la interpolacion
        '''
          
        # dominio para la interpolacion
        dominio = np.linspace(self.valores_t[0], self.valores_t[-1], num_valores)
        # interpolacion
        interpolacion = self.interpolar(dominio)
          
        # graficar
        if ax is None:
            fig, ax = plt.subplots(figsize = (5, 5))
        ax.scatter(self.valores_t, self.valores_f_t, label = "Puntos", color = color_puntos)
        ax.plot(dominio, interpolacion, label = "Interpolacion", color = color_interpolacion)
        ax.set_title(titulo)



def integrate_function(function, a = 0, b = 2*np.pi):
    '''
    Integra una funcion en [a, b]
    '''
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad
    if metodo_integracion == "quad":
        res = integrate.quad(function, a, b, limit = 150)
        return res[0]
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.fixed_quad.html#scipy.integrate.fixed_quad
    elif metodo_integracion == "fixed_quad":
        res = integrate.fixed_quad(function, a, b, n = n_fixed_quad)
        return res[0]
    
    # otro metodo
    else:
        raise ValueError("Metodo de integracion no reconocido")



# clase que haga analisis de fourier de una funcion real
class Fourier_analysis_real:
    '''
    Analisis de fourier de una funcion real f en [0, 2*pi]
    f(t) = a_0 / 2  + sum_{n=1}^{infinito} (a_n * cos(nt) + b_n * sin(nt))
    
    donde:
    a_0 = 1/pi * integral en [0, 2*pi] de: f(t) dt
    a_n = 1/pi * integral en [0, 2*pi] de: f(t) cos(nt) dt
    b_n = 1/pi * integral en [0, 2*pi] de: f(t) sin(nt) dt
    '''
    
    def __init__(self, funcion_f, n_coeficientes = 10):
        # poner como atributos
        self.funcion_f = funcion_f
          
        # inicar coeficientes vacio
        self.a0 = None
        self.an = dict()
        self.bn = dict()
          
        # calcular a0
        self.compute_a0()
        # calcular tantos coeficientes como se quierea
        self.compute_n_coeficients(n_coeficientes)
    
    # calcular a0
    def compute_a0(self):
        # poner como atributo
        self.a0 = (1/np.pi) * integrate_function(self.funcion_f, a = 0, b = 2*np.pi)
    
    # compute an bn
    def compute_n_coeficients(self, n_coeficientes):
        # calcular tantos coeficientes como se quierea
          
        # desde 1 hasta n
        for n in range(1, n_coeficientes+1):
            # calcular an y bn si es que hacen falta en el diccionario
            
            # calcular an
            if n not in self.an.keys():
                self.an[n] = (1/np.pi) * integrate_function(lambda t: self.funcion_f(t) * np.cos(n*t), a = 0, b = 2*np.pi)
            
            # calcular bn
            if n not in self.bn.keys():
                self.bn[n] = (1/np.pi) * integrate_function(lambda t: self.funcion_f(t) * np.sin(n*t), a = 0, b = 2*np.pi)
    
    # devolver coeficientes
    def get_coefficients(self):
        # devolver a0, an, bn
        return {"a0": self.a0,
                "an": self.an,
                "bn": self.bn}
    
    # visualizar descomposicion
    def visualizar_n_componentes(self, n_visualizar = 4, puntos_dominio = 1000, title = "Fourier series of f", ax = None):
        '''
        Visualizar los n_visualizar primeros componentes
        '''
          
        # dominio donde graficar la funcion, puntos en [0, 2*pi]
        dominio = np.linspace(0, 2*np.pi, puntos_dominio)
          
        # evaluar la funcion
        f_t = self.funcion_f(dominio)
          
        # serie de fourier
        f_t_fourier = np.ones_like(dominio) * self.a0 / 2   # inicar con a0
        for n in range(1, n_visualizar+1):
            # agregar an y bn
            f_t_fourier = f_t_fourier +  self.an[n] * np.cos(n*dominio) + self.bn[n] * np.sin(n*dominio)
          
        # graficar
        if ax is None:
            fig, ax = plt.subplots(figsize = (8, 5))
        ax.plot(dominio, f_t, label = "Funcion original", color = color_interpolacion)
        ax.plot(dominio, f_t_fourier, label = "Fourier series", color = color_fourier)
        ax.set_title(title + f" - max n = {n_visualizar}")
        
          
    
# clase que haga analisis de fourier de una funcion compleja
class Fourier_analysis_complex:
    '''
    Analisis de fourier de una funcion compleja f(t) en [0, 2*pi]
    f = f_1 + i * f_2
    con f_1, f_2 funciones reales
    
    f(t) = sum_{n = - infinito}^{infinito} c_n e^{int}
    donde
    cn = 1/2*pi integral en [0, 2*pi] de: f(t) e^{-int} dt
    
    se sabe que:
    c_{0} =  (1/2)(a0_1 + a0_2)
    para n natural: 
    c_{n}  = (1/2)(an_1 + bn_2) + i(1/2)(an_2 - bn_1) 
    c_{-n} = (1/2)(an_1 - bn_2) - i(1/2)(an_2 + bn_1)
    '''
    
    def __init__(self, funcion_1, funcion_2, n_coeficientes = 10):
        # poner como atributos la funcion real e imaginaria
        self.funcion_1 = funcion_1
        self.funcion_2 = funcion_2
          
        # inicar coeficientes vacio
        self.cn = dict()
          
        # hacer series de fourier en las funciones f1 y f2
        fourier_1 = Fourier_analysis_real(funcion_1, n_coeficientes = n_coeficientes)
        fourier_2 = Fourier_analysis_real(funcion_2, n_coeficientes = n_coeficientes)
        # guardar sus coeficientes
        self.coef_f1 = fourier_1.get_coefficients()
        self.coef_f2 = fourier_2.get_coefficients()
          
        # ahora calcular los coeficientes cn
        self.compute_n_coeficients(n_coeficientes)
    
    
    # calcular cn
    def compute_n_coeficients(self, n_coeficientes):
        '''
        Dado un N, se calculan los coeficientes
        c_{-N}, ..., c_{N}
        '''
          
        assert n_coeficientes >= 0
          
        # poner c0 si no se ha calculado
        if 0 not in self.cn.keys():
            # c_{0} =  (1/2)(a0_1 + a0_2)
            self.cn[0] = (0.5)* (self.coef_f1["a0"] + 1j * self.coef_f2["a0"])
          
        # intentar calculas los coeficientes c_n
        for n in range(1, n_coeficientes+1):
            # agregar c_{n} y c_{-n}
            # solo si no se han calculado
            
            # c_{n}
            if n not in self.cn.keys():
                # c_{n}  = (1/2)(an_1 + bn_2) + i(1/2)(an_2 - bn_1)
                self.cn[n] = (0.5)*(self.coef_f1["an"][n] + self.coef_f2["bn"][n]) + 1j * (0.5)*(self.coef_f2["an"][n] - self.coef_f1["bn"][n])
            
            # c_{-n}
            if -n not in self.cn.keys():
                # c_{-n} = (1/2)(an_1 - bn_2) - i(1/2)(an_2 + bn_1)
                self.cn[-n] = (0.5)*(self.coef_f1["an"][n] - self.coef_f2["bn"][n]) + 1j * (0.5)*(self.coef_f2["an"][n] + self.coef_f1["bn"][n])
    
    
    # devolver los coeficientes
    def get_coefficients(self):
        return self.cn
    
    
    # ver la serie de fourier
    def visualizar_n_componentes(self, n_visualizar = 4, puntos_dominio = 1000, title = "Fourier series of f", ax = None):
        '''
        Visualizar los n_visualizar primeros componentes
        '''
      
        # dominio donde graficar la funcion, puntos en [0, 2*pi]
        dominio = np.linspace(0, 2*np.pi, puntos_dominio)
      
        # evaluar la funcion f
        self.f = self.funcion_1(dominio) + 1j * self.funcion_2(dominio)
      
        # serie de fourier
        f_t_fourier = np.ones_like(dominio) * self.cn[0]   # inicar con c0
        for n in range(1, n_visualizar+1):
            # agregar c_{n} y c_{-n}
            f_t_fourier = f_t_fourier +  self.cn[n] *  np.exp(1j * n*dominio)
            f_t_fourier = f_t_fourier +  self.cn[-n] * np.exp(1j *(-n)*dominio)
      
        # graficar
        if ax is None:
            fig, ax = plt.subplots(figsize = (5, 5))
        ax.plot(self.funcion_1(dominio), self.funcion_2(dominio), label = "Funcion original", color = color_interpolacion)
        ax.plot(f_t_fourier.real, f_t_fourier.imag, label = "Fourier series", color = color_fourier)
        ax.set_title(title + f" - max abs n = {n_visualizar}")
 
        
 
def puntos_imagen_2_puntos_grafica(coordenadas_imagen):
    '''
    Dado un arreglo nx2 de coordenadas en la imagen
    transofrmar a escala normal para usar despeus
    '''
    
    # la imagen se asume de 600x300
    
    # separar en coord x, y de la imagen
    coord_x_image = coordenadas_imagen[:, 0]
    coord_y_image = coordenadas_imagen[:, 1]
    
    # transformar coordenadas x
    # de 0 - 600 al intervalo [a, b] que es x_lim_graficas
    a = x_lim_graficas[0]
    b = x_lim_graficas[1]
    coord_x_image = coord_x_image/600 * (b-a) + a

    # transformar coordenadas y
    # de 0 - 300 al intervalo [a, b] que es y_lim_graficas
    # tambien se le da la vuelta
    a = y_lim_graficas[0]
    b = y_lim_graficas[1]
    coord_y_image = (300 - coord_y_image)/300 * (b-a) + a
    
    # unir para tomar coordenadas en la grafica
    coordenadas_grafica = np.array([coord_x_image, coord_y_image]).T
    return coordenadas_grafica
        


def obtener_puntos(commands):
    '''
    Dados los comandos de un path
    obtener puntos por los que pasa
    '''
    puntos = []
    for command in commands:
        if command[0] == 'M':  # Comando de movimiento
            puntos.append((command[1], command[2]))
        elif command[0] == 'Q':  # Comando de curva Bézier cuadrática
            puntos.append((command[1], command[2]))  # Punto inicial de la curva
            puntos.append((command[3], command[4]))  # Punto final de la curva
        elif command[0] == 'L':  # Comando de línea recta
            puntos.append((command[1], command[2]))  # Punto final de la línea
    return puntos
    




# funcion para hacer la animacion
def crear_animacion():
    '''
    Crear la animacion final
    '''

    # valores de t para la animacion
    t_vals = np.linspace(0, 2*np.pi, frames_amimacion) 
    
    # definir un color para cada flecha
    vector_colors = plt.cm.viridis(np.linspace(0, 1, len(coeficientes_fourier_f)))
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # cosas de estilo
    ax.set_xlim(x_lim_graficas[0], x_lim_graficas[1])
    ax.set_ylim(y_lim_graficas[0], y_lim_graficas[1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect("equal")
    
    # graficar la original (interpolacion mas bien)
    ax.plot(fx_interpolacion_grafica, fy_interpolacion_grafica, color = color_interpolacion)
    
    # ver cuantas flechas hay (una por cada termino cn)
    num_flechas = len(coeficientes_fourier_f)
    # al principio, todas estan totalmente en el origen (0 + i0)
    x_start = np.zeros(num_flechas)
    y_start = np.zeros(num_flechas)
    x_end = np.zeros(num_flechas)
    y_end = np.zeros(num_flechas)
    
    # inicar las flechas, de entrada todas en el origen, con tamaño 0
    quiver = ax.quiver(x_start, y_start, x_end, y_end, angles='xy', 
                       scale_units='xy', scale=1, color=vector_colors,
                       headwidth = 2.5, headlength = 3, headaxislength = 2)
    
    # punto al final de todas las flechas
    punto_final, = ax.plot([], [], marker='o', color= color_fourier)
    # trayectoria marcada por la serie de fourier
    posiciones_trayectoria_fourier = []
    trayectoria_fourier, = ax.plot([], [], '-', color=color_fourier)

    
    
    # funcion que hace update a la animacion
    def update(frame):
        """Actualizar la animación en cada frame."""
        
        # ver cual es el valor t de ese frame
        t = t_vals[frame]
        
        # variables de interes:
        
        f_fourier_t = 0   # suma de todas las flechas
        # Puntos iniciales de flechas
        # la primera flecha inicia en el origen
        x_start = [0]  
        y_start = [0]
        # Puntos finales de flechas
        x_end = []  
        y_end = []
        
        # por cada fleca (termino c_n e^{int})
        for n, c_n in coeficientes_fourier_f.items():
        
            # calcular c_n e^{int}
            termino_actual_n = c_n * np.exp(1j * n * t)
          
            # agregar a la suma de todas las flechas
            f_fourier_t += termino_actual_n
          
            # la flecha actual termina en esta posicion 
            x_end.append(f_fourier_t.real)
            y_end.append(f_fourier_t.imag)
            # y ahi mismo inicia la ultima flecha
            x_start.append(f_fourier_t.real)
            y_start.append(f_fourier_t.imag)
        
        # end for terminos cn
        
        # eliminar el ultimo final por redundancia
        x_start.pop()  
        y_start.pop()
        # actualizar la posicion de las flechas
        quiver.set_offsets(np.column_stack([x_start, y_start]))
        quiver.set_UVC(np.array(x_end) - np.array(x_start), np.array(y_end) - np.array(y_start))
        
        
        # poner el punto final en la posicion calculada, y agregarla a todas las de la taryectoria
        punto_final.set_data([f_fourier_t.real], [f_fourier_t.imag])
        posiciones_trayectoria_fourier.append((f_fourier_t.real, f_fourier_t.imag))
        # actualizar la trayectoria
        trayectoria_fourier.set_data(*zip(*posiciones_trayectoria_fourier))
        
        return quiver, punto_final, trayectoria_fourier
    
    # Crear la animación
    animacion_fourier = FuncAnimation(fig, update, frames= frames_amimacion, blit=False, interval=50)
    # devolver
    return animacion_fourier


# -----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------   
# -----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------



# Constantes
x_lim_graficas = [-6, 6]
y_lim_graficas = [-3, 3]
figsize = (6, 3)



# main
st.title('Complex Fourier Series Visualization')



## Delimitar puntos dibujando


# ver si se quieren seleccionar puntos o dibujar libremente
modo_input = st.selectbox(label = "Modo de especificar la curva:",
                               options = ["Dibujar", "Delimitar puntos"])
  
# ver cual es el drawing mode
drawing_mode = {"Delimitar puntos": "point", "Dibujar" : "freedraw"}[modo_input]


# si es dibujar, pedir el numero de puntos
if drawing_mode == "freedraw":
    num_puntos = st.slider("Numero de puntos en la curva a considerar",
                           min_value= 3, max_value= 200,
                           value=20, step=1)


# seleccionar el color para dibujar
color_dibujar = st.color_picker("Color para dibujar: ")



#  hacer el espacio para dibujar
canvas_result = st_canvas(
    stroke_width= 4,
    stroke_color= color_dibujar,
    background_color= "#eee", # por decir algo
    background_image= None,
    update_streamlit= True,
    height= 300,
    width  = 600,
    drawing_mode= drawing_mode,
    point_display_radius = 2,
    key="canvas",
)


# tomar la info dibujada 
if canvas_result.json_data is not None:
    # hacer df
    objects = pd.json_normalize(canvas_result.json_data["objects"])
# hacer objects vacio
else:
    objects = pd.DataFrame()
    
        

# si es que si hay df con cosas
if objects.shape[0] > 0:
    
    # tomar info de interes

    
    # si se van a poner puntos
    if modo_input == "Delimitar puntos":
        
        # tomar todos los puntos de canvas
        df_puntos = objects[objects["type"] == "circle"]
        
        # tomar las coordenadas en la imagen de los puntos
        coord_x_imagen = df_puntos["left"].values
        coord_y_imagen = df_puntos["top"].values
        puntos_imagen = np.array([coord_x_imagen, coord_y_imagen]).T
        
        # transformar a puntos con las unidades de matplotlib
        puntos = puntos_imagen_2_puntos_grafica(puntos_imagen)
        
        
    # si se va a dibujar
    elif modo_input == "Dibujar":
        
        # tomar todos los path de canvas
        df_path = objects[objects["type"] == "path"]
        
        # tomar solo el primero y tomar su path
        info_dibujo = df_path.iloc[0, :]
        path_dibujo = info_dibujo["path"]
        
        # segun la info del path, tomar puntos en la trayectoria
        puntos_trayectoria = np.array(obtener_puntos(path_dibujo))
        
        # ahora solo tomar unos cuantos puntos de estos
        
        # los indices de los que se van a seleccionar
        indices_puntos = np.linspace(0, len(puntos_trayectoria) - 1,
                                     num_puntos, dtype=int)
        
        # tomar estos puntos
        puntos_imagen = puntos_trayectoria[indices_puntos]
        
        # transformar a puntos con las unidades de matplotlib
        puntos = puntos_imagen_2_puntos_grafica(puntos_imagen)
    

# ------------------------------------------------------------------
# ------------------------------------------------------------------
## Parametros


with st.expander("Set parameters"):
    st.subheader("Parametros")
    
    # distancia entre nodos para la parametrizacion
    metodo_dist_t_largo = st.selectbox(label = "Metodo para seleccionar distancia entre puntos al parametrizar con interpolación",
                                       options = ["Misma distancia", "Distancia en el plano"])
    metodo_dist_t = {"Misma distancia":"equidistante", "Distancia en el plano":"equidistante"}[metodo_dist_t_largo]
    
    # metodo interpolacion
    metodo_interpolacion_largo = st.selectbox(label = "Metodo para hacer interpolacion",
                                              options = ["Cubic Spline", "Un polinomio (Lagrange interpolation)"])
    metodo_interpolacion = {"Cubic Spline": "cubicspline", "Un polinomio (Lagrange interpolation)": "lagrange"}[metodo_interpolacion_largo]
    
    # talvez condiciones de frontera
    if metodo_interpolacion == "cubicspline":
        bound_condition = st.selectbox(label = "Boundary conditions for the cubic splines:",
                                       options = ["Periodic", "Not-a-Knot", "Clamped", "Natural"])
        bound_condition = bound_condition.lower()
    
    # metodo integracion
    metodo_integracion_largo = st.selectbox(label = "Metodo para hacer integracion",
                                            options = ["Quad", "Fixed-order Gaussian quadrature"])
    metodo_integracion = {"Quad": "quad", "Fixed-order Gaussian quadrature": "fixed_quad"}[metodo_integracion_largo]
    
    # talvez orden para Gaussian quadrature
    if metodo_integracion == "fixed_quad":
        n_fixed_quad = st.slider("Order of Gaussian quadrature",
                                 min_value= 10, max_value=60,
                                 value=30, step=1)
    
    
    # max n fourier
    n_fourier = st.slider("Max n for coeficients in fourier series",
                          min_value= 0, max_value= 80,
                          value=20, step=1)
    
    
    # numero de valores para graficar
    num_valores_grafica = st.slider("Number of points used to make the graphs",
                                    min_value = 10, max_value = 500,
                                    value = 200, step = 1)
    
    # frames en anumacion
    frames_amimacion = st.slider("Number of frames in animation",
                                 min_value = 10, max_value = 500,
                                 value = 200, step = 1)
    
    
    # ver si se va a querer descargar
    descargar_animacion = st.checkbox(label = "Descargar la animacion", value = True)
    
    # si se quiere descargar
    if descargar_animacion:
        # segundos para el video a descargar
        segundos_animacion = st.slider("Duracion en segundos de la animacion (para descargar descargar)",
                                 min_value= 2, max_value=120,
                                 value=10, step=1)
    
    
    # colores
    color_puntos = st.color_picker(label = "Color for the points", value  = "#FF0000")
    color_fourier = st.color_picker(label = "Color for the fourier series", value  = "#00FF00")
    color_interpolacion = st.color_picker(label = "Color for the interpolation", value  = "#0000FF")

    


# ------------------------------------------------------------------
# ------------------------------------------------------------------


# analisar hasta que se pida
if st.button("Analyze", type="primary"):
    
    # poner todo el analisis en un expander
    with st.expander("View analysis"):
        st.subheader("Analysis")
        

        # Ver los puntos
        fig, ax = plt.subplots(figsize = figsize)
        ax.scatter(puntos[:, 0], puntos[:, 1], color = color_puntos)
        
        # cosas de estilo
        ax.set_xlim(x_lim_graficas[0], x_lim_graficas[1])
        ax.set_ylim(y_lim_graficas[0], y_lim_graficas[1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title("Puntos")
        ax.set_aspect("equal")
        # mostrar
        plt.tight_layout()
        st.pyplot(fig, use_container_width = True)
        
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        
        ## Interpolar las funciones fx y fy
        
        # obtener los puntos a interpolar
        res_puntos_interpolar = obtener_puntos_interpolar(puntos, distancia = metodo_dist_t)
        valores_t = res_puntos_interpolar["t"]
        valores_fx_t = res_puntos_interpolar["fx_t"]
        valores_fy_t = res_puntos_interpolar["fy_t"]

        # interpolar x y y
        if metodo_interpolacion == "lagrange":
            interpolate_x = Lagrange_interpolation(valores_t, valores_fx_t)
            interpolate_y = Lagrange_interpolation(valores_t, valores_fy_t)
        elif metodo_interpolacion == "cubicspline":
            interpolate_x = CubicSpline_interpolation(valores_t, valores_fx_t, boundary_condition = bound_condition)
            interpolate_y = CubicSpline_interpolation(valores_t, valores_fy_t, boundary_condition = bound_condition)
        else:
            raise ValueError("Metodo de interpolacion no reconocido")
            
        # ver la interpolacion 
        
        fig, ax = plt.subplots(2, 1, figsize = (6, 5), sharex=True)
        
        interpolate_x.ver_interpolacion(titulo = "Interpolacion en x", ax = ax[0])
        interpolate_y.ver_interpolacion(titulo = "Interpolacion en y", ax = ax[1])
        
        
        # cosas de estilo
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        # cosas de estilo
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        
        # mostrar
        plt.tight_layout()
        st.pyplot(fig, use_container_width = True)
        
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        
        # Ver la interpolacion completa

        # dominio de t para hacer la grafica
        dom_t_grafica = np.linspace(valores_t[0], valores_t[-1], num_valores_grafica + 1) # +1 por la periodicidad en la grafica
        # iterpolar fx y fx
        fx_interpolacion_grafica = interpolate_x.interpolar(dom_t_grafica)
        fy_interpolacion_grafica = interpolate_y.interpolar(dom_t_grafica)
        
         
        # ver
        fig, ax = plt.subplots(figsize = figsize)
        ax.plot(fx_interpolacion_grafica, fy_interpolacion_grafica, color = color_interpolacion)
        ax.scatter(puntos[:, 0], puntos[:, 1], color = color_puntos)
        
        # cosas de estilo
        ax.set_xlim(x_lim_graficas[0], x_lim_graficas[1])
        ax.set_ylim(y_lim_graficas[0], y_lim_graficas[1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title("Puntos y su interpolación")
        ax.set_aspect("equal")
        # mostrar
        plt.tight_layout()
        st.pyplot(fig, use_container_width = True)
        
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------

        ## Fourier de fx y fy
        
        # fourier de x
        fourier_x = Fourier_analysis_real(interpolate_x.interpolar, n_coeficientes = n_fourier)        
        # fourier de y
        fourier_y = Fourier_analysis_real(interpolate_y.interpolar, n_coeficientes = n_fourier)
        
        # imrpimir
        st.write("Fourier coeficients for fx:")
        st.write(fourier_x.get_coefficients())
        st.write("Fourier coeficients for fy:")
        st.write(fourier_y.get_coefficients())
                
        # ver
        fig, ax = plt.subplots(2, 1, figsize = (6, 5), sharex=True)
        
        fourier_x.visualizar_n_componentes(n_visualizar = n_fourier, title = "Fourier series of fx", ax=ax[0])
        fourier_y.visualizar_n_componentes(n_visualizar = n_fourier, title = "Fourier series of fy", ax=ax[1])
                
        # cosas de estilo
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        # cosas de estilo
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        
        # mostrar
        plt.tight_layout()
        st.pyplot(fig, use_container_width = True)
        
        
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        
        ## Fourier de f
        
        # hacer el analisis
        fourier_f = Fourier_analysis_complex(interpolate_x.interpolar, interpolate_y.interpolar, n_coeficientes = n_fourier)
        # sacar coeficientes
        coeficientes_fourier_f = fourier_f.get_coefficients()
        
        # imprimir
        st.write("Fourier coeficients for fx:")
        st.write(coeficientes_fourier_f)
            
        # ver
        fig, ax = plt.subplots(figsize = figsize)
        
        fourier_f.visualizar_n_componentes(n_visualizar = n_fourier, title = "Fourier series of f", ax=ax)
                
        # cosas de estilo
        ax.set_xlim(x_lim_graficas[0], x_lim_graficas[1])
        ax.set_ylim(y_lim_graficas[0], y_lim_graficas[1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect("equal")
        # mostrar
        plt.tight_layout()
        st.pyplot(fig, use_container_width = True)

     
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
    
    # end expander del analisis
    
    
    # poner la animacion
    st.subheader("Animación")
    animacion_mostrar = crear_animacion()
    components.html(animacion_mostrar.to_jshtml(),  height = 500)
    
    # si se quiere descargar la animacion
    if descargar_animacion:
        
        # hacer una animacion para descargar
        anim_descargar = crear_animacion()
    
        # calcular los fps
        fps_animacion = int(frames_amimacion/segundos_animacion)
    
        # un truquito para poder descargar la animacion
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False, dir=".") as tmpfile:
            
            #writer = FFMpegWriter(fps=fps_animacion)
            #anim_descargar.save(tmpfile.name, writer=writer)
            
            writer = PillowWriter(fps=fps_animacion)
            anim_descargar.save(tmpfile.name, writer=writer)
            
            tmpfile.seek(0)  # Move to the beginning of the file
            # Load the content into a BytesIO object
            buffer = io.BytesIO(tmpfile.read())
        
        # un boton para poder descargar la animacion
        st.download_button(
            label="Download Animation",
            data=buffer,
            file_name="animation.gif",
            mime="video/gif"
        )























