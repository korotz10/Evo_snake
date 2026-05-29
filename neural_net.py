"""
Red neuronal feedforward manual implementada con NumPy.
Arquitectura: [11 inputs → 8 neuronas ocultas ReLU → 4 salidas Softmax]
Activación: ReLU en capa oculta, softmax en salida.
No usa backpropagation, los pesos son optimizados por el GA.
"""

import numpy as np


class NeuralNetwork:
    """
    Red neuronal feedforward simple con 1 capa oculta.
    
    Atributos:
        input_size: cantidad de neuronas de entrada (11 con nuevos sensores)
        hidden_size: cantidad de neuronas ocultas (8)
        output_size: cantidad de neuronas de salida (4)
        weights_ih: matriz de pesos entrada → oculta
        bias_h: bias de la capa oculta
        weights_ho: matriz de pesos oculta → salida
        bias_o: bias de la capa de salida
    """
    
    def __init__(self, input_size=11, hidden_size=8, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicializar pesos con distribución normal pequeña
        self.weights_ih = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_h = np.zeros((1, hidden_size))
        
        self.weights_ho = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_o = np.zeros((1, output_size))
    
    def forward(self, inputs):
        """
        Propaga los inputs hacia adelante.
        
        Args:
            inputs: array de forma (n_samples, input_size)
        
        Returns:
            output: array de forma (n_samples, output_size) con probabilidades
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Capa oculta: z = X @ W_ih + b_h
        z_h = np.dot(inputs, self.weights_ih) + self.bias_h
        
        # Activación ReLU
        h = np.maximum(0, z_h)
        
        # Capa de salida: z = h @ W_ho + b_o
        z_o = np.dot(h, self.weights_ho) + self.bias_o
        
        # Activación softmax
        output = self._softmax(z_o)
        
        return output
    
    @staticmethod
    def _relu(x):
        """Función de activación ReLU."""
        return np.maximum(0, x)
    
    @staticmethod
    def _softmax(x):
        """Función de activación softmax."""
        # Restar el máximo por estabilidad numérica
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def predict(self, inputs):
        """
        Realiza predicción y retorna la clase con mayor probabilidad.
        
        Args:
            inputs: array de forma (input_size,) o (n_samples, input_size)
        
        Returns:
            predictions: índices de las clases predichas
        """
        output = self.forward(inputs)
        return np.argmax(output, axis=1)
    
    def get_weights_flat(self):
        """
        Retorna todos los pesos como un vector 1D.
        Útil para el algoritmo genético.
        
        Returns:
            vector: todos los pesos concatenados
        """
        return np.concatenate([
            self.weights_ih.flatten(),
            self.bias_h.flatten(),
            self.weights_ho.flatten(),
            self.bias_o.flatten()
        ])
    
    def set_weights_flat(self, weights_flat):
        """
        Establece los pesos desde un vector 1D.
        Útil para el algoritmo genético.
        
        Args:
            weights_flat: vector con todos los pesos
        """
        idx = 0
        
        # Pesos entrada → oculta
        size_ih = self.input_size * self.hidden_size
        self.weights_ih = weights_flat[idx:idx + size_ih].reshape(
            self.input_size, self.hidden_size
        )
        idx += size_ih
        
        # Bias oculta
        size_bh = self.hidden_size
        self.bias_h = weights_flat[idx:idx + size_bh].reshape(1, self.hidden_size)
        idx += size_bh
        
        # Pesos oculta → salida
        size_ho = self.hidden_size * self.output_size
        self.weights_ho = weights_flat[idx:idx + size_ho].reshape(
            self.hidden_size, self.output_size
        )
        idx += size_ho
        
        # Bias salida
        self.bias_o = weights_flat[idx:].reshape(1, self.output_size)
    
    def get_total_params(self):
        """Retorna la cantidad total de parámetros de la red."""
        return (
            self.input_size * self.hidden_size +  # W_ih
            self.hidden_size +                      # b_h
            self.hidden_size * self.output_size +  # W_ho
            self.output_size                        # b_o
        )
    
    @staticmethod
    def create_from_weights(weights_flat, input_size=11, hidden_size=8, output_size=4):
        """
        Crea una red neuronal a partir de un vector de pesos.
        Útil para crear individuos en el GA.
        
        Args:
            weights_flat: vector con todos los pesos
            input_size: cantidad de entradas (default: 11)
            hidden_size: cantidad de neuronas ocultas (default: 8)
            output_size: cantidad de salidas (default: 4)
        
        Returns:
            NeuralNetwork: red neuronal con los pesos establecidos
        """
        net = NeuralNetwork(input_size, hidden_size, output_size)
        net.set_weights_flat(weights_flat)
        return net
