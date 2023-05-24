# Activation Functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class AF:
  def RELU(x):
    return np.maximum(0, x)

  def tanh(x):
    return np.tanh(x)

  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  def softmax(x):
    exp_vals = np.exp(x - np.max(x))
    return exp_vals / np.sum(exp_vals,axis=0)

  def step(x):
    return np.heaviside(x, 1)

  def signum(x):
    return (x+1) 

x= np.linspace(-10, 10)

while True:
  n = int(input('''Enter your choice: \n
                   1. RELU \n
                   2. tanh \n
                   3. Sigmoid \n
                   4. Softmax \n
                   5. Step_fucntion \n
                   6. Signum \n
                   0. EXIT \n\n\t\t'''))
  
  if(n == 0):
    print('Thank you!')
    break

  if(n == 1):
    plt.plot(x, AF.RELU(x))
    plt.title("RELU")
    plt.show()
  elif(n ==2):
    plt.plot(x, AF.tanh(x))
    plt.title("Tanh")
    plt.show()
  elif(n ==3):
    plt.plot(x, AF.sigmoid(x))
    plt.title("Sigmoid")
    plt.show()
  elif(n ==4):
    plt.plot(x, AF.softmax(x))
    plt.title("Softmax")
    plt.show()
  elif(n ==5):
    plt.plot(x, AF.step(x))
    plt.title("Step_function")
    plt.show()
  elif(n ==6):
    plt.plot(x, AF.signum(x))
    plt.title("RELU")
    plt.show() 
  else:
    print("Invalid Input") 