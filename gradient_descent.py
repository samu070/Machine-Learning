import numpy as np
import matplotlib.pyplot as plt

'''
This code is based on the Stanford's Machine Learning Specialization coursera's course by Andrew Ng.

It's a simple example of a single feature linear regression only with two feature values to
practice the implementation of an gradient descent algorithm

'''


x_train = np.array([  1,   2])
y_train = np.array([300, 500])

# Values obtained by the model
wopt = 200
bopt = 100

# Set random test values for w and b
w = np.random.randint(wopt - 90, wopt + 90)
b = np.random.randint(bopt - 90, bopt + 90)

def model(x, w, b):
    f = w*x + b
    return f

def cost_function(x, y, w, b):
    m = len(x)
    suma = 0
    for i in range(m):
        term = (model(x[i], w, b) - y[i])**2
        suma += term
    J = 1/(2*m)*suma
    return J

# Create a meshgrid in order to plot later "isolayers" of the cost function
warr = np.linspace(wopt - 100, wopt + 100, 1000)
barr = np.linspace(bopt - 100, bopt + 100, 1000)
W, B = np.meshgrid(warr, barr)



def grad_descent(x, y, w, b):
    
    def dJdw(x, y, w, b):
        m = len(x)
        suma = 0
        for i in range(m):
            term = (model(x[i], w, b) - y[i])*x[i]
            suma += term
        dJw = 1/m * suma
        return dJw
    
    def dJdb(x, y, w, b):
        m = len(x)
        suma = 0
        for i in range(m):
            term = (model(x[i], w, b) - y[i])
            suma += term
        dJb = 1/m * suma
        return dJb
        
    a = 0.3 # Learn rate  
    J_hist = [] # Registers the values of J for each iteration
    i = 0
    while abs(dJdw(x, y, w, b)) >= 0.0001 and abs(dJdb(x, y, w, b)) >= 0.0001:
        i += 1 # Iterations of the loop
        wnew = w - a*dJdw(x, y, w, b)
        bnew = b - a*dJdb(x, y, w, b)
        w = wnew
        b = bnew
        J_hist.append(cost_function(x, y, w, b))
        plot = plt.contourf(W, B, cost_function(x_train, y_train, W, B), levels=30)
        plt.plot(w, b, 'w.', label=r'$(w, b)$')
        plt.plot(wopt, bopt, 'r*', label=r'$(w_{opt}, b_{opt})$')
        plt.colorbar()
        plt.text(110, 185, r'$w = %.2f$' %w, color='w')
        plt.text(110, 175, r'$b = %.2f$' %b, color='w')
        plt.text(110, 165, r'$\alpha = %.1f$' %a, color='w')
        plt.text(110, 155, r'Iteration = %i' %i, color='w')
        plt.title(r'Cost function $J(w,b)$')
        plt.xlabel(r'w')
        plt.ylabel(r'b')
        plt.legend()
        plt.savefig('%i.png' %i)
        plt.pause(0.00000001)
        plt.clf()
    return[w, b, J_hist]
        

# If you want to plot J vs iterations
J = grad_descent(x_train, y_train, w, b)[2]
iterations = np.linspace(0, len(J), len(J))


plt.show()


    

