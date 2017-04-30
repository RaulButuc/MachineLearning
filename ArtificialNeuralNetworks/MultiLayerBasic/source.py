import numpy as np


def activation(z, deriv=False):
    if deriv == True:
        return z * (1-z)
    return 1 / (1 + np.exp(-z))


def forwardpropagation(X, theta0, theta1):
    layer0 = X
    layer1 = activation(np.dot(layer0, theta0))
    layer2 = activation(np.dot(layer1, theta1))
    return layer0, layer1, layer2


def backpropagation(y, theta1, layer1, layer2, it):
    err2 = y - layer2
    if it % 10000 == 0:
        print 'Error: ' + str(np.mean(np.abs(err2)))
    delta2 = err2 * activation(layer2, deriv=True)
    err1 = delta2.dot(theta1.T)
    delta1 = err1 * activation(layer1, deriv=True)
    return delta1, delta2


def gradientdescent(theta0, theta1, layer0, layer1, delta1, delta2):
    theta1 += layer1.T.dot(delta2)
    theta0 += layer0.T.dot(delta1)
    return theta0, theta1


def run():
    X = np.array([[0, 0, 1], [ 0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    np.random.seed(1)
    theta0 = 2 * np.random.random((3, 4)) - 1
    theta1 = 2 * np.random.random((4, 1)) - 1

    print 'Starting training'

    for j in xrange(100000):
        layer0, layer1, layer2 = forwardpropagation(X, theta0, theta1)
        delta1, delta2 =  backpropagation(y, theta1, layer1, layer2, j)
        theta0, theta1 = gradientdescent(theta0, theta1, layer0, layer1, delta1, delta2)

    print 'Actual values'
    print y
    print 'Predictions after training'
    print layer2

if __name__ == '__main__':
    run()

