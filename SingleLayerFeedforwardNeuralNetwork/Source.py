import numpy as np


def activation(z, deriv=False):
    if deriv == True:
        return z * (1-z)
    return 1 / (1 + np.exp(-z))


def forwardpropagation(X, theta):
    return activation(np.dot(X, theta))


def backpropagation(y, theta, output, it):
    err = y - output
    if it % 10000 == 0:
        print 'Error: ' + str(np.mean(np.abs(err)))
    delta = err * activation(output, deriv=True)
    return delta


def gradientdescent(X, theta, delta):
    theta += X.T.dot(delta)
    return theta


def run():
    X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    np.random.seed(1)
    theta = 2 * np.random.random((3, 1)) - 1

    print 'Starting training'

    for j in xrange(100000):
        output = forwardpropagation(X, theta)
        delta =  backpropagation(y, theta, output, j)
        theta = gradientdescent(X, theta, delta)

    print 'Actual values'
    print y
    print 'Predictions after training'
    print output

if __name__ == '__main__':
    run()

