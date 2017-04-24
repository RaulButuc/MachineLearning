from numpy import loadtxt, where, zeros, e, array, log, ones, mean, where
from pylab import scatter, show, legend, xlabel, ylabel, plot
from scipy.optimize import fmin_bfgs

class MLHelper:


    def __init__(self):
        pass


    @staticmethod
    def sigmoid(X):
	return 1.0 / (1.0 + e ** (-1.0 * X))

class Admission:

    data = None
    X = None
    y = None
    pos = None
    neg = None
    m = None
    n = None
    it = None


    def __init__(self):
        self.data = loadtxt('data.csv', delimiter=',')
        self.X = self.data[:, 0:2]
        self.y = self.data[:, 2]
        self.pos = where(self.y == 1)
        self.neg = where(self.y == 0)
        scatter(self.X[self.pos, 0], self.X[self.pos, 1], marker='o', c='b')
        scatter(self.X[self.neg, 0], self.X[self.neg, 1], marker='x', c='r')
        xlabel('Exam 1 score')
        ylabel('Exam 2 score')
        legend(['Admitted', 'Not Admitted'])
        self.m, self.n = self.X.shape
        self.y.shape = (self.m, 1)
        self.it = ones(shape=(self.m, 3))
        self.it[:, 1:3] = self.X


    @staticmethod
    def compute_cost(theta, X, y):
        theta.shape = (1, 3)
        m = y.size
        h = MLHelper.sigmoid(X.dot(theta.T))
        J = (1.0 / m) * ((-y.T.dot(log(h))) - ((1.0 - y.T).dot(log(1.0 - h))))
        return -1 * J.sum()


    @staticmethod
    def compute_grad(theta, X, y, m):
        theta.shape = (1, 3)
        grad = zeros(3)
        h = MLHelper.sigmoid(X.dot(theta.T))
        delta = h - y
        l = grad.size
        for i in range(l):
            sumdelta = delta.T.dot(X[:, i])
            grad[i] = (1.0 / m) * sumdelta * - 1
        theta.shape = (3,)
        return grad


    def decorated_cost(self):
        def f(theta):
            return Admission.compute_cost(theta, self.it, self.y)

        def fprime(theta):
            return Admission.compute_grad(theta, self.it, self.y, self.m)

        theta = zeros(3)
        return fmin_bfgs(f, theta, fprime, disp=True, maxiter=400)


    def predict(self, theta, X):
        m, n = X.shape
        p = zeros(shape=(m, 1))
        h = MLHelper.sigmoid(X.dot(theta.T))
        for it in range(0, h.shape[0]):
            if h[it] > 0.5:
                p[it, 0] = 1
            else:
                p[it, 0] = 0
        return p

if __name__ == '__main__':
    admission = Admission()
    admission.decorated_cost()
    theta = [-25.161272, 0.206233, 0.201470]

    plot_x = array([min(admission.it[:, 1]) - 2, max(admission.it[:, 2]) + 2])
    plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
    plot(plot_x, plot_y)
    legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
    show()

    prob = MLHelper.sigmoid(array([1.0, 45.0, 85.0]).dot(array(theta).T))
    print 'For a student with scores 45 and 85, we predict an admission probability of %f' % prob

    p = admission.predict(array(theta), admission.it)
    print 'Training accuracy: %f' % ((admission.y[where(p == admission.y)].size / float(admission.y.size)) * 100.0)

