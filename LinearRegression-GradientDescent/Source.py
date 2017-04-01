from numpy import *


def compute_error_for_given_points(theta_0, theta_1, points):
    total_error = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (theta_0 + theta_1 * x)) ** 2

    return total_error / float(len(points))


def step_gradient(current_theta_0, current_theta_1, points, learning_rate):
    # Gradient Descent
    gradient_theta_0 = 0
    gradient_theta_1 = 0
    m = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        gradient_theta_0 += -(2/m) * (y - (current_theta_0 + current_theta_1 * x))
        gradient_theta_1 += -(2/m) * x * (y - (current_theta_0 + current_theta_1 * x))

    new_theta_0 = current_theta_0 - (learning_rate * gradient_theta_0)
    new_theta_1 = current_theta_1 - (learning_rate * gradient_theta_1)

    return new_theta_0, new_theta_1


def gradient_descent_runner(points, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1

    for i in range(num_iterations):
        theta_0, theta_1 = step_gradient(theta_0, theta_1, array(points), learning_rate)
    return [theta_0, theta_1]


def run():
    points = genfromtxt('data.csv', delimiter = ',')

    # HyperParameter
    learning_rate = 0.0001

    # y = mx + b (slope formula) <=> y = theta_0 + theta_1 * x
    initial_theta_0 = 0
    initial_theta_1 = 0
    num_iterations = 1000
    [theta_0, theta_1] = gradient_descent_runner(points, initial_theta_0, initial_theta_1, learning_rate, num_iterations)
    print(theta_0)
    print(theta_1)

if __name__ == '__main__':
    run()
