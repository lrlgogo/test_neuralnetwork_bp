import numpy as np


def fun_train_data():
    x = np.random.uniform(0, 2, (100, 2))
    y = np.array([t[0] ** 2 + t[1] ** 2 for t in x], dtype='float32')
    return x, y


def relu(x):
    return np.where(x > 0, x, 0)


def relu_def(x):
    return np.where(x > 0, 1., 0)


def fun_linear(x, w, b):
    return np.matmul(w, x) + b


def fun_loss_l2(y_pred, y_true):
    return 0.5 * np.mean(np.square(y_pred - y_true))


class MODEL:
    def __init__(self):
        self.w1 = np.random.randn(5, 2)
        self.w2 = np.random.randn(5, 5)
        self.w3 = np.random.randn(1, 5)
        self.b1 = 0.
        self.b2 = 0.
        self.b3 = 0.
        self.y1, self.y2, self.y3 = 0., 0., 0.
        self.z1, self.z2, self.z3 = 0., 0., 0.
        self.grad_w1, self.grad_w2, self.grad_w3 = 0., 0., 0.
        self.grad_b1, self.grad_b2, self.grad_b3 = 0., 0., 0.

    def forward(self, x):
        self.y1 = fun_linear(x, self.w1, self.b1)
        self.z1 = relu(self.y1)
        self.y2 = fun_linear(self.z1, self.w2, self.b2)
        self.z2 = relu(self.y2)
        self.y3 = fun_linear(self.z2, self.w3, self.b3)
        self.z3 = relu(self.y3)
        return self.z3

    def gradients(self, x, y_pred, y_true):
        delta_3 = (y_pred - y_true) * relu_def(self.y3)
        delta_2 = np.matmul(self.w3.T, delta_3) * relu_def(self.y2)
        delta_1 = np.matmul(self.w2.T, delta_2) * relu_def(self.y1)

        self.grad_w1 = np.matmul(delta_1, x.T)
        self.grad_w2 = np.matmul(delta_2, self.z1.T)
        self.grad_w3 = np.matmul(delta_3, self.z2.T)

        self.grad_b1, self.grad_b2, self.grad_b3 = delta_1, delta_2, delta_3

        return np.array([
            self.grad_w1, self.grad_w2, self.grad_w3,
            self.grad_b1, self.grad_b2, self.grad_b3
        ])

    def train(self, x_train, y_train, epochs=20, lr=0.01):
        for epoch in range(epochs):
            print('EPOCH {}\n'.format(epoch))
            for x0, y_true in zip(x_train, y_train):
                x = x0.reshape((2, -1))
                y = self.forward(x)
                loss = fun_loss_l2(y, y_true)
                grads = self.gradients(x, y, y_true)
                self.w1 -= lr * self.grad_w1
                self.w2 -= lr * self.grad_w2
                self.w3 -= lr * self.grad_w3
                self.b1 -= lr * self.grad_b1
                self.b2 -= lr * self.grad_b2
                self.b3 -= lr * self.grad_b3

                print('  loss: {}\n'.format(loss))


"""=========================================================================="""
x, y = fun_train_data()
model = MODEL()
model.train(x, y)
