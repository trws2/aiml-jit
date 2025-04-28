import numpy as np


np.random.seed(42)

# generate 2D data for two classes

mean1 = [1, 1]
var1 = [[1, 0], [0, 1]]
x_class1 = np.random.multivariate_normal(mean1, var1, size=10000)
ones_column = np.ones((x_class1.shape[0], 1))
x_class1 = np.column_stack((x_class1, ones_column))


mean2 = [-1, -1]
var2 = [[1, 0], [0, 1]]
x_class2 = np.random.multivariate_normal(mean2, var2, size=10000)
ones_column = np.ones((x_class2.shape[0], 1))
x_class2 = np.column_stack((x_class2, ones_column))


# include bias as the 3rd term
w = np.zeros((3, 1))
num_examples = x_class1.shape[0] + x_class2.shape[0]
learning_rate = 0.01
prev_loss = None

for i in range(1000):
    y_pred1 = np.dot(x_class1, w)
    y_pred2 = np.dot(x_class2, w)

    inv_p_pred1 = (1 + np.exp(-y_pred1))
    p_pred1 = 1 / inv_p_pred1
    inv_p_pred2 = (1 + np.exp(-y_pred2))
    p_pred2 = 1 / inv_p_pred2

    # logloss
    loss = (-1 / num_examples) * (np.sum(1 * np.log(p_pred1)) + np.sum(1 * np.log(1 - p_pred2)))

    if prev_loss is not None:
        change_perc = (loss - prev_loss) / prev_loss
        if change_perc < 0.000001:
            print(f"iter {i}: loss={loss}, prev_loss={prev_loss}, change_perc={change_perc}")
            break

    grad_class1 = (-1 / num_examples) * (-inv_p_pred1) * p_pred1 * (1 - p_pred1) * x_class1
    grad_class1 = np.sum(grad_class1, axis=(0))
    grad_class2 = (-1 / num_examples) * (1 / (1 - p_pred2)) * p_pred2 * (1 - p_pred2) * x_class2
    grad_class2 = np.sum(grad_class2, axis=(0))
    grad = grad_class1 + grad_class1
    grad = grad.reshape(w.shape)

    w -= learning_rate * grad
    if i % 100 == 0:
        print(f"iter {i}: w={w}")
    prev_loss = loss
    
print(f"final weights (last term is the bias) w={w}")


