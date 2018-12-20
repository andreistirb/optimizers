# Here will be the training loop applying an optimizer by importing different optimizer modules

import pandas as pd
import numpy as np

def mean_squared_error(y, y_hat):
    return (y - y_hat) ** 2

if __name__ == "__main__":
    data = pd.read_csv("data.csv")

    x = np.array(data['x'])
    y = np.array(data['y'])

    data_len = len(x)

    # initialize params
    a = 1
    b = 1
    
    # initialize hyperparams
    learning_rate = 0.001
    beta = 0.9
    v_a = 100
    v_b = 0

    # let's see how many epochs do we need in order to achieve correct params
    # a should be 2
    # b should be 30
    # we check for an error of 0.001
    epochs = 0

    while (abs(2 - a) > 0.001) or (abs(30 - b) > 0.001):
        #print(abs(30 - b))
        #print(abs(2 - a))
        for i in range(data_len):
            out = a * x[i] + b
            loss = mean_squared_error(out, y[i])
            dL_db = 2 * (out - y[i])
            dL_da = dL_db * x[i]
            v_a = beta * v_a + (1 - beta) * dL_da
            v_b = beta * v_b + (1 - beta) * dL_db
            a = a - learning_rate * v_a
            b = b - learning_rate * v_b
        epochs += 1
    print("Final a is: %f"%a)
    print("Final b is: %f"%b)

    print("Number of epochs: %d"%epochs)
