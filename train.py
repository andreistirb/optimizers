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
    learning_rate = 0.0001

    # let's see how many epochs do we need in order to achieve correct params
    # a should be 2
    # b should be 30
    # we check for an error of 0.0001
    epochs = 0

    while abs(2 - a) > 0.0001 and abs(30 - b) > 0.0001:
        for i in range(data_len):
            out = a * x[i] + b
            loss = mean_squared_error(out, y[i])
            dL_db = 2 * (out - y[i])
            dL_da = dL_db * x[i]
            a = a - learning_rate * dL_da
            b = b - learning_rate * dL_db
        epochs += 1
    print("Final a is: %f"%a)
    print("Final b is: %f"%b)

    print("Number of epochs: %d"%epochs)

