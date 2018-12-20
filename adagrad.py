# Here will be the training loop applying an optimizer by importing different optimizer modules

import pandas as pd
import numpy as np
import math

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
    learning_rate = 10

    # let's see how many epochs do we need in order to achieve correct params
    # a should be 2
    # b should be 30
    # we check for an error of 0.001
    epochs = 0
    smoothing_term = 0.00000001
    sum_dL_db = 10
    sum_dL_da = 10
    while abs(2 - a) > 0.001 or abs(30 - b) > 0.001:
        for i in range(data_len):
            out = a * x[i] + b
            loss = mean_squared_error(out, y[i])
            dL_db = 2 * (out - y[i])
            dL_da = dL_db * x[i]
            sum_dL_da += dL_da ** 2
            sum_dL_db += dL_db ** 2
            a = a - (learning_rate / (math.sqrt(sum_dL_da) + smoothing_term)) * dL_da
            b = b - (learning_rate / (math.sqrt(sum_dL_db) + smoothing_term)) * dL_db
        print("a is: %f"%a)
        print("b is: %f"%b)
        epochs += 1
        print("Epochs: %f"%epochs)
    print("Final a is: %f"%a)
    print("Final b is: %f"%b)

    print("Number of epochs: %d"%epochs)
