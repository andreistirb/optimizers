# Gradient Descent Optimizers
The algorithm runs a classic linear regression that outputs a*x + b in the forward pass. 


It stops when both parameters reach a 0.001 difference between learned weights and true weights.

# Results

Below we show results: number of epochs until it reaches stopping condition described above

Optimizer | Number of epochs | Learning Rate | Observations 
----------|-----------------|--------------|------
Stochastic Gradient Descent | 6644 | 0.0001 | diverges with a higher learning rate
Momentum | 643 | 0.001
Adagrad | 88 | 10
Rmsprop | 2094 | 0.001 | gamma = 0.999
Adam | 21 | 25 | beta = 0.9, gamma = 0.999
Eve | | 
Adagrad ann | | 
Adam ann | |

# Theory
Best article: http://ruder.io/optimizing-gradient-descent/index.html

