# Module6.py
from collections import defaultdict
import sys # required for command line arguments
from sklearn.neural_network import MLPClassifier  # required for MLPClassifier

def perceptron(weights, inputs):
    """
    Computes the output of a perceptron given weights and inputs.
    
    :param weights: List of weights for the perceptron.
    :param inputs: List of inputs to the perceptron.
    :return: Output of the perceptron (1 or 0).
    """
    if len(weights) != len(inputs):
        raise ValueError("Weights and inputs must have the same length.")
    
    # compute weighted sum
    weighted_sum = 0
    for i in range(len(weights)):
        weighted_sum += weights[i] * inputs[i]

    return 1 if weighted_sum > 0 else 0     


def train_perceptron(training_data, learning_rate=0.1, epochs=100):
    """
    Trains a perceptron using the provided training data.
    
    :param training_data: List of tuples (inputs, expected_output).
    :param learning_rate: Learning rate for weight updates.
    :param epochs: Number of epochs to train the perceptron.
    :return: List of weights after training.
    """
    if not training_data:
        raise ValueError("Training data cannot be empty.")
    
    num_inputs = len(training_data[0][0])
    weights = [0.0] * num_inputs
    
    for epoch in range(epochs):
        for inputs, expected in training_data:
            output = perceptron(weights, inputs)
            error = expected - output

            # update weights based on error
            for i in range(len(weights)):
                weights[i] += learning_rate * error * inputs[i]
    
    return weights

#------------------------------------------------------------------------
# Training Data
#------------------------------------------------------------------------
function1_training_data = [
        ([1, 0, 1], 0), 
        ([0, 1, 1], 0),  
        ([0, 0, 1], 0),  
        ([1, 1, 1], 1)   
    ]

function2_training_data = [
        ([1, 0, 1], 1),  
        ([0, 1, 1], 1), 
        ([0, 0, 1], 0),  
        ([1, 1, 1], 1)   
    ]

function3_training_data = [
        ([1, 0, 1], 1),  
        ([0, 1, 1], 1),  
        ([0, 0, 1], 0),  
        ([1, 1, 1], 0)   
    ]


#------------------------------------------------------------------------
# Create a MLP and train to solve function 3 
#------------------------------------------------------------------------
def test_function3_on_mlp(p_verbose, p_solver, p_hidden_layer_nodes, p_activation):
    inputs = [data[0] for data in function3_training_data]
    outputs = [data[1] for data in function3_training_data]
    
    mlp = MLPClassifier(
        verbose = p_verbose, solver=p_solver, 
        hidden_layer_sizes=(p_hidden_layer_nodes), 
        activation=p_activation, max_iter=10000
        ) 
    mlp.fit(inputs, outputs) 

    # Get the structure of the MLP model
    if p_verbose:
        print("Number of layers (including input and output):", mlp.n_layers_)
        print("Number of outputs:", mlp.n_outputs_)
        print("Hidden layer sizes:", mlp.hidden_layer_sizes)

        # Access weights and biases
        print("Weights for each layer:", [coef.shape for coef in mlp.coefs_])
        print("Biases for each layer:", [bias.shape for bias in mlp.intercepts_])

    # Test the trained MLP
    if p_verbose:
        print()
        print("Test the trained MLP for Function 3")
    passed = True
    for inputs, expected in function3_training_data: 
        output = mlp.predict([inputs])[0]
        if output != expected:
            passed = False
        if p_verbose:  
            print(f"Function 3 (MLP): Input {inputs} classified as: {output} (Expected: {expected})")  
    return passed


#------------------------------------------------------------------------
# TODO: Train the perceptron manually from the command line
#------------------------------------------------------------------------
weights = list(map(float, sys.argv[1:]))
if 0: # set to 1 to test weights from command line inputs for function 1
    for inputs, expected in function1_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 1: Input {inputs} classified as: {output} (Expected: {expected})")

if 0: # set to 1 to test weights from command line inputs for function 2
    for inputs, expected in function2_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 2: Input {inputs} classified as: {output} (Expected: {expected})")

if 0: # set to 1 to test weights from command line inputs for function 3
    for inputs, expected in function3_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 3: Input {inputs} classified as: {output} (Expected: {expected})")


#------------------------------------------------------------------------
# TODO: Test your algorithm to train the perceptron
#------------------------------------------------------------------------
if 0: 
    function1_weights = train_perceptron(function1_training_data)
    function2_weights = train_perceptron(function2_training_data)
    function3_weights = train_perceptron(function3_training_data)

    print(f"Trained weights for Function 1: {function1_weights}")
    print(f"Trained weights for Function 2: {function2_weights}")
    print(f"Trained weights for Function 3: {function3_weights}")
    
    print()
    print("Test the trained perceptron")
    for inputs, expected in function1_training_data:
        output = perceptron(function1_weights, inputs)
        print(f"Function 1: Input {inputs} classified as: {output} (Expected: {expected})")

    print()
    for inputs, expected in function2_training_data:
        output = perceptron(function2_weights, inputs)
        print(f"Function 2: Input {inputs} classified as: {output} (Expected: {expected})")

    print()
    for inputs, expected in function3_training_data:
        output = perceptron(function3_weights, inputs)
        print(f"Function 3: Input {inputs} classified as: {output} (Expected: {expected})")

 
 #------------------------------------------------------------------------
 # Test the MLPClassifier on function 3 by calling test_function3_on_mlp()
 #------------------------------------------------------------------------
# TODO - call test_function3_on_mlp() with different parameters
# test_function3_on_mlp(p_verbose, p_solver, p_hidden_layer_nodes, p_activation)
if 1:
    activations = ['relu', 'identity', 'logistic', 'tanh']
    solvers = ['lbfgs', 'sgd', 'adam']

    runs = 15
    pass_counts = defaultdict(int)

    # used to find weights and biases for a chosen set of hyperparameters
    # test_function3_on_mlp(1, 'adam', [15], 'tanh')

    for i in range(runs):
        for a in activations:
            for s in solvers:
                if test_function3_on_mlp(0, s, [3], a):
                    pass_counts[f'{a},{s}'] += 1

    for a in activations:
        print(f'\nActivation: {a}')
        for s in solvers:
            k = f'{a},{s}'
            print(f'{k}: {pass_counts[k]}/{runs} passed')