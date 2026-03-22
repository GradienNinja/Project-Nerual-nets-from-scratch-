import numpy as np 

class Perceptron:
    def __init__(self, num_inputs, bias):
        # We generate random weights based on the number of inputs
        self.weights = np.random.rand(num_inputs)
        self.bias = bias 

    def sigmoid(self, x):
        # The activation function that squashes values between 0 and 1
        return 1 / (1 + np.exp(-x))

    def predict(self, current_input):
        # Calculate the weighted sum using the dot product
        # Math: (input1 * weight1) + (input2 * weight2) + ... + bias
        weighted_sum = np.dot(current_input, self.weights) + self.bias 
        
        # Pass the sum through the sigmoid and return the result
        return self.sigmoid(weighted_sum)

# --- THE PROJECT DATA ---
scenarios = [
    np.array([1, 1, 1]), # Sunny, weekend, warm
    np.array([0, 0, 0]), # Raining, weekday, cold
    np.array([1, 0, 1])  # Sunny, weekday, warm
]

# A bias of 0.5 makes the neuron more likely to say "let's go"
bias = 0.5 

# Initialize our brain for 3 pieces of information
perceptron = Perceptron(num_inputs=3, bias=bias) 

# --- THE TESTING LOOP ---
print("--- Perceptron Decision Project ---")
for i, scenario in enumerate(scenarios):
    # The 'scenario' is passed into the brain's predict method
    result = perceptron.predict(scenario)
    
    decision = "let's go" if result > 0.5 else "no need to go outside"
    print(f"Scenario {i+1}: Score {result:.4f} -> {decision}")