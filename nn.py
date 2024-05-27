import numpy as np

class NN:
    def __init__(self, input_shape, hidden_shape, output_shape):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.weights_hidden = np.random.randn(self.input_shape, self.hidden_shape) 
        self.weights_output = np.random.randn(self.hidden_shape, self.output_shape)

        self.biases_hidden = np.random.randn(self.hidden_shape)
        self.biases_output = np.random.randn(self.output_shape)


        # print(self.weights_hidden)
        # print(self.weights_output)
        # print()
        # print(self.biases_hidden)
        # print(self.biases_output)

    def forward_pass(self, input_array):
        hidden = self.sigmoid(np.dot(input_array, self.weights_hidden) + self.biases_hidden)
        output = self.sigmoid(np.dot(hidden, self.weights_output) + self.biases_output)

        if output <= 0.5:
            return 0
        return 1
        
    def backward_pass():
        pass

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

nn = NN(2,0,1)
input_array = np.array([1,1])
print(nn.forward_pass(input_array))
