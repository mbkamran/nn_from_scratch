import numpy as np

class NN:
    def __init__(self, network_shape):
        self.input_shape = network_shape[0]
        self.output_shape = network_shape[-1]
        self.layer_outputs = []
        self.weights = []
        self.biases = []
        self.weights_update_matrix = []

        previous_layer_shape = self.input_shape

        for hidden_shape in network_shape[1:]:
            weights = np.random.randn(previous_layer_shape, hidden_shape)
            self.weights.append(weights)

            biases = np.random.randn(hidden_shape)
            self.biases.append(biases)

            previous_layer_shape = hidden_shape
            
        # print(f"Weights: \n{self.weights}")
        # print("----")

    def forward_pass(self, input_array):
        previous_output = input_array
        each_layer_outputs = [input_array]

        for weights, biases in zip(self.weights, self.biases):
            previous_output = self.sigmoid(np.dot(previous_output, weights) + biases)
            each_layer_outputs.append(previous_output)

        # print(each_layer_outputs)
        self.layer_outputs.append(each_layer_outputs)
        # print("----")
        return previous_output

    def backward_pass(self, predicted, target):
        derivative_error_y = -(target - predicted)
        derivative_y_z = predicted * (1 - predicted)
        derivative_z_weights = np.array([layers_output[-2] for layers_output in self.layer_outputs])

        # print(f"self.layer_outputs: \n{self.layer_outputs}")
        # print("----")

        result = np.dot((derivative_error_y * derivative_y_z), derivative_z_weights).reshape(-1,1)
        self.weights_update_matrix.insert(0, result)

        # print(f"self.weights_update_matrix: \n{self.weights_update_matrix}")
        # print("----")


        for layer in range(-1, -len(self.weights), -1):
            derivative_z_y = self.weights[layer]
            prev_layer_output = np.array([layers_output[layer-1] for layers_output in self.layer_outputs])
            derivative_y_z_prev = prev_layer_output * (1 - prev_layer_output)
            derivative_z_weights_prev = np.array([layers_output[layer-2] for layers_output in self.layer_outputs])

            # print(f"derivative_error_y: \n{derivative_error_y}")
            # print(f"derivative_y_z: \n{derivative_y_z}")
            # print(f"derivative_z_y: \n{derivative_z_y.T}")
            # print(f"derivative_y_z_prev: \n{derivative_y_z_prev}")
            # print(f"derivative_z_weights_prev: \n{derivative_z_weights_prev}")

            # temp_ans = (derivative_error_y * derivative_y_z)
            # temp_ans2 = (derivative_z_y.T * derivative_y_z_prev)
            # temp_ans3 = np.dot(derivative_z_weights_prev.T, (derivative_z_y.T * derivative_y_z_prev))

            # print("----")
            # print(f"temp_ans: \n{temp_ans}")
            # print(f"temp_ans2: \n{temp_ans2}")
            # print(f"temp_ans3: \n{temp_ans3}")
            # print("----")

            equation_temp1 = (derivative_error_y * derivative_y_z).reshape(-1,1)
            equation_temp2 = np.dot(equation_temp1, derivative_z_y.T)
            equation_temp3 = (equation_temp2 * derivative_y_z_prev)
            final_output = np.dot(derivative_z_weights_prev.T, equation_temp3)

            # print(final_output)



            self.weights_update_matrix.insert(0, final_output)

        # print(self.weights_update_matrix)

        
        # print(self.weights_update_matrix)
        # store the result of each layer as well, they will be used in backward propagation

    def weight_update(self, alpha):
        for index, updates in enumerate(self.weights_update_matrix):
            self.weights[index] -= alpha*updates

        # print(f"Updated weights: \n{self.weights}")

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def binary_cross_entropy_loss(self, predicted, target):
        return -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted)).mean()
    
    def fit(self, input_array, target_array, alpha, epochs):

        for epoch in range(epochs):
            fwdpass_result = np.array([])
            for input_data in input_array:
                result = self.forward_pass(input_data)
                fwdpass_result = np.concatenate((fwdpass_result, result))

            print(f"Error: {self.binary_cross_entropy_loss(fwdpass_result, target_array)}")
            self.backward_pass(fwdpass_result, target_array)
            self.weight_update(alpha)

            self.weights_update_matrix = []
            self.layer_outputs = []

    def predict(self, input_array):
        fwdpass_result = np.array([])
        for input_data in input_array:
            result = self.forward_pass(input_data)
            if result <= 0.5:
                result = [0]
            else:
                result = [1]
            fwdpass_result = np.concatenate((fwdpass_result, result))

        return fwdpass_result




nn = NN([2,3,1])
input_mat = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])
target = np.array([0,1,1,0])

nn.fit(input_mat, target, 2.3, 10000)
pred = nn.predict(input_mat)
print(pred)

# input_array1 = np.array([0,0])
# input_array2 = np.array([0,1])
# input_array3 = np.array([1,0])
# input_array4 = np.array([1,1])

# ans1 = nn.forward_pass(input_array1)
# ans1 = np.concatenate((ans1, nn.forward_pass(input_array2)))
# ans1 = np.concatenate((ans1, nn.forward_pass(input_array3)))
# ans1 = np.concatenate((ans1, nn.forward_pass(input_array4)))

# target = np.array([0,0,0,1])
# # target = np.array([0])
# # target_single = np.array([1])
# nn.backward_pass(ans1, target)
# nn.weight_update(0.1)
# print(f"Final output: {ans1}")
# print(target)

# print(nn.binary_cross_entropy_loss(ans1, target))