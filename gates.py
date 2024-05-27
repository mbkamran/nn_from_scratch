import numpy as np
import math


def sigmoid(z):
        return 1/(1+math.exp(-z[0]))



class AND:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = 1

        self.and_weights = np.ones((self.input_shape, self.output_shape))
        self.and_biases = np.ones(self.output_shape) * (1 - self.input_shape)


    def forward_pass_and(self, input_array):
        result = sigmoid(np.dot(input_array, self.and_weights) + self.and_biases)

        if result <= 0.5:
            return 0
        return 1


# and_ob = AND(3)
# input_array = np.array([1,0,1])
# print(and_ob.forward_pass_and(input_array))

class OR:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = 1

        self.or_weights = np.ones((self.input_shape, self.output_shape)) * self.input_shape
        self.or_biases = np.ones(self.output_shape) * (1 - self.input_shape)

    def forward_pass_or(self, input_array):
        result = sigmoid(np.dot(input_array, self.or_weights) + self.or_biases)

        if result <= 0.5:
            return 0
        return 1


# or_ob = OR(2)
# input_array = np.array([1,1])
# print(or_ob.forward_pass_or(input_array))

class NOT:
    def __init__(self):

        self.not_weights = [-1]
        self.not_biases = [1]

    def forward_pass_not(self, input_array):
        result = sigmoid(np.dot(input_array, self.not_weights) + self.not_biases)

        if result <= 0.5:
            return 0
        return 1
    
# not_ob = NOT()
# input_array = np.array([0])
# print(not_ob.forward_pass_not(input_array))

class NAND:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = 1

        self.nand_weights = np.ones((self.input_shape, self.output_shape)) * -1
        self.nand_biases = np.ones(self.output_shape) * (self.input_shape)


    def forward_pass_nand(self, input_array):
        result = sigmoid(np.dot(input_array, self.nand_weights) + self.nand_biases)

        if result <= 0.5:
            return 0
        return 1
    
# nand_ob = NAND(3)
# input_array = np.array([1,1,0])
# print(nand_ob.forward_pass_nand(input_array))

class XOR(NAND, OR, AND):
    def __init__(self, input_shape):
        NAND.__init__(self, input_shape)
        OR.__init__(self, input_shape)
        AND.__init__(self, 2)

    def forward_pass_xor(self, input_array):
        or_result = self.forward_pass_or(input_array)
        nand_result = self.forward_pass_nand(input_array)

        and_result = self.forward_pass_and(np.array([or_result, nand_result]))

        return and_result

xor_ob = XOR(3)
input_array = np.array([0,1,0])
print(xor_ob.forward_pass_xor(input_array))