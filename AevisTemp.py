import math
from random import random

#region Vectors and Activations
class Vector:
    def __init__(self , dimensions = 2 , values = [ 0 , 0 ]):
        self.dim = dimensions
        assert (len(values) - 1) != self.dim , "The Vectors dimension does not meet the required dimensions"
        self.components = []
        for i in range(self.dim) :
            self.components.append(values[i])
    def set_component(self , index , value):
        self.components[index] = value
    def get_component(self , index):
        return self.components[index]
    def __add__(self , other):
        vec = Vector(self.dim)
        for i in self.dim :
            vec.set_component(i , self.get_componenent(i) + self.get_component(i))
        return vec

class Activation:
    def __init__(self , type = 'none'):
        self.type = type
    def sigmoid(self):
        self.type = 'sigmoid'
    def tanh(self):
        self.type = 'tanh'
    def __call__(self , x):
        if self.type == 'sigmoid':
            return 1/(1+(math.e**x))
        if self.type == 'tanh':
            return 2/(1+math.e**((-2)*x))
        if self.type == 'none':
            return x

#endregion

#region Primitives

class Connection:
    def __init__(self , weight = random()):
        self.weight = weight

    def __call__(self , x):
        return self.weight*x

class Neuron :
    def __init__(self , bias = random() , weight = random() , activation = Activation()):
        self.bias = bias
        self.activation = activation
        self.weight = weight
    
    def __call__(self , x):
        return self.activation(self.bias + (x*self.weight))

class lstmCell:
    def __init__(self , bias = random() ,
                 weight = random() , 
                 forget_input_weight = random() , 
                 forget_hidden_weight = random() , 
                 forget_bias = random() , 
                 input_input_weight = random() , 
                 input_hidden_weight = random() , 
                 input_bias = random() , 
                 candidate_input_weight = random() , 
                 candidate_hidden_weight = random() , 
                 candidate_bias = random() ,
                 output_hidden_weight = random(),
                 output_input_weight = random(),
                 output_bias = random()
                 ):
        
        self.bias = bias
        self.sigmoid = Activation('sigmoid')
        self.tanh = Activation('tanh')
        
        self.forget_input_weight =  forget_input_weight
        self.forget_hidden_weight = forget_hidden_weight
        self.forget_bias = forget_bias
        
        self.input_input_weight = input_input_weight
        self.input_hidden_weight = input_hidden_weight
        self.input_bias = input_bias

        self.candidate_input_weight = candidate_input_weight
        self.candidate_hidden_weight = candidate_hidden_weight
        self.candidate_bias = candidate_bias

        self.output_input_weight = output_input_weight
        self.output_hidden_weight = output_hidden_weight
        self.output_bias = output_bias
        self.weight = weight
    
    def __call__(self , x , cell_state , hidden_state):
        # forget gate
        c = cell_state*(self.sigmoid(self.forget_bias + ((hidden_state*self.forget_hidden_weight) + (x*self.forget_input_weight)) ))
        
        #input gate
        input_out = (self.sigmoid( ( (self.input_hidden_weight*hidden_state)     + (self.input_input_weight*x))     + self.input_bias )     * 
                     self.tanh(    ( (self.candidate_hidden_weight*hidden_state) + (self.candidate_input_weight*x)) + self.candidate_bias ))
        
        c = c + input_out

        #output
        o = self.sigmoid( (self.output_hidden_weight*hidden_state) + (self.input_input_weight*x)) + self.bias

        final_output = o * self.tanh(c)

        return [c , final_output*self.weight , final_output*self.weight]
#endregion

#region Models
class LSTMNetwork :
    def __init__(self , input_dimensions = 3 , stack_num = 3 , stack_dimension = 5 , output_dimension = 3):
        self.input_dimensions = input_dimensions
        self.stack_num = stack_num
        self.stack_dimension = stack_dimension
        self.output_dimension = output_dimension

        self.cell_state = 0
        self.hidden_state = 0

        self.Input_Layer_Neurons = []
        self.Stack = []
        self.Stack_Connections = []
        self.Output_Layer = []

        #Create input neurons
        for i in range(self.input_dimensions) :
            self.Input_Layer_Neurons.append(Neuron())
        
        #Create hidden lstm cells
        for i in range(self.stack_dimension):
            curr_layer = []
            for j in range(self.stack_num):
                curr_layer.append(lstmCell())
            self.Stack.append(curr_layer)
            
        
        #Create Output Neurons
        for i in range(self.output_dimension) :
            self.Output_Layer.append(Neuron())
    
    def __call__(self , x : Vector ):

        fx = 0

        if x.dim == self.input_dimensions:
            outputs = [0]
            for i in range(self.input_dimensions):
                fx = fx + self.Input_Layer_Neurons[i](x.get_component(i))
                print(fx)
            
            outputs[0] = fx

            for i in range(len(self.Stack)):
                outputs.append(0)
                for j in range(len(self.Stack[i])):
                    lstm_out = self.Stack[i][j](outputs[i] , self.cell_state , self.hidden_state)
                    self.cell_state = lstm_out[0]
                    self.hidden_state = lstm_out[1]
                    outputs[i] = outputs[i] + lstm_out[2]
                    print(outputs[i])
            
            outputs.pop()
            
            vec_dim = []
            for i in range(self.output_dimension):
                vec_dim.append(0)

            sum = 0
            i = len(outputs) - 1
            while i >= len(outputs) - self.stack_dimension:
                sum = sum + outputs[i]
                i = i - 1

            out = Vector(self.output_dimension , vec_dim)
            for i in range(len(self.Output_Layer)):
                curr_out = self.Output_Layer[i](outputs[len(outputs)-1-i])
                out.set_component(i , curr_out)
                print(curr_out)
            
            print("-----------------------------------------")

            a = []
            for i in range(out.dim):
                a.append(out.get_component(i))
            return a

        else :
            raise IndexError("Input Vector dimensions does not match Input Layer Dimension")

    def backpropogate(self , io_pairs : dict):
        pass
        
#endregion