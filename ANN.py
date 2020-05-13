# Imports
import numpy as np
import matplotlib.pyplot as plt

#Arabic alphabet string for aleph and baa
aleph = [0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]
baa   = [0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0]

#English string for A and B
an_A = [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1]
a_B = [1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0]

# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-1*x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Class definition
class NeuralNetwork:
    """This is a class describing a two layer, FF NN.
        The user can define the number of hidden neurons, output neurons,
        The learning rate, and feed into it examples and targets for said
        examples"""

    def __init__(self, x,y, number_of_neurons, output_neurons, learning_rate):
        """Specifies the parameters for the whole network from user inputs"""
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],number_of_neurons) 
        self.weights2 = np.random.rand(number_of_neurons,output_neurons)
        self.y = y
        self.output = np. zeros(y.shape)
        self.l_rate = learning_rate

    def feedforward(self):
        """Calculates all activations in the network, returns final output activations"""
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))    #the hidden layer
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))   #the output layer
        return self.layer2

    def backpropagate(self):
        """uses gradient descent principle to calculate the deltas for each
           weight and update them to reach a lower error value."""

        #deltas for the weights into the outputlayer from the hidden layer.
        d_weights2 = np.dot(self.layer1.T, (self.y - self.output)*sigmoid_derivative(self.output))
        #deltas for the weights into the hidden layer from the input.
        d_weights1 = np.dot(self.input.T, np.dot((self.y - self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))

        self.weights1 += self.l_rate*d_weights1
        self.weights2 += self.l_rate*d_weights2

    def train(self, X, y):
        """Chains the forward and backward passes for effectively running 1 epoch"""
        self.output = self.feedforward()
        self.backpropagate()

    def testing_output(self, value):
        """Using an input, generate and return the final output activations"""
        self.layer1 = sigmoid(np.dot(value, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        ErrorA = np.mean(0.5*np.square(aleph - self.layer2))
        ErrorB = np.mean(0.5*np.square(baa - self.layer2))
        print("Predicted  to be an Aleph by: {:.4f}%".format(100*(1-ErrorA)))
        print("Predicted to be a Baa by: {:.4f}%".format(100*(1-ErrorB)))


# Each row is a training example, each coloumn represents a singualr pixel
# for their respective example.

# #This is for generating the 5x5 array of pixels for english to arabic classification
examples=np.array(([1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1],
                   [1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0],
                   [1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0],
                   [1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1],
                   [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,1,0],
                   [1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0]), dtype=int)
targets=np.array((aleph,baa,baa,aleph,aleph,baa), dtype=int)

#this is for generating a NN to detect A from B.
# examples=np.array(([1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1],
#                    [1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0],
#                    [1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0],
#                    [1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1],
#                    [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,1,0],
#                    [1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0]), dtype=int)
# targets=np.array(([1,0],[0,1],[0,1],[1,0],[1,0],[0,1]), dtype=int)

#for plotting the error as a function of training iterations
x_axis = []    #Iterations
y_axis = []    #Error as a %

#specify a NN with 7 hidden neurons, 2 output neurons
NN = NeuralNetwork(examples,targets,10,25,0.5)

i = 0
while(i < 1500): # trains the NN 1,500 times
    
    if i % 10 == 0: 
        Error = np.mean(0.5*np.square(targets - NN.feedforward())) # mean sum squared loss
        x_axis.append(i)
        y_axis.append(Error*100)

    NN.train(examples, targets)

    if Error < 0.005:
        print(f"\nNumber of iterations to reach Error less than 1%: {i}\n")
        break
    i += 1

test_array = a_B
#test_array = [1,1,1,1,1,1,0,0,,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1] #slightly missed up A
NN.testing_output(test_array)

print("\nYour input of:")
print("",test_array[:5] , "\n",
       test_array[5:10] , "\n",
      test_array[10:15] , "\n",
      test_array[15:20] , "\n",
      test_array[20:25] , "\n",)

print("Is aproximated to:") #after rounding,print the output
print("",NN.layer2[:5].round() , "\n",
       NN.layer2[5:10].round() , "\n",
      NN.layer2[10:15].round() , "\n",
      NN.layer2[15:20].round() , "\n",
      NN.layer2[20:25].round() , "\n",)


#for plotting purposes
fig = plt.figure(figsize=(13,6))
ax = fig.add_subplot(111)
ax.plot(x_axis,y_axis, marker='.')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error %')

plt.axvspan(0, i, color='red', alpha=0.5) 
plt.axvspan(i, i*(1.05), color='green', alpha=0.5)
ax.text(100, 40, 'Error > 1%')
ax.text(i, 40, 'Error < 1%')

plt.grid()
plt.show()