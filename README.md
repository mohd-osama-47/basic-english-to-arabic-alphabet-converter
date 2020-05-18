![English "B" to Arabic "ب"](/images/b_to_baa.png)

# Basic English-to-Arabic Alphabet Converter
A basic feed-forward network that can convert a 5x5 array representing an English A or B into an equivalent Arabic Aleph or Baa through the use of gradient descent and python, numpy, and matplotlib. _(This was done as part of an assignment for a uni elective course, so excuse the incomplete nature of the project!)_

The network has one hidden layer and one output layer that the user has the choice of defning their amount. The user can also define the learning rate and the activation function to whatever they desire. The defualt activation function used is a sigmoid.

## The main class ```NeuralNetwork```
Takes the following parameters as inputs: the training examples set, said examples expected outputs (Targets), Number of neurons in the hidden layer, number of output layer neurons, and the learning rate. Within the class, there are 5 main functions.

### The first one is the ```__init__``` function
Which initializes some internal variables for the class. Said variables are the inputs, the randomized weights going into each hidden layer neuron (as a numpy array with weights randomly assigned between 0 and 1), the randomized weights for the output neurons (also randomized as a numpy array), the target values as an array using the data inputted by the user, the actual output initialized as zeros and calculated later in the forward pass of the network, and finally the learning rate.

### The second function is the ```feedforward``` function 
Calculates the weighted sum of all weights into a neuron and then passes the value into the sigmoid activation function. All of that is done in an array form for easier manipulation of data with the help of the numpy mathematical library. The function returns the output of the 2nd layer.

### The third function is the ```backpropagate``` function 
Responsible for calculating the needed deltas for each weight in the network using the principle of gradient descent. After the deltas are calculated for each weight, it is then added to all the appropriate weights and thus the weights of all the network neurons are updated. This calculation makes use of the dot product function of numpy to streamline the multiplication process and save everything in one neat array.

### The fourth function is the ```train``` function
Basically calls both the “feedforward” and “backpropagate” functions to basically run a single epoch and update the weights according to the available data.

### The final function is the ```testing_output``` function
Used after the user has trained the model to their desired value of error. The function takes a vector representing a character as an input and runs it through a forward pass of the network. Then, it displays how confident the network is if the detected vector is an Aleph or a Baa. Naturally, the higher percentage is what the network is more confident about.


## Sample Output

After that, the user simply defines an object of the class and defines how many hidden neurons they desire and what their learning rate is. For the purposes of demonstration, I have chosen 25 output neurons, each representing a singular pixel in the 5x5 character array. That array is approximated to show an array of 1s and 0s so that the user can see if the correct letter was predicted.

![Example of inputs and outputs](/images/network.png)

Another part of the code is centered around plotting. The code trains the model for 1500 iterations or until an error of less than 1% is reached. Some points are saved in a list for the purposes of plotting using the “matplotlib” python library. Two regions are colored to showcase regions of interest. Red region is where the error was greater than 1% and green is where its less than that. Thank you for reading this very brief explanation!

![Example of inputs and outputs](/images/sampleoutput.png)

