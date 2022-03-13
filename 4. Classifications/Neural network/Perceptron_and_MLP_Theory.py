#Perceptron:
#--The process of creating a neural network begins with single Perceptron.
#--A Perceptron is like a single neuron. It consists of one or more inputs,a bias(a0=1),
#--an activation function and a single output. The Perceptron receives inputs, multiplies them
#--by some weight and then passes them into activation function to produce an output.
#--There are many activation functions such as Logistic(sigmoid),trigonometric,step function etc.
#--Adding a bias to the Perceptron to avoids the issues where all input could be '0'.
#--weights are randomly initialize.

#To create a neural network we simply begin to add layers of Perceptrons together creating a 
#Multi-Layer Perceptrons(MLP) model. The MLP contains complete neural network layers(input,hidden & output).
#The hidden layer consists of one or more Layers.

#MLP has two models
#----1> MLPRegressor: Implements MLP for regression problems. Predict multiple target values at one time.
#----2> MLPClassifier: Implements MLP for Classifications problems. Its output one or many depending on no.
#                       of classes. 

#ADVANTAGES OF MLP:
    #1> Capability of learning non-linear models.
    #2> Capability of learning models in real-time.
    
#DISADVANTAGES OF MLP:
    #1> MLP with hidden layers have a non-convex loss function where there exists more than one local minima.
        #Therefore different random weight initializations can lead to different validation accuracy.
    #2> MLP requires tuning a no. of hyperparameters such as the no. of hidden neurons, layers and iterations.
        #to get a good accuracy.
    #3> MLP is sensitive to feature scaling.  