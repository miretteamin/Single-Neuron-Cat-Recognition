## Logistic Regression ##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mim
import h5py
import scipy
import scipy.misc
from scipy import ndimage
from PIL import Image
import random


# from lr_utils import load_dataset
def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s


# Loading DATA
train_dataset = h5py.File('Datasets/Cats/train_catvnoncat.h5/train_catvnoncat.h5', 'r')
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])

test_dataset = h5py.File('Datasets/Cats/test_catvnoncat.h5', 'r')
test_set_x_orig = np.array(test_dataset["test_set_x"][:])
test_set_y_orig = np.array(test_dataset["test_set_y"][:])

# classes = np.array(test_dataset["list_classes"][:])
train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


# Showing an image
def showImage(index):
    plt.imshow(train_set_x_orig[index])
    plt.show()
    x = "cat"
    if (train_set_y_orig[index] == 0):
        x = "non cat"
    # instead of x it was witten classes[np.squeeze(train_set_y[:,
    # index])].decode("utf-8") as there are 2 classes cat and non cat
    # and was written also str(train_set_y[:, index])
    print("y = " + str(train_set_y_orig[index]) + ", it's a '" + x + "' picture.")


# Reshape the training and test examples
train_2D_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_2D_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Normalizing the Dataset to be between [-1,1] so all the neurons would be
# trained
train_x_normalized = train_2D_x / 255
test_x_normalized = test_2D_x / 255

# Initialize the model's parameters (weights and bias term)
def initialize_with_zeros(dimensions):
    #weights = np.random.rand(dimensions, 1) * 0.1
    #b = random.random() * 0.1

    weights = np.zeros((dimensions,1))
    b = 0

    return weights, b


def propagate(w, b, X, Y):
    """ Implement the cost function and its gradient for the propagation
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b  """
    nb = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # Compute Activation
    costFn = (-1.0 / nb) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # Compute Cost Function

    # BACKWARD PROPAGATION (TO FIND GRAD)
    # Calculate Derivatives
    dw = (1.0 / nb) * (np.dot(X, (A - Y).T))
    db = (1.0 / nb) * np.sum(A - Y)
    grads = {"dw": dw, "db": db}
    costFn = np.squeeze(costFn)
    return grads, costFn


# lets update parameters
def update(w, b, X, Y, num_iterations, alpha, print_cost=False):
    """ This function optimizes w and b by running a gradient descent algorithm
    Arguments:
    num_iterations -- number of iterations of the optimization loop
    alpha -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve."""

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        # taking values of derivatives
        dw = grads["dw"]
        db = grads["db"]

        w = w - alpha * dw
        b = b - alpha * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}

        if (i >= 1):
            if (abs((cost - last_cost)) < (1 / 1000000)):
                break

        last_cost = cost

    return params, grads, costs


def predict(w, b, X):
    nb = X.shape[1]
    Y_prediction = np.zeros((1, nb))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if (A[0, i] > 0.5):
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
        # It can be written as Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, alpha=0.1, print_cost=False):
    # Initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent
    params, grads, costs = update(w, b, X_train, Y_train, num_iterations, alpha, print_cost)

    w = params["w"]
    b = params["b"]

    Y_test_prediction = predict(w, b, X_test)
    Y_train_prediction = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_prediction - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_prediction - Y_test)) * 100))

    final = {"costs": costs,
             "Y_test_prediction": Y_test_prediction,
             "Y_train_prediction": Y_train_prediction,
             "w": w,
             "b": b,
             "learning_rate": alpha,
             "num_iterations": num_iterations}

    return final

##Training
finalmodel = model(train_x_normalized, train_set_y, test_x_normalized,test_set_y, num_iterations = 15000, alpha = 0.0001, print_cost = True)

# models = [model(train_x_normalized, train_set_y,
# test_x_normalized,test_set_y,
# num_iterations = 1000, alpha = 0.05, print_cost =
# False),model(train_x_normalized, train_set_y, test_x_normalized,test_set_y,
# num_iterations = 1000, alpha = 0.001, print_cost = False)]


# Plot learning curve (with costs)
def plotsinglemodel(finalmodel):
    costs = np.squeeze(finalmodel['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = " + str(finalmodel["learning_rate"]))
    plt.show()


def plotallmodels(models):
    for i in models:
        plt.plot(np.squeeze(i["costs"]), label=str(i["learning_rate"]))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    legend = plt.legend(loc='down center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('red')
    plt.show()

# My image's Path
path = "Datasets/kagglecatsanddogs_3367a/PetImages/Cat/1021.jpg"

def showmyimage(path):
    my_image_data = mim.imread(path)
    plt.imshow(my_image_data)
    plt.show()

myimage = Image.open(path)
myimage = myimage.resize((64,64))
my_image_data = np.array(myimage)
my_image_data = my_image_data / 255
my_image_data = my_image_data.reshape(my_image_data.shape[0] * my_image_data.shape[0] * 3,1)

#print(my_image_data)
result = predict(finalmodel["w"], finalmodel["b"], my_image_data)
showmyimage(path)

if result == 1:
    x = "cat"
else:
    x = "non-cat"

print("y = " + str(np.squeeze(result)) + ", your algorithm predicts a \"" + x + "\" picture.")

"""my_image = "my_image.jpg"   # change this to the name of your image file 

    # We preprocess the image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    image = image/255.
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")"""
