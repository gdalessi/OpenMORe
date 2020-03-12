'''
MODULE: ANN.py

@Author: 
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Details:
    This module contains a set of functions/classes which are based on ANN. The following architectures are implemented:
    1) Single-layer MLP for classification tasks.
    2) Autoencoder for non-linear dimensionality reduction.
    3) Single-layer MLP for regression tasks.

@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import os 
import os.path
from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint 
from keras.layers import LeakyReLU


class MLP_classifier:
    def __init__(self, X, Y, save=False):
        self.X = X
        self.Y = Y
        #self.neurons = n_neurons
        
        self._activation = 'relu'
        self._batch_size = 64
        self._n_epochs = 1000
        self._n_neurons = 2
        
        self.save_txt = save

        if self.X.shape[0] != self.Y.shape[0]:
            raise Exception("The number of observations (Input and Output) does not match: please check again your Input/Output. Exiting...")
            exit()

    @property
    def neurons(self):
        return self._n_neurons
    
    @neurons.setter
    def neurons(self, new_number):
        self._n_neurons = new_number

        if self._n_neurons <= 0:
            raise Exception("The number of neurons in the hidden layer must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._n_neurons, int) != True: 
            raise Exception("The number of neurons in the hidden layer must be an integer. Exiting..")
            exit()

    @property
    def activation(self):
        return self._activation
    
    @activation.setter
    def activation(self, new_activation):
        if new_activation == 'leaky_relu':
            LR = LeakyReLU(alpha=0.0001)
            LR.__name__= 'relu'
            self._activation= LR
        else:
            self._activation = new_activation

    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, new_batchsize):
        self._batch_size = new_batchsize

        if self._batch_size <= 0:
            raise Exception("The batch size must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._batch_size, int) != True: 
            raise Exception("The batch size must be an integer. Exiting..")
            exit()

    @property
    def n_epochs(self):
        return self._n_epochs
    
    @n_epochs.setter
    def n_epochs(self, new_epochs):
        self._n_epochs = new_epochs

        if self._n_epochs <= 0:
            raise Exception("The number of epochs must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._n_epochs, int) != True: 
            raise Exception("The number of epochs must be an integer. Exiting..")
            exit()

    @staticmethod
    def set_hard_parameters():
        activation_output = 'softmax'
        path_ = os.getcwd()
        monitor_early_stop= 'val_loss'
        optimizer_= 'adam'
        patience_= 5
        loss_classification_= 'categorical_crossentropy'
        metrics_classification_= 'accuracy'

        return activation_output, path_, monitor_early_stop, optimizer_, patience_, loss_classification_, metrics_classification_

    
    @staticmethod
    def idx_to_labels(idx):
        k = max(idx) +1
        n_observations = idx.shape[0]
        labels = np.zeros(n_observations, k)

        for ii in range(0,n_observations):
            for jj in range(0,k):
                if idx[ii] == jj:
                    labels[ii,jj] = 1
        
        return labels
    

    def fit_network(self):

        if self.Y.shape[1] == 1:        # check if the Y matrix is in the correct form
            print("Changing idx shape in the correct format: [n x k]..")
            self.Y = MLP_classifier.idx_to_labels(self.Y)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.3)
        input_dimension = self.X.shape[1]
        number_of_classes = self.Y.shape[1]
        activation_output, path_, monitor_early_stop, optimizer_, patience_, loss_classification_, metrics_classification_ = MLP_classifier.set_hard_parameters()

        classifier = Sequential()
        classifier.add(Dense(self._n_neurons, activation=self._activation, kernel_initializer='random_normal', input_dim=input_dimension))
        classifier.add(Dense(number_of_classes, activation=activation_output, kernel_initializer='random_normal'))
        classifier.summary()

        earlyStopping = EarlyStopping(monitor=monitor_early_stop, patience=patience_, verbose=1, mode='min')
        mcp_save = ModelCheckpoint(filepath=path_ + '/best_weights.h5', verbose=1, save_best_only=True, monitor=monitor_early_stop, mode='min')

        classifier.compile(optimizer =optimizer_,loss=loss_classification_, metrics =[metrics_classification_])
        history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=self._batch_size, epochs=self._n_epochs, callbacks=[earlyStopping, mcp_save])

        # Summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch number')
        plt.legend(['Train', 'Test'], loc='lower right')
        plt.show()

        # Summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

        if self.save_txt:

            first_layer_weights = classifier.layers[0].get_weights()[0]
            first_layer_biases  = classifier.layers[0].get_weights()[1]
            class_layer_weights = classifier.layers[1].get_weights()[0]
            class_layer_biases = classifier.layers[1].get_weights()[1]
            
            np.savetxt(path_+ '/weightsHL1.txt', first_layer_weights)
            np.savetxt(path_+ '/biasHL1.txt', first_layer_biases)
            np.savetxt(path_+ '/weightsCL.txt', class_layer_weights)
            np.savetxt(path_+ '/biasCL.txt', class_layer_biases)
        
        test = classifier.predict(self.X)
        return test


class Autoencoder:
    def __init__(self, X, save=False):
        self.X = X
        
        self._n_neurons = 1
        self._activation = 'relu'
        self._batch_size = 64
        self._n_epochs = 1000

    @property
    def neurons(self):
        return self._n_neurons
    
    @neurons.setter
    def neurons(self, new_number):
        self._n_neurons = new_number

        if self._n_neurons <= 0:
            raise Exception("The number of neurons in the hidden layer must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._n_neurons, int) != True: 
            raise Exception("The number of neurons in the hidden layer must be an integer. Exiting..")
            exit()
        elif self._n_neurons >= self.X.shape[1]:
            raise Exception("The reduced dimensionality cannot be larger than the original number of variables. Exiting..")
            exit()

    @property
    def activation(self):
        return self._activation
    
    @activation.setter
    def activation(self, new_activation):
        if new_activation == 'leaky_relu':
            LR = LeakyReLU(alpha=0.0001)
            LR.__name__= 'relu'
            self._activation= LR
        else:
            self._activation = new_activation

    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, new_batchsize):
        self._batch_size = new_batchsize

        if self._batch_size <= 0:
            raise Exception("The batch size must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._batch_size, int) != True: 
            raise Exception("The batch size must be an integer. Exiting..")
            exit()

    @property
    def n_epochs(self):
        return self._n_epochs
    
    @n_epochs.setter
    def n_epochs(self, new_epochs):
        self._n_epochs = new_epochs

        if self._n_epochs <= 0:
            raise Exception("The number of epochs must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._n_epochs, int) != True: 
            raise Exception("The number of epochs must be an integer. Exiting..")
            exit()


    @staticmethod
    def set_hard_parameters():
        activation_output = 'linear'
        path_ = os.getcwd()
        monitor_early_stop= 'val_loss'
        optimizer_= 'adam'
        patience_= 5
        loss_function_= 'mse'
        metrics_= 'accuracy'

        return activation_output, path_, monitor_early_stop, optimizer_, patience_, loss_function_, metrics_

    
    def fit(self):
        from keras.layers import Input, Dense
        from keras.models import Model
        
        input_dimension = self.X.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.X, test_size=0.3)
        
        activation_output, path_, monitor_early_stop, optimizer_, patience_, loss_function_, metrics_ = Autoencoder.set_hard_parameters()

        input_data = Input(shape=(input_dimension,))
        encoded = Dense(self._n_neurons, activation=self._activation)(input_data)
        decoded = Dense(input_dimension, activation=activation_output)(encoded)
        
        autoencoder = Model(input_data, decoded)

        encoder = Model(input_data, encoded)
        encoded_input = Input(shape=(self._n_neurons,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer=optimizer_, loss=loss_function_)

        earlyStopping = EarlyStopping(monitor=monitor_early_stop, patience=patience_, verbose=1, mode='min')
        history = autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=self._n_epochs, batch_size=self._batch_size, shuffle=True, callbacks=[earlyStopping])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

        encoded_X = encoder.predict(self.X)

        if self.save_txt:
            first_layer_weights = encoder.get_weights()[0]
            first_layer_biases  = encoder.get_weights()[1]

            np.savetxt(path_+ 'AEweightsHL1.txt', first_layer_weights)
            np.savetxt(path_+ 'AEbiasHL1.txt', first_layer_biases)

            np.savetxt(path_+ 'Encoded_matrix.txt', encoded_X)


class MLP_regressor:
    def __init__(self, X, Y, save=False):
        self.X = X
        self.Y = Y

        self._n_neurons = 2
        self._activation = 'relu'
        self._batch_size = 64
        self._n_epochs = 1000
        
        self.save_txt = save

        if self.X.shape[0] != self.Y.shape[0]:
            raise Exception("The number of observations (Input and Output) does not match: please check again your Input/Output. Exiting...")
            exit()
    
    @property
    def neurons(self):
        return self._n_neurons
    
    @neurons.setter
    def neurons(self, new_number):
        self._n_neurons = new_number

        if self._n_neurons <= 0:
            raise Exception("The number of neurons in the hidden layer must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._n_neurons, int) != True: 
            raise Exception("The number of neurons in the hidden layer must be an integer. Exiting..")
            exit()

    @property
    def activation(self):
        return self._activation
    
    @activation.setter
    def activation(self, new_activation):
        if new_activation == 'leaky_relu':
            LR = LeakyReLU(alpha=0.0001)
            LR.__name__= 'relu'
            self._activation= LR
        else:
            self._activation = new_activation

    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, new_batchsize):
        self._batch_size = new_batchsize

        if self._batch_size <= 0:
            raise Exception("The batch size must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._batch_size, int) != True: 
            raise Exception("The batch size must be an integer. Exiting..")
            exit()

    @property
    def n_epochs(self):
        return self._n_epochs
    
    @n_epochs.setter
    def n_epochs(self, new_epochs):
        self._n_epochs = new_epochs

        if self._n_epochs <= 0:
            raise Exception("The number of epochs must be a positive integer. Exiting..")
            exit()
        elif isinstance(self._n_epochs, int) != True: 
            raise Exception("The number of epochs must be an integer. Exiting..")
            exit()


    @staticmethod
    def set_hard_parameters():
        activation_output = 'linear'
        path_ = os.getcwd()
        monitor_early_stop= 'mean_squared_error'
        optimizer_= 'adam'
        patience_= 5
        loss_function_= 'mean_squared_error'
        metrics_= 'mse'

        return activation_output, path_, monitor_early_stop, optimizer_, patience_, loss_function_, metrics_


    def fit_network(self):
        input_dimension = self.X.shape[1]
        output_dimension = self.Y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.3)

        activation_output, path_, monitor_early_stop, optimizer_, patience_, loss_function_, metrics_ = MLP_regressor.set_hard_parameters()

        model = Sequential()
        model.add(Dense(self._n_neurons, input_dim=input_dimension, kernel_initializer='normal', activation=self._activation)) 
        model.add(Dense(output_dimension, activation=activation_output))
        model.summary()

        earlyStopping = EarlyStopping(monitor=monitor_early_stop, patience=patience_, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(filepath=path_+ '/best_weights2c.h5', verbose=1, save_best_only=True, monitor=monitor_early_stop, mode='min')
        model.compile(loss=loss_function_, optimizer=optimizer_, metrics=[metrics_])
        history = model.fit(X_train, y_train, batch_size=self._batch_size, epochs=self._n_epochs, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping, mcp_save])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

        model.load_weights(path_+ '/best_weights2c.h5')

        test = model.predict(self.X)

        if self.save_txt:

            first_layer_weights = model.layers[0].get_weights()[0]
            first_layer_biases  = model.layers[0].get_weights()[1]
            out_layer_weights = model.layers[1].get_weights()[0]
            out_layer_biases = model.layers[1].get_weights()[1]

            np.savetxt(path_+ '/weightsHL1.txt', first_layer_weights)
            np.savetxt(path_+ '/biasHL1.txt', first_layer_biases)
            np.savetxt(path_+ '/weightsCL.txt', out_layer_weights)
            np.savetxt(path_+ '/biasCL.txt', out_layer_biases)

        return test