import keras
import time
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from dataset.DigitsSymbolsDataset import DigitsSymbolsDataset

K.clear_session()

class DigitsSymbolsClassifierCNN:

    def __init__(self):

        self.__X_train_set = None
        self.__Y_train_set = None
        self.__X_test_set = None
        self.__Y_test_set = None
        self.__model = Sequential()
        self.__score = None

    def __load_data(self):

        ds = DigitsSymbolsDataset()
        (self.__X_train_set, self.__Y_train_set), \
        (self.__X_test_set, self.__Y_test_set) = ds.load_data()

    def __preprocess_data(self):

        self.__X_train_set = self.__X_train_set.reshape(self.__X_train_set.shape[0], 28, 28, 1)
        self.__X_test_set = self.__X_test_set.reshape(self.__X_test_set.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

        # Convert class vectors to binary class matrices
        num_classes = 18
        self.__Y_train_set = keras.utils.to_categorical(self.__Y_train_set, num_classes)
        self.__Y_test_set = keras.utils.to_categorical(self.__Y_test_set, num_classes)

    def __build_model(self):

        self.__model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
        self.__model.add(Conv2D(64, (3, 3), activation='relu'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))
        self.__model.add(Dropout(0.25))
        self.__model.add(Flatten())
        self.__model.add(Dense(256, activation='relu'))
        self.__model.add(Dropout(0.5))
        self.__model.add(Dense(num_classes, activation='softmax'))

        self.__model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    def __train_model(self):

        hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
        print('The model has successfully trained')

    def __evaluate_model(self):

        self.__score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def build_digits_symbols_classifier(self):

        self.__load_data()
        self.__preprocess_data()
        self.__build_model()
        self.__train_model()
        self.__evaluate_model()

        return self.__score[1], self.__model    # accuracy, model



def build_digits_symbols_classifier(batch_size, epochs):

    # ACCURACY ########################################################
    accuracy = score[1]
    ###################################################################

    return accuracy, model

if __name__ == '__main__':
    ##  PARSING COMMAND LINE ARGUMENTS FOR GETTING 'batch size' AND 'number of epochs' #####
    try:
        m_batch_size = int(sys.argv[1])
        m_epochs = int(sys.argv[2])
    except Exception:
        print("[ERROR]: Needed arguments wasn't set")
    ########################################################################################
    ## BUILD AND SAVE THE SYMBOLS/DIGITS CLASSIFIER ########################################
    try:
        start_time = time.time()
        accuracy, digits_symbols_classifier = build_digits_symbols_classifier(batch_size=m_batch_size, epochs=m_epochs)
        digits_symbols_classifier.save('./classifiers/digits_symbols_cnn_classif_' + str(m_batch_size) + '_' + str(m_epochs) + '.h5')
        print('Saving the model as digits_symbols_cnn_classif.h5')
        print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        print('[ERROR]:', e)
    #########################################################################################

    print("[INFO]: Finishing program...")
