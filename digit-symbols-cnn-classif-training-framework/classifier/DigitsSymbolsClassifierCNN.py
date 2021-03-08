import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


K.clear_session()

class DigitsSymbolsClassifierCNN:

    def __init__(self, dataset, batch_size, epochs):

        self.__dataset = dataset
        self.__X_train_set = None
        self.__Y_train_set = None
        self.__X_test_set = None
        self.__Y_test_set = None
        self.__model = Sequential()
        self.__score = None
        self.__num_classes = 18
        self.__batch_size = batch_size
        self.__epochs = epochs

    def __load_data(self):

        (self.__X_train_set, self.__Y_train_set), \
        (self.__X_test_set, self.__Y_test_set) = self.__dataset.load_data()

    def __preprocess_data(self):

        self.__X_train_set = self.__X_train_set.reshape(self.__X_train_set.shape[0], 28, 28, 1)
        self.__X_test_set = self.__X_test_set.reshape(self.__X_test_set.shape[0], 28, 28, 1)

        # Convert class vectors to binary class matrices
        self.__Y_train_set = keras.utils.to_categorical(self.__Y_train_set, self.__num_classes)
        self.__Y_test_set = keras.utils.to_categorical(self.__Y_test_set, self.__num_classes)

    def __build_model(self):

        input_shape = (28, 28, 1)

        self.__model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
        self.__model.add(Conv2D(64, (3, 3), activation='relu'))
        self.__model.add(MaxPooling2D(pool_size=(2, 2)))
        self.__model.add(Dropout(0.25))
        self.__model.add(Flatten())
        self.__model.add(Dense(256, activation='relu'))
        self.__model.add(Dropout(0.5))
        self.__model.add(Dense(self.__num_classes, activation='softmax'))

        self.__model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    def __train_model(self):

        self.__model.fit(self.__X_train_set,
                                self.__Y_train_set,
                                batch_size=self.__batch_size,
                                epochs=self.__epochs,
                                verbose=1,
                                validation_data=(self.__X_test_set, self.__Y_test_set))

        print('The model has been trained successfully')

    def __evaluate_model(self):

        self.__score = self.__model.evaluate(self.__X_test_set, self.__Y_test_set, verbose=0)
        print('Test loss:', self.__score[0])
        print('Test accuracy:', self.__score[1])

    def build_digits_symbols_classifier(self):

        self.__load_data()
        self.__preprocess_data()
        self.__build_model()
        self.__train_model()
        self.__evaluate_model()

        return self.__score, self.__model
