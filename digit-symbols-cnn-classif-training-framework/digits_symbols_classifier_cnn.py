import keras
import time
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from DigitsSymbolsDataset import DigitsSymbolsDataset

K.clear_session()

def build_digits_symbols_classifier(batch_size, epochs):
    ## LOAD DATA ######################################################
    ds = DigitsSymbolsDataset()
    (x_train, y_train), (x_test, y_test) = ds.load_data()
    ###################################################################

    ## PREPROCESS DATA ################################################
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Convert  class vectors to binary class matrices
    num_classes = 18
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    ###################################################################

    # MODEL BUILDING ##################################################
    # Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    ###################################################################

    # MODEL TRAINING ##################################################
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    print('The model has successfully trained')
    ###################################################################

    # MODEL EVALUATION ################################################
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    ###################################################################

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
