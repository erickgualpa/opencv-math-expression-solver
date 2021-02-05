from dataset.DigitsSymbolsDataset import DigitsSymbolsDataset
from classifier import DigitsSymbolsClassifierCNN

def parse_command_line_args():

    try:
        batch_size = int(sys.argv[1])
        epochs = int(sys.argv[2])
        return batch_size, epochs
    except Exception:
        print("[ERROR]: Needed arguments wasn't set")

def build_and_save_classifier(batch_size, epochs):

    try:
        start_time = time.time()
        ds = DigitsSymbolsDataset()
        dscCNN = DigitsSymbolsClassifierCNN(ds, batch_size, epochs)

        score, digits_symbols_classifier = dscCNN.build_digits_symbols_classifier()

        digits_symbols_classifier.save('./classifiers/digits_symbols_cnn_classif_' + str(m_batch_size) + '_' + str(m_epochs) + '.h5')
        print('Saving the model as digits_symbols_cnn_classif.h5')
        print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        print('[ERROR]:', e)

if __name__ == '__main__':

    batch_size, epochs = parse_command_line_args()
    build_and_save_classifier(batch_size, epochs)
    print("[INFO]: Finishing program...")