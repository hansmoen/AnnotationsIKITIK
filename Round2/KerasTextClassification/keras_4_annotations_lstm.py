'''Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''

from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Merge, LSTM, SimpleRNN, GRU
from keras.layers.wrappers import Bidirectional
# from keras.datasets import imdb
from keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint
from itertools import islice
import argparse
import sys
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import w2v_handler
from data4keras import X_y_dataHandler



class MyFileLog(Callback):
    '''
    Creates a log file with information about the run.

    TODO: Legg til mer info om tidsintervall og resterende model parametere
    '''
    def __init__(self, write_to_filename):
        super(Callback, self).__init__()
        self.write_to_filename = write_to_filename
        self.metrics = None
        self.run_start_dtm = None
        self.current_epoch_start_dtm = None

        # Clear log file in case it already exists
        f = open(self.write_to_filename, 'w')
        f.close()

    def on_train_begin(self, logs={}):
        self.run_start_dtm = datetime.now()
        self.metrics = self.params.get('metrics')
        # with open(self.write_to_filename, 'a') as f:
        #    f.write('nb_epoch: %s - nb_sample: %s\n\n' % (self.params.get('nb_epoch'), self.params.get('nb_sample')))

    def on_train_end(self, logs={}):
        run_end_dtm = datetime.now()
        diff = relativedelta(run_end_dtm, self.run_start_dtm)
        dtm_end_string = ('%s %02d:%02d:%02d' % (str(run_end_dtm.date()), run_end_dtm.hour, run_end_dtm.minute, run_end_dtm.second))
        with open(self.write_to_filename, 'a') as f:
            f.write('\nRun ended: ' + dtm_end_string)
            f.write('\nTotal run time: %s days - %s hours - %s minutes - %s seconds' % (
            diff.days, diff.hours, diff.minutes, diff.seconds))

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch_start_dtm = datetime.now()
        with open(self.write_to_filename, 'a') as f:
            #f.write('Epoch %s/%s (%s/%s)\n' % (epoch, self.params['nb_epoch'] - 1, epoch + 1, self.params['nb_epoch']))
            f.write('Epoch %s/%s\n' % (epoch + 1, self.params['nb_epoch']))

    def on_epoch_end(self, epoch, logs={}):
        delta = datetime.now() - self.current_epoch_start_dtm
        sec_diff = delta.seconds + delta.microseconds / 1E6
        with open(self.write_to_filename, 'a') as f:
            f.write(str(sec_diff) + 's - ')
            for i, p in enumerate(self.metrics):
                # f.write(p + ': ' + str(logs.get(p)) + (' - ' if (i+1) < len(self.metrics) else '\n'))
                f.write(('%s: %.04f' % (p, logs.get(p))) + (' - ' if (i + 1) < len(self.metrics) else '\n'))

    def append(self, string):
        with open(self.write_to_filename, 'a') as f:
            f.write(string)


if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='keras_4_annotations.py')
    """
    ## LOCAL TESTING ###################%%%%%%%%%%%%%%%%%%%%%
    parser.add_argument('-data_folder', type=str, help='Location of the data folder', default='data')
    parser.add_argument('-ann_set', type=str, help='What annotation set to use, choices={"kipu", "sekavuus", "infektio"}', choices=['kipu', 'sekavuus', 'infektio'], default='kipu')
    parser.add_argument('-ann_type', type=str, help='Train on sentence or document level, choices={"sent", "doc"}', choices=['sent', 'doc'], default='sent')
    parser.add_argument('-save_folder', type=str, help='A new folder with model and log file will be created here.', default='models')
    parser.add_argument('-word_embeddings', type=str, help='Filename of the pre-created word embeddings to use for the X data.', default=None)
    parser.add_argument('-lemma_embeddings', type=str, help='Filename of the pre-created lemma embeddings to use for the X data.', default=None)
    parser.add_argument('-pos_embeddings', type=str, help='Filename of the pre-created pos embeddings to use for the X data.', default=None)
    parser.add_argument('-normalize_embeddings', type=int, help='Wether or not to normalize the loaded pre-created word embeddings; default=1 (True)', choices=[0, 1], default=1)
    parser.add_argument('-batch_size', type=int, help='Size of batches; default=100', default=100)
    parser.add_argument('-nb_epoch', type=int, help='Number of epochs, default=1', default=1)
    parser.add_argument('-fit_verbose', type=int, help='Verbose during training, 0=silent, 1=normal, 2=minimal; default=1', choices=[0, 1, 2], default=1)
    parser.add_argument('-padding_side', type=str, help='From what side to do the padding, choices={"right", "left"}; default="left"', choices=['right', 'left'], default='left')
    parser.add_argument('-negatives', type=int, help='Include negative O labels in training?; default=1 (True)', choices=[0, 1], default=1)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    """
    ## EVEX RUN ########################%%%%%%%%%%%%%%%%%%%%%
    parser.add_argument('-data_folder', type=str, help='Location of the data folder', default='/home/hanmoe/annotation/text-classification/DATA')
    parser.add_argument('-ann_set', type=str, help='What annotation set to use, choices={"kipu", "sekavuus", "infektio"}', choices=['kipu', 'sekavuus', 'infektio'], required=True)
    parser.add_argument('-ann_type', type=str, help='Train on sentence or document level, choices={"sent", "doc"}', choices=['sent', 'doc'], required=True)
    parser.add_argument('-save_folder', type=str, help='A new folder with model and log file will be created here.', default='MODELS')
    parser.add_argument('-word_embeddings', type=str, help='Filename of the pre-created word embeddings to use for the X data.', default=None)
    parser.add_argument('-lemma_embeddings', type=str, help='Filename of the pre-created lemma embeddings to use for the X data.', default=None)
    parser.add_argument('-pos_embeddings', type=str, help='Filename of the pre-created pos embeddings to use for the X data.', default=None)
    parser.add_argument('-normalize_embeddings', type=int, help='Wether or not to normalize the loaded pre-created word embeddings; default=1 (True)', choices=[0, 1], default=1)
    parser.add_argument('-batch_size', type=int, help='Size of batches; default=100', default=100)
    parser.add_argument('-nb_epoch', type=int, help='Number of epochs, default=10', default=10)
    parser.add_argument('-fit_verbose', type=int, help='Verbose during training, 0=silent, 1=normal, 2=minimal; default=1', choices=[0, 1, 2], default=1)
    parser.add_argument('-padding_side', type=str, help='From what side to do the padding, choices={"right", "left"}; default="left"', choices=['right', 'left'], default='left')
    parser.add_argument('-negatives', type=int, help='Include negative O labels in training?; default=0 (False)', choices=[0, 1], default=0)
    ####################################%%%%%%%%%%%%%%%%%%%%%

    args = parser.parse_args(sys.argv[1:])

    print("Start ... ")

    train_filename = args.data_folder + '/' + args.ann_set + '/' + args.ann_type + '/' + args.ann_type + '-train-annotations.txt'
    devel_filename = args.data_folder + '/' + args.ann_set + '/' + args.ann_type + '/' + args.ann_type + '-devel-annotations.txt'
    test_filename = args.data_folder + '/' + args.ann_set + '/' + args.ann_type + '/' + args.ann_type + '-test-annotations.txt'

    #################################
    X_lower_row_len = 1 # Lower text length threshold
    X_upper_row_len = 400 # Upper text length threshold
    X_used_row_len = -1
    default_embeddings_dim = 300  # default size of the used word embeddings when no pre-created embeddings model is given
    word_embeddings_dim = lemma_embeddings_dim = pos_embeddings_dim = default_embeddings_dim
    lstm_out_dim = 300  # embeddings_dim
    #################################


    dtm_start = datetime.now()
    dtm_start_string = ('%s %02d:%02d:%02d' % (str(dtm_start.date()), dtm_start.hour, dtm_start.minute, dtm_start.second))

    word_embeddings_model = None
    lemma_embeddings_model = None
    pos_embeddings_model = None
    embeddings_used_string = 'False'
    if args.word_embeddings and args.lemma_embeddings and args.pos_embeddings:
        # FOR GUIDE ON WORD VECTORS, SEE: https://github.com/fchollet/keras/issues/853
        print('Loading pre-created embedding models to use as weights ... ')

        try:
            word_embeddings_model = w2v_handler.W2vModel()
            word_embeddings_model.load_w2v_model(args.word_embeddings, binary=True)
            word_embeddings_dim = word_embeddings_model.get_dim()

            lemma_embeddings_model = w2v_handler.W2vModel()
            lemma_embeddings_model.load_w2v_model(args.lemma_embeddings, binary=True)
            lemma_embeddings_dim = lemma_embeddings_model.get_dim()

            pos_embeddings_model = w2v_handler.W2vModel()
            pos_embeddings_model.load_w2v_model(args.pos_embeddings, binary=True)
            pos_embeddings_dim = pos_embeddings_model.get_dim()

            embeddings_used_string = 'True'

            if (args.normalize_embeddings):
                embeddings_used_string += '_norm'
        except:
            print('No embeddings model found in: ' + args.embeddings + '\nStopping the run!')
            sys.exit(-1)

            word_embeddings_model = lemma_embeddings_model = pos_embeddings_model = None
            embeddings_used_string = 'False'
            word_embeddings_dim = lemma_embeddings_dim = pos_embeddings_dim = default_embeddings_dim

    # lstm_output_layer_size = embeddings_dim  # Same as its input, OK?

    run_name = args.ann_set + '-' + args.ann_type + '-batch_size' + str(args.batch_size) + '-nb_epoch' + str(args.nb_epoch) + '-pre_embeddings' + embeddings_used_string
    # run_name = ('%s-%02d_%02d_%02d' % (str(dtm_start.date()).replace("-", "_"), dtm_start.hour, dtm_start.minute, dtm_start.second))



    run_save_folder = args.save_folder + '/run-' + run_name
    if os.path.isdir(args.save_folder):
        try:
            os.stat(run_save_folder)
        except:
            os.mkdir(run_save_folder)
    else:
        sys.exit('save_folder "' + args.save_folder + '" does not exist, exiting!')

    # log_filename = args.save_folder + '/log-' + dtm_start_string + '.txt'



    # Get information about the data set
    print('Fetching information about the data set ... ')
    # ----------------------------------
    train_data_obj = X_y_dataHandler(args.ann_set, args.negatives)
    train_data_obj.load_data_set(train_filename)
    # ----------------------------------
    devel_data_obj = X_y_dataHandler(args.ann_set, args.negatives)
    devel_data_obj.load_data_set(devel_filename)
    # ----------------------------------
    test_data_obj = X_y_dataHandler(args.ann_set, args.negatives)
    test_data_obj.load_data_set(test_filename)
    # ----------------------------------

    X_word_max_value = max([train_data_obj.get_X_max_word_value(), devel_data_obj.get_X_max_word_value(), test_data_obj.get_X_max_word_value()])
    X_lemma_max_value = max([train_data_obj.get_X_max_lemma_value(), devel_data_obj.get_X_max_lemma_value(), test_data_obj.get_X_max_lemma_value()])
    X_pos_max_value = max([train_data_obj.get_X_max_pos_value(), devel_data_obj.get_X_max_pos_value(), test_data_obj.get_X_max_pos_value()])
    y_max_value = max([train_data_obj.get_y_max_value(), devel_data_obj.get_y_max_value(), test_data_obj.get_y_max_value()])
    # ----------------------------------
    X_data_max_row_len = max([train_data_obj.get_X_max_len(), devel_data_obj.get_X_max_len(), test_data_obj.get_X_max_len()])
    if X_data_max_row_len <= X_lower_row_len:
        X_used_row_len = X_lower_row_len
    elif X_data_max_row_len >= X_upper_row_len:
        X_used_row_len = X_upper_row_len
    else:
        X_used_row_len = X_data_max_row_len
    # ----------------------------------
    train_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=args.padding_side)
    # ----------------------------------
    devel_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=args.padding_side)
    # ----------------------------------
    test_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=args.padding_side)
    # ----------------------------------
    train_data_size = train_data_obj.get_size()
    devel_data_size = devel_data_obj.get_size()
    test_data_size = test_data_obj.get_size()

    print('X_words_max_value:', X_word_max_value)
    print('X_lemma_max_value:', X_lemma_max_value)
    print('X_pos_max_value:', X_pos_max_value)
    print('X_used_row_length:', X_used_row_len)
    print('X_data_max_row_len:', X_data_max_row_len)
    print('y_max_value:', y_max_value)
    print('train_data_size:', train_data_size)
    print('devel_data_size:', devel_data_size)
    print('test_data_size:', test_data_size)
    print('padding_side:', args.padding_side)


    ###########################################################
    # Log stuff ###############################################
    file_log = MyFileLog(run_save_folder + '/log-' + run_name + '.txt')
    file_log.append('Run started: %s' % (dtm_start_string))

    file_log.append('\n\nX word max value: %i' % (X_word_max_value))
    file_log.append('\nX lemma max value: %i' % (X_lemma_max_value))
    file_log.append('\nX pos max value: %i' % (X_pos_max_value))
    file_log.append('\nX used row length: %i' % (X_used_row_len))
    file_log.append('\nX max row length: %i' % (X_data_max_row_len))
    file_log.append('\ny max value: %i' % (y_max_value))
    file_log.append('\ny padding side: %s' % (args.padding_side))

    file_log.append('\n\nTrain data: %s' % (train_filename))
    file_log.append('\n\tsize: %s' % (train_data_size))

    file_log.append('\nDevel data: %s' % (devel_filename))
    file_log.append('\n\tsize: %s' % (devel_data_size))

    file_log.append('\nTest data: %s' % (test_filename))
    file_log.append('\n\tsize: %s' % (test_data_size))

    file_log.append('\n\nBatch size: %s' % (args.batch_size))
    file_log.append('\nEpochs: %s' % (args.nb_epoch))
    file_log.append('\nUsing pre-created word embeddings: %s' % (embeddings_used_string))
    file_log.append('\n\n')
    ###########################################################

    # == Word =================================================
    word_weights = None
    if word_embeddings_model:
        print('Extracting word embeddings from the embeddings model ...')
        embeddings_count = max(X_word_max_value + 1, len(word_embeddings_model.get_vocab()) + 1)
        embedding_weights = np.zeros((embeddings_count, word_embeddings_dim))
        for word_placeholder in word_embeddings_model.get_vocab():
            # print(word_placeholder) #----------
            try:
                word_placeholder_as_int = int(word_placeholder)
                if word_placeholder_as_int <= embeddings_count:
                    if args.normalize_embeddings:
                        embedding_weights[word_placeholder, :] = w2v_handler.norm(
                            word_embeddings_model.get_vec(word_placeholder))
                    else:
                        embedding_weights[word_placeholder, :] = word_embeddings_model.get_vec(word_placeholder)
            except ValueError:
                pass
        word_weights = [embedding_weights]
    # == Lemma ================================================
    lemma_weights = None
    if lemma_embeddings_model:
        print('Extracting lemma embeddings from the embeddings model ...')
        embeddings_count = max(X_lemma_max_value + 1, len(lemma_embeddings_model.get_vocab()) + 1)
        embedding_weights = np.zeros((embeddings_count, lemma_embeddings_dim))
        for word_placeholder in lemma_embeddings_model.get_vocab():
            # print(word_placeholder) #----------
            try:
                word_placeholder_as_int = int(word_placeholder)
                if word_placeholder_as_int <= embeddings_count:
                    if args.normalize_embeddings:
                        embedding_weights[word_placeholder, :] = w2v_handler.norm(
                            lemma_embeddings_model.get_vec(word_placeholder))
                    else:
                        embedding_weights[word_placeholder, :] = lemma_embeddings_model.get_vec(word_placeholder)
            except ValueError:
                pass
        lemma_weights = [embedding_weights]
    # == PoS ==================================================
    pos_weights = None
    if pos_embeddings_model:
        print('Extracting PoS embeddings from the embeddings model ...')
        embeddings_count = max(X_pos_max_value + 1, len(pos_embeddings_model.get_vocab()) + 1)
        embedding_weights = np.zeros((embeddings_count, pos_embeddings_dim))
        for word_placeholder in pos_embeddings_model.get_vocab():
            # print(word_placeholder) #----------
            try:
                word_placeholder_as_int = int(word_placeholder)
                if word_placeholder_as_int <= embeddings_count:
                    if args.normalize_embeddings:
                        embedding_weights[word_placeholder, :] = w2v_handler.norm(
                            pos_embeddings_model.get_vec(word_placeholder))
                    else:
                        embedding_weights[word_placeholder, :] = pos_embeddings_model.get_vec(word_placeholder)
            except ValueError:
                pass
        pos_weights = [embedding_weights]
    # =========================================================


    print('\nBuild model ...')


    # word + lemma + pos
    word_model = Sequential()
    word_model.add(Embedding(input_dim=X_word_max_value + 1, output_dim=word_embeddings_dim, input_length=X_used_row_len, weights=word_weights, dropout=0.2, trainable=True, mask_zero=True))
    word_model.add(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2))
    #word_model.add(Bidirectional(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2)))

    lemma_model = Sequential()
    lemma_model.add(Embedding(input_dim=X_lemma_max_value + 1, output_dim=lemma_embeddings_dim, input_length=X_used_row_len, weights=lemma_weights, dropout=0.2, trainable=True, mask_zero=True))
    lemma_model.add(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2))
    #lemma_model.add(Bidirectional(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2)))

    pos_model = Sequential()
    pos_model.add(Embedding(input_dim=X_pos_max_value + 1, output_dim=pos_embeddings_dim, input_length=X_used_row_len, weights=pos_weights, dropout=0.2, trainable=True, mask_zero=True))
    pos_model.add(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2))
    #pos_model.add(Bidirectional(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2)))

    merged = Merge([word_model, lemma_model, pos_model], mode='concat')

    #TODO: PROV A DIREKTE KONKATINERE ALLE TRE SETT FOR LSTM, OG TREN EN LSTM!

    final_model = Sequential()
    final_model.add(merged)
    #final_model.add(Dense(output_dim=(3*lstm_out_dim)))
    final_model.add(Dense(output_dim=y_max_value))
    final_model.add(Activation('sigmoid'))

    # Farrokh sier: vurder optimizer='scd'
    #final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # binary_crossentropy, categorical_crossentropy
    """
    # word only
    final_model = Sequential()
    final_model.add(Embedding(input_dim=X_word_max_value + 1, output_dim=word_embeddings_dim, input_length=X_used_row_len, weights=word_weights, dropout=0.2, trainable=True, mask_zero=True))
    final_model.add(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2))
    final_model.add(Dense(output_dim=y_max_value))
    final_model.add(Activation('sigmoid'))
    final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    """
    # From https://keras.io/getting-started/sequential-model-guide : For a multi-class classification problem, use: model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


    print('\nTrain ...')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')  # mode='max'
    save_model_checkpoint = ModelCheckpoint(run_save_folder + '/model.epoch.{epoch:d}.h5', monitor='val_loss',
                                            verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    # word + lemma + pos
    final_model.fit([train_data_obj.get_X_word_np_array(), train_data_obj.get_X_lemma_np_array(), train_data_obj.get_X_pos_np_array()],
                     train_data_obj.get_y_n_hot_np_array(),
                     batch_size=args.batch_size, nb_epoch=args.nb_epoch, callbacks=[early_stop, file_log, save_model_checkpoint],
                     validation_data=([devel_data_obj.get_X_word_np_array(), devel_data_obj.get_X_lemma_np_array(), devel_data_obj.get_X_pos_np_array()], devel_data_obj.get_y_n_hot_np_array()),
                     verbose=args.fit_verbose, shuffle=True)
    """
    # word only
    final_model.fit(train_data_obj.get_X_word_np_array(),
                    train_data_obj.get_y_n_hot_np_array(),
                    batch_size=args.batch_size, nb_epoch=args.nb_epoch,
                    callbacks=[early_stop, file_log, save_model_checkpoint],
                    validation_data=(devel_data_obj.get_X_word_np_array(), devel_data_obj.get_y_n_hot_np_array()),
                    verbose=args.fit_verbose, shuffle=True)
    """

    print('\nFinally, evaluate on test data ...')

    # word + lemma + pos
    eval_results = final_model.evaluate([test_data_obj.get_X_word_np_array(), test_data_obj.get_X_lemma_np_array(), test_data_obj.get_X_pos_np_array()], test_data_obj.get_y_n_hot_np_array(), batch_size=args.batch_size, verbose=args.fit_verbose)
    for i in range(0, len(final_model.metrics_names)):
        res_str = str('%s: %f' % (final_model.metrics_names[i], eval_results[i]))
        print(res_str)
        file_log.append('\n\n' + res_str)
    """
    # word only
    eval_results = final_model.evaluate(test_data_obj.get_X_word_np_array(), test_data_obj.get_y_n_hot_np_array(), batch_size=args.batch_size, verbose=args.fit_verbose)
    for i in range(0, len(final_model.metrics_names)):
        res_str = str('%s: %f' % (final_model.metrics_names[i], eval_results[i]))
        print(res_str)
        file_log.append('\n\n' + res_str)
    """
    #print('Ending ...')

    #file_log.append('\n\nTest score: %s' % (score))
    #file_log.append('\nTest accuracy: %s' % (acc))

    print('\nDone!')

'''
Info regarding running Keras on Taito-GPU (csc) nodes:
https://github.com/TurkuNLP/SRNNMT/blob/master/README.csc
https://github.com/TurkuNLP/SRNNMT/blob/master/train_bsub.sh
'''




