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
from keras.layers import Dense, Dropout, Activation, Embedding, Merge
from keras.layers import LSTM, SimpleRNN, GRU
# from keras.datasets import imdb
from keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint
from itertools import islice
import argparse
import sys
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import w2v_handler

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


    #################################
    X_lower_row_len = 1 # Lower text length threshold
    X_upper_row_len = 400 # Upper text length threshold
    X_used_row_len = -1

    word_embeddings_dim = lemma_embeddings_dim = pos_embeddings_dim = default_embeddings_dim
    lstm_out_dim = 300  # embeddings_dim

    #################################



    # lstm_output_layer_size = embeddings_dim  # Same as its input, OK?

    run_name = args.ann_set + '-' + args.ann_type + 
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

    word_model = Sequential()
    word_model.add(Embedding(input_dim=X_word_max_value + 1, output_dim=word_embeddings_dim, input_length=X_used_row_len, weights=word_weights, dropout=0.2, trainable=True, mask_zero=True))
    word_model.add(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2))

    lemma_model = Sequential()
    lemma_model.add(Embedding(input_dim=X_lemma_max_value + 1, output_dim=lemma_embeddings_dim, input_length=X_used_row_len, weights=lemma_weights, dropout=0.2, trainable=True, mask_zero=True))
    lemma_model.add(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2))

    pos_model = Sequential()
    pos_model.add(Embedding(input_dim=X_pos_max_value + 1, output_dim=pos_embeddings_dim, input_length=X_used_row_len, weights=pos_weights, dropout=0.2, trainable=True, mask_zero=True))
    pos_model.add(LSTM(output_dim=lstm_out_dim, dropout_W=0.2, dropout_U=0.2))


    merged = Merge([word_model, lemma_model, pos_model], mode='concat')

    #TODO: PROV A DIREKTE KONKATINERE ALLE TRE SETT FOR LSTM, OG TREN EN LSTM!

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(output_dim=y_max_value))
    #model.add(Activation('softmax'))
    final_model.add(Activation('sigmoid'))
    # Try using different optimizers and different optimizer configs
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Farrokh sier: vurder optimizer='scd'

    # From https://keras.io/getting-started/sequential-model-guide : For a multi-class classification problem, use: model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


    print('\nTrain ...')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')  # mode='max'
    save_model_checkpoint = ModelCheckpoint(run_save_folder + '/model.epoch.{epoch:d}.h5', monitor='val_loss',
                                            verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    final_model.fit([train_data_obj.get_X_word_np_array(), train_data_obj.get_X_lemma_np_array(), train_data_obj.get_X_pos_np_array()],
                     train_data_obj.get_y_n_hot_np_array(),
                     batch_size=args.batch_size, nb_epoch=args.nb_epoch, callbacks=[early_stop, file_log, save_model_checkpoint],
                     validation_data=([devel_data_obj.get_X_word_np_array(), devel_data_obj.get_X_lemma_np_array(), devel_data_obj.get_X_pos_np_array()], devel_data_obj.get_y_n_hot_np_array()),
                     verbose=args.fit_verbose, shuffle=True)


    print('\nFinally, evaluate on test data ...')

    eval_results = final_model.evaluate([test_data_obj.get_X_word_np_array(), test_data_obj.get_X_lemma_np_array(), test_data_obj.get_X_pos_np_array()], test_data_obj.get_y_n_hot_np_array(), batch_size=args.batch_size, verbose=args.fit_verbose)
    for i in range(0, len(final_model.metrics_names)):
        res_str = str('%s: %f' % (final_model.metrics_names[i], eval_results[i]))
        print(res_str)
        file_log.append('\n\n' + res_str)


    #print('Ending ...')

    #file_log.append('\n\nTest score: %s' % (score))
    #file_log.append('\nTest accuracy: %s' % (acc))

    print('\nDone!')

'''
Info regarding running Keras on Taito-GPU (csc) nodes:
https://github.com/TurkuNLP/SRNNMT/blob/master/README.csc
https://github.com/TurkuNLP/SRNNMT/blob/master/train_bsub.sh
'''




