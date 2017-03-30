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




    print('\nBuild model ...')



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




