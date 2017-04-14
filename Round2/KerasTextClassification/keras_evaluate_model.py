from __future__ import print_function
from __future__ import division

import argparse
from keras.models import load_model
import sys
import numpy as np
from sklearn.metrics import f1_score
from data4keras import X_y_dataHandler


def evaluate_model(test_data_filename, model_filename, batch_size, ann_set, predict_negatives=0, true_threshold=0.5):


    print('\nLoading model "' + model_filename + '" ...')
    model = load_model(model_filename)


    # word + lemma + pos
    X_row_len = [X_shape[1] for X_shape in model.input_shape][0] # All input layers should have the same shape!
    """
    # word only
    X_row_len = model.input_shape[1]
    """

    y_max_value = model.output_shape[1]


    #if X_row_len < batch_size:
    #    X_row_len = batch_size

    print('X:', X_row_len, 'y:', y_max_value)

    test_data_obj = X_y_dataHandler(ann_set, predict_negatives)
    test_data_obj.load_data_set(test_data_filename)
    test_data_obj.make_numpy_arrays(X_row_len, y_max_value)


    #print(model.batch_input_shape())


    #f1_score = f1_score(y_true, y_pred, average=average)

    print('\nPredicting ...')

    # word + lemma + pos
    y_predicted_np_array = model.predict([test_data_obj.get_X_word_np_array(), test_data_obj.get_X_lemma_np_array(), test_data_obj.get_X_pos_np_array()], batch_size=batch_size)
    """
    # word only
    y_predicted_np_array = model.predict(test_data_obj.get_X_word_np_array(), batch_size=batch_size)
    """

    print('SHAPE test_data_obj.get_y_n_hot_np_array():', test_data_obj.get_y_n_hot_np_array().shape)
    print('SHAPE y_predicted_np_array:', y_predicted_np_array.shape)

    # Fetch the gold data
    gold_np_array = test_data_obj.get_y_n_hot_np_array()
    #if not test_data_obj.include_o_labels:
    #    gold_np_array = np.delete(gold_np_array, test_data_obj.o_label_id - 1, 1)

    print('Calculating scores ...')
    bool_predicted_np_array = np.zeros(y_predicted_np_array.shape, dtype=np.int32)
    for i in range(0, y_predicted_np_array.shape[0]):
        for j in range(0, y_predicted_np_array.shape[1]):
            #if test_data_obj.include_negatives or j+1 != test_data_obj.o_label_id:
            if y_predicted_np_array[i, j] >= true_threshold:
                bool_predicted_np_array[i, j] = 1

    #if not test_data_obj.include_o_labels:
    #    bool_predicted_np_array = np.delete(bool_predicted_np_array, test_data_obj.o_label_id - 1, 1)

    #print('PREDICTED')
    #print(bool_predicted_np_array) #-------
    #print('GOLD')
    #print(test_data_obj.get_y_n_hot_np_array()) #------

    assert test_data_obj.get_y_n_hot_np_array().shape == bool_predicted_np_array.shape

    f1_score_macro = f1_score(test_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average='macro')
    print('f1_score_macro', f1_score_macro)
    f1_score_micro = f1_score(test_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average='micro')
    print('f1_score_micro', f1_score_micro)
    f1_score_weighted = f1_score(test_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average='weighted')
    print('f1_score_weighted', f1_score_weighted)
    f1_score_samples = f1_score(test_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average='samples')
    print('f1_score_samples', f1_score_samples)

    f1_score_class_list = f1_score(test_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average=None)
    print('\nf1 score for individual classes')

    if not test_data_obj.include_o_labels:
        i_class = 0
        for i in range(0, len(f1_score_class_list)):
            class_f1_score = f1_score_class_list[i]
            i_class += 1
            if i_class == test_data_obj.o_label_id:
                print('\t' + str(i_class) + ': ' + 'O label, not included/calculated!')
                i_class += 1
            print('\t' + str(i_class) + ': ' + str(class_f1_score))
    else:
        for i_class, class_f1_score in enumerate(f1_score_class_list):
            print('\t' + str(i_class + 1) + ': ' + str(class_f1_score))



if __name__ == "__main__":
    """
    ## LOCAL TESTING ###################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='keras_evaluate_model.py')
    parser.add_argument('-model', type=str, help='Keras model to load', default='models/run-kipu-sent-batch_size100-nb_epoch1-pre_embeddingsFalse/model.epoch.0.h5') #required=True)
    parser.add_argument('-test', type=str, help='Filename for test data to load.', default='data/kipu/sent/sent-test-annotations.txt') #required=True)
    parser.add_argument('-batch_size', type=int, help='Size of batches; default=100', default=100)
    parser.add_argument('-ann_set', type=str, help='What annotation set to use, choices={"kipu", "sekavuus", "infektio"}', choices=['kipu', 'sekavuus', 'infektio'], default='kipu') #required=True)
    parser.add_argument('-negatives', type=int, help='Include negative O labels in training?; default=1 (True)', choices=[0, 1], default=1)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    """
    ## EVEX RUN ########################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='keras_evaluate_model.py')
    parser.add_argument('-model', type=str, help='Keras model to load', required=True)
    parser.add_argument('-test', type=str, help='Filename for test data to load.', required=True)
    parser.add_argument('-batch_size', type=int, help='Size of batches; default=100', default=100)
    parser.add_argument('-ann_set', type=str, help='What annotation set to use, choices={"kipu", "sekavuus", "infektio"}', choices=['kipu', 'sekavuus', 'infektio'], required=True)
    parser.add_argument('-negatives', type=int, help='Include negative O labels in training?; default=0 (False)', choices=[0, 1], default=0)
    ####################################%%%%%%%%%%%%%%%%%%%%%

    args = parser.parse_args(sys.argv[1:])


    print('\nStart ... ')

    evaluate_model(test_data_filename=args.test, model_filename=args.model, batch_size=args.batch_size, ann_set=args.ann_set, predict_negatives=args.negatives)

    print('\nDone!')

"""
Kipu:

Classes
-------
1       Voimakkuus      2023
2       Kipuun_liittyva_asia    7785
3       Hoidon_onnistuminen     721
4       Kipu    3397
5       Laatu   597
6       Kivunhoito      4540
7       O       706438
8       Toistuva_tilanne        1934
9       Potentiaalinen_kipu     1288
10      Implisiittinen_kipu     1235
11      Tilanne 799
12      Sijainti        2329
13      Toimenpide      4244
14      Suunnitelma     2923
15      Aika    4759
16      Ohjeistus       69


Kipu SENT test set:
-------------------
f1_score_macro 0.650159718117
f1_score_micro 0.952008518763
f1_score_weighted 0.947406768905

f1 score for individual classes
1: 0.711111111111
2: 0.508196721311
3: 0.666666666667
4: 0.911368015414
5: 0.441988950276
6: 0.824228028504
7: 0.986670062829
8: 0.737142857143
9: 0.903361344538
10: 0.59807073955
11: 0.155172413793
12: 0.7744
13: 0.621749408983
14: 0.828282828283
15: 0.734146341463
16: 0.0

=================================

Kipu DOC test set:
lstm_out_dim = 400
------------------
f1_score_macro 0.459013930098
f1_score_micro 0.60278551532
f1_score_weighted 0.571876086906

f1 score for individual classes
1: 0.606593406593
2: 0.587699316629
3: 0.0449438202247
4: 0.757363253857
5: 0.0
6: 0.709433962264
7: 0.714285714286
8: 0.632352941176
9: 0.539772727273
10: 0.320346320346
11: 0.0
12: 0.501240694789
13: 0.616966580977
14: 0.647619047619
15: 0.665605095541
16: 0.0


=================================


Kipu DOC test set:
lstm_out_dim = 300
------------------
PADDING FROM LEFT:
f1_score_macro 0.44368259309
f1_score_micro 0.587672052663
f1_score_weighted 0.552118386418

f1 score for individual classes
        1: 0.629464285714
        2: 0.579075425791
        3: 0.0
        4: 0.748137108793
        5: 0.108108108108
        6: 0.723076923077
        7: 0.674329501916
        8: 0.592
        9: 0.440816326531
        10: 0.163043478261
        11: 0.030303030303
        12: 0.435294117647
        13: 0.641095890411
        14: 0.709401709402
        15: 0.624775583483
        16: 0.0


PADDING FROM LEFT (run 2):
f1_score_macro 0.410823887055
f1_score_micro 0.551263902932
f1_score_weighted 0.515119441406

f1 score for individual classes
        1: 0.565789473684
        2: 0.577215189873
        3: 0.0
        4: 0.693215339233
        5: 0.0
        6: 0.597701149425
        7: 0.649606299213
        8: 0.643356643357
        9: 0.142857142857
        10: 0.286995515695
        11: 0.0
        12: 0.530612244898
        13: 0.593659942363
        14: 0.696428571429
        15: 0.595744680851
        16: 0.0


PADDING FROM RIGHT:
f1_score_macro 0.38229001295
f1_score_micro 0.545138133565
f1_score_weighted 0.493838084127

f1 score for individual classes
        1: 0.607142857143
        2: 0.591346153846
        3: 0.0
        4: 0.741085271318
        5: 0.0
        6: 0.538043478261
        7: 0.745222929936
        8: 0.620689655172
        9: 0.0
        10: 0.1875
        11: 0.0
        12: 0.374558303887
        13: 0.475524475524
        14: 0.623762376238
        15: 0.611764705882
        16: 0.0



"""


"""
[{'class_name': 'Merge', 'config': {'layers': [{'class_name': 'Sequential', 'config': [{'class_name': 'Embedding', 'config': {'input_length': 49, 'W_constraint': None, 'name': u'embedding_1', 'activity_regularizer': None, 'trainable': True, 'init': 'uniform', 'input_dtype': 'int32', 'mask_zero': True, 'batch_input_shape': (None, 49), 'W_regularizer': None, 'dropout': 0.2, 'input_dim': 217, 'output_dim': 300}}, {'class_name': 'LSTM', 'config': {'inner_activation': 'hard_sigmoid', 'trainable': True, 'inner_init': 'orthogonal', 'output_dim': 600, 'unroll': False, 'consume_less': u'cpu', 'init': 'glorot_uniform', 'dropout_U': 0.2, 'input_dtype': 'float32', 'b_regularizer': None, 'input_length': None, 'dropout_W': 0.2, 'activation': 'tanh', 'stateful': False, 'batch_input_shape': (None, None, 300), 'U_regularizer': None, 'name': u'lstm_1', 'go_backwards': False, 'input_dim': 300, 'return_sequences': False, 'W_regularizer': None, 'forget_bias_init': 'one'}}]}, {'class_name': 'Sequential', 'config': [{'class_name': 'Embedding', 'config': {'input_length': 49, 'W_constraint': None, 'name': u'embedding_2', 'activity_regularizer': None, 'trainable': True, 'init': 'uniform', 'input_dtype': 'int32', 'mask_zero': True, 'batch_input_shape': (None, 49), 'W_regularizer': None, 'dropout': 0.2, 'input_dim': 206, 'output_dim': 300}}, {'class_name': 'LSTM', 'config': {'inner_activation': 'hard_sigmoid', 'trainable': True, 'inner_init': 'orthogonal', 'output_dim': 600, 'unroll': False, 'consume_less': u'cpu', 'init': 'glorot_uniform', 'dropout_U': 0.2, 'input_dtype': 'float32', 'b_regularizer': None, 'input_length': None, 'dropout_W': 0.2, 'activation': 'tanh', 'stateful': False, 'batch_input_shape': (None, None, 300), 'U_regularizer': None, 'name': u'lstm_2', 'go_backwards': False, 'input_dim': 300, 'return_sequences': False, 'W_regularizer': None, 'forget_bias_init': 'one'}}]}, {'class_name': 'Sequential', 'config': [{'class_name': 'Embedding', 'config': {'input_length': 49, 'W_constraint': None, 'name': u'embedding_3', 'activity_regularizer': None, 'trainable': True, 'init': 'uniform', 'input_dtype': 'int32', 'mask_zero': True, 'batch_input_shape': (None, 49), 'W_regularizer': None, 'dropout': 0.2, 'input_dim': 15, 'output_dim': 300}}, {'class_name': 'LSTM', 'config': {'inner_activation': 'hard_sigmoid', 'trainable': True, 'inner_init': 'orthogonal', 'output_dim': 600, 'unroll': False, 'consume_less': u'cpu', 'init': 'glorot_uniform', 'dropout_U': 0.2, 'input_dtype': 'float32', 'b_regularizer': None, 'input_length': None, 'dropout_W': 0.2, 'activation': 'tanh', 'stateful': False, 'batch_input_shape': (None, None, 300), 'U_regularizer': None, 'name': u'lstm_3', 'go_backwards': False, 'input_dim': 300, 'return_sequences': False, 'W_regularizer': None, 'forget_bias_init': 'one'}}]}], 'name': u'merge_1', 'concat_axis': -1, 'mode_type': 'raw', 'dot_axes': -1, 'mode': u'concat', 'output_shape': None, 'output_shape_type': 'raw'}}, {'class_name': 'Dense', 'config': {'W_constraint': None, 'b_constraint': None, 'name': u'dense_1', 'activity_regularizer': None, 'trainable': True, 'init': 'glorot_uniform', 'bias': True, 'input_dim': None, 'b_regularizer': None, 'W_regularizer': None, 'activation': 'linear', 'output_dim': 8}}, {'class_name': 'Activation', 'config': {'activation': 'sigmoid', 'trainable': True, 'name': u'activation_1'}}]
"""