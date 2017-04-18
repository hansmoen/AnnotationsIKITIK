class ANN_Architecture_Builder:
    def __init__ (self, PARAMS , lp , PROGRAM_Halt, RandomSeed=None):
        self.PARAMS = PARAMS
        self.lp = lp  
        self.PROGRAM_Halt = PROGRAM_Halt
        self.RandomSeed = RandomSeed if RandomSeed <> None else 1337 ; 
        self.WhichFeaturesToUse = {}

        self.WhichFeaturesToUse["words"]  = False 
        self.WhichFeaturesToUse["lemmas"] = False 
        self.WhichFeaturesToUse["postgs"] = False 
        
    def Hans_1(self,hyper_params):
        #<<<CRITICAL>>> : Setting np random seed everytime BEFORE IMPORTING FROM KERAS!
        self.lp ("Building Neural Network Model. RandomSeed:" + str(self.RandomSeed) + "  , Please wait ..."); 
        import numpy as np ; 
        np.random.seed (self.RandomSeed) ; 
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Embedding, Merge
        from keras.layers import LSTM

        self.WhichFeaturesToUse["words"]  = True
        self.WhichFeaturesToUse["lemmas"] = True
        self.WhichFeaturesToUse["postgs"] = True
        
        H_word_embeddings_dim  = hyper_params["wed"]
        H_lemma_embeddings_dim = hyper_params["led"]
        H_pos_embeddings_dim   = hyper_params["ped"]
        H_lstm_out_dim         = hyper_params["lsd"]
        H_dropout_W            = hyper_params["dw"]
        H_dropout_U            = hyper_params["du"]
        
        word_weights = lemma_weights = pos_weights = None 
        
        word_model = Sequential()
        word_model.add(Embedding(input_dim=self.PARAMS["X_word_max_value"] + 1, output_dim= H_word_embeddings_dim, input_length=self.PARAMS["X_used_row_len"], weights=word_weights, dropout=0.2, trainable=True, mask_zero=True))
        word_model.add(LSTM(output_dim=H_lstm_out_dim, dropout_W=H_dropout_W, dropout_U=H_dropout_U))
    
        lemma_model = Sequential()
        lemma_model.add(Embedding(input_dim=self.PARAMS["X_lemma_max_value"] + 1, output_dim=H_lemma_embeddings_dim, input_length=self.PARAMS["X_used_row_len"], weights=lemma_weights, dropout=0.2, trainable=True, mask_zero=True))
        lemma_model.add(LSTM(output_dim=H_lstm_out_dim, dropout_W=H_dropout_W, dropout_U=H_dropout_U))
    
        pos_model = Sequential()
        pos_model.add(Embedding(input_dim=self.PARAMS["X_pos_max_value"] + 1, output_dim=H_pos_embeddings_dim, input_length=self.PARAMS["X_used_row_len"], weights=pos_weights, dropout=0.2, trainable=True, mask_zero=True))
        pos_model.add(LSTM(output_dim=H_lstm_out_dim, dropout_W=H_dropout_W, dropout_U=H_dropout_U))
    
    
        merged = Merge([word_model, lemma_model, pos_model], mode='concat')
    
        #TODO: PROV A DIREKTE KONKATINERE ALLE TRE SETT FOR LSTM, OG TREN EN LSTM!
    
        final_model = Sequential()
        final_model.add(merged)
        final_model.add(Dense(output_dim=self.PARAMS["y_max_value"]))
        #model.add(Activation('softmax'))
        final_model.add(Activation('sigmoid'))
        return final_model , self.WhichFeaturesToUse

