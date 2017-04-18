import os,sys,inspect,argparse,collections ; 

import GeneralFunctions as GF ; 
import Architectures as ARCHITECTURES 
from data4keras import X_y_dataHandler

class Optimization_Pipeline ():
    def __init__ (self, args):
        self.GLOBAL_BEST_DEVEL_PRED_RESULTS = []

        self.args = args 
        self._LogFileHandler = open (args.logfileaddress, "wt")
        self.lp ("Program started ...") 
        self.__validate_args__() 
        
        self.PARAMS = collections.OrderedDict()  
        self.PARAMS["train_filename"] = args.data_folder + '/' + args.ann_set + '/' + args.ann_type + '/' + args.ann_type + '-train-annotations.txt'
        self.PARAMS["devel_filename"] = args.data_folder + '/' + args.ann_set + '/' + args.ann_type + '/' + args.ann_type + '-devel-annotations.txt'
        self.PARAMS["test_filename" ] = args.data_folder + '/' + args.ann_set + '/' + args.ann_type + '/' + args.ann_type + '-test-annotations.txt'

        self.PARAMS["X_lower_row_len"] = 1 # Lower text length threshold
        self.PARAMS["X_upper_row_len"] = 400 # Upper text length threshold
        self.PARAMS["X_used_row_len"]  = -1
        self.PARAMS["default_embeddings_dim"] = 300  # default size of the used word embeddings when no pre-created embeddings model is given

        MSG = [""*80,"PARAMETERS:","-"*20]
        for key in self.PARAMS.keys():
            MSG.append (GF.NVLR (key,20) + " : " + str(self.PARAMS[key]))
        MSG.append ("*"*80)
        self.lp (MSG) 
        
    def __validate_args__(self):
        self.lp ("Validating args ...")
        D = self.args.__dict__
        MSG = [""*80,"Command-Line args:","-"*20]
        for key in sorted(D.keys()):
            MSG.append (GF.NVLR (key,20) + " : " + str(D[key]))
        MSG.append ("*"*80)
        self.lp (MSG) 
        
    def __exit__ (self):
        self.lp (["Running destructor and closing log file.","Exiting program.","END."]) ; 
        self._LogFileHandler.close (); 

    def lp (self, ARG_msg): #log and print message
        try:        
            LOCAL_CallerClassName = str(inspect.stack()[1][0].f_locals["self"].__class__) ; 
        except:
            LOCAL_CallerClassName = "" ; 
            
        LOCAL_CallerFunction = inspect.currentframe().f_back.f_code.co_name ; 
        if isinstance(ARG_msg, basestring):
            ARG_msg = [ARG_msg] ;
        HEADER = "[" + GF.DATETIME_GetNowStr() + "] [" + LOCAL_CallerClassName + "." + LOCAL_CallerFunction + "]: " ; 
        print HEADER ;
        self._LogFileHandler.write (HEADER+"\n") ; 
        for itemstr in ARG_msg:
            try:
                itemstr = str(itemstr).replace ('\r','\n'); 
            except:
                itemstr = itemstr.encode('utf-8').replace ('\r','\n')

            for item in itemstr.split ("\n"):
                if len(item)==0:
                    item = "-" ;
                item = "      "+item ; 
                print item ; 
                self._LogFileHandler.write (item+"\n") ;
        print "" ;
        self._LogFileHandler.write ("\n") ; 

    def PROGRAM_Halt (self, ARG_HaltErrMSG):
        PARAM_CallerFunctionName = inspect.currentframe().f_back.f_code.co_name ; 
        self.lp (["*"*80 , "HALT REQUESTED BY FUNCTION: " + PARAM_CallerFunctionName , "HALT MESSAGE: "+ ARG_HaltErrMSG , "HALTING PROGRAM!!!" , "*"*80]);
        self.__exit__ (); 
        sys.exit (-1); 
    
    def __LoadData__(self):
        self.lp ("Fetching information about the data set ...") 
        # ----------------------------------
        train_data_obj = X_y_dataHandler(ann_set=self.args.ann_set, include_o_labels=0)
        train_data_obj.load_data_set(self.PARAMS["train_filename"])
        # ----------------------------------
        devel_data_obj = X_y_dataHandler(ann_set=self.args.ann_set, include_o_labels=0)
        devel_data_obj.load_data_set(self.PARAMS["devel_filename"])
        # ----------------------------------
        test_data_obj = X_y_dataHandler(ann_set=self.args.ann_set, include_o_labels=0)
        test_data_obj.load_data_set(self.PARAMS["test_filename"])
        # ----------------------------------
        X_word_max_value = max([train_data_obj.get_X_max_word_value(), devel_data_obj.get_X_max_word_value(), test_data_obj.get_X_max_word_value()])
        X_lemma_max_value = max([train_data_obj.get_X_max_lemma_value(), devel_data_obj.get_X_max_lemma_value(), test_data_obj.get_X_max_lemma_value()])
        X_pos_max_value = max([train_data_obj.get_X_max_pos_value(), devel_data_obj.get_X_max_pos_value(), test_data_obj.get_X_max_pos_value()])
        y_max_value = max([train_data_obj.get_y_max_value(), devel_data_obj.get_y_max_value(), test_data_obj.get_y_max_value()])
        # ----------------------------------
        X_data_max_row_len = max([train_data_obj.get_X_max_len(), devel_data_obj.get_X_max_len(), test_data_obj.get_X_max_len()])
        if X_data_max_row_len <= self.PARAMS["X_lower_row_len"]:
            X_used_row_len = self.PARAMS["X_lower_row_len"]
        elif X_data_max_row_len >= self.PARAMS["X_upper_row_len"]:
            X_used_row_len = self.PARAMS["X_upper_row_len"]
        else:
            X_used_row_len = X_data_max_row_len
        # ----------------------------------
        train_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=self.args.padding_side)
        # ----------------------------------
        devel_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=self.args.padding_side)
        # ----------------------------------
        test_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=self.args.padding_side)
        # ----------------------------------
        # Need to check again due to potential removal of the O label column
        y_max_value = max([train_data_obj.get_y_max_value(), devel_data_obj.get_y_max_value(), test_data_obj.get_y_max_value()])

        train_data_size = train_data_obj.get_size()
        devel_data_size = devel_data_obj.get_size()
        test_data_size = test_data_obj.get_size()
        
        MSG = ["*"*80,"Information about data:" , "-"*30]
        MSG.append(GF.NVLR ('word max value' , 40) + ": " + str (X_word_max_value))
        MSG.append(GF.NVLR ('lemma max value', 40) + ": " + str (X_lemma_max_value))
        MSG.append(GF.NVLR ('pos max value'  , 40) + ": " + str (X_pos_max_value))
        MSG.append(GF.NVLR ('used row length', 40) + ": " + str (X_used_row_len))
        MSG.append(GF.NVLR ('max row length' , 40) + ": " + str (X_data_max_row_len))
        MSG.append(GF.NVLR ('max value'      , 40) + ": " + str (y_max_value))
    
        MSG.append(GF.NVLR ('Train data'     , 40) + ": " + self.PARAMS["train_filename"])
        MSG.append(GF.NVLR ('Train size'     , 40) + ": " + str(train_data_size))
        MSG.append("")

        MSG.append(GF.NVLR ('Devel data'     , 40) + ": " + self.PARAMS["devel_filename"])
        MSG.append(GF.NVLR ('Devel size'     , 40) + ": " + str(devel_data_size))
        MSG.append("")
    
        MSG.append(GF.NVLR ('Test data'      , 40) + ": " + self.PARAMS["test_filename"])
        MSG.append(GF.NVLR ('Test size'      , 40) + ": " + str(test_data_size))
        MSG.append("")
        self.lp (MSG)

        self.train_data_obj = train_data_obj
        self.devel_data_obj = devel_data_obj
        self.test_data_obj  = test_data_obj

        self.PARAMS["X_word_max_value" ]  = X_word_max_value
        self.PARAMS["X_lemma_max_value"]  = X_lemma_max_value
        self.PARAMS["X_pos_max_value"  ]  = X_pos_max_value
        self.PARAMS["X_used_row_len"   ]  = X_used_row_len
        self.PARAMS["y_max_value"      ]  = y_max_value       
    
    def __train__(self):
        ANN_INPUT  = []
        ANN_OUTPUT = self.train_data_obj.get_y_n_hot_np_array()
        
        if self.__WhichFeaturesToUse["words"] == True:
            ANN_INPUT.append (self.train_data_obj.get_X_word_np_array())
            
        if self.__WhichFeaturesToUse["lemmas"] == True:
            ANN_INPUT.append (self.train_data_obj.get_X_lemma_np_array())

        if self.__WhichFeaturesToUse["postgs"] == True:
            ANN_INPUT.append (self.train_data_obj.get_X_pos_np_array())
        
        H = self.__model.fit(ANN_INPUT,ANN_OUTPUT, batch_size= self.args.batch_size, nb_epoch=1, verbose=self.args.fit_verbose, shuffle=False)
        #self.lp ("Training loss: " + str(H.history)); 
        #self.TrainMetricLog.append (H)

    def __predict__(self):
        ANN_INPUT  = []
        
        if self.__WhichFeaturesToUse["words"] == True:
            ANN_INPUT.append (self.devel_data_obj.get_X_word_np_array())
            
        if self.__WhichFeaturesToUse["lemmas"] == True:
            ANN_INPUT.append (self.devel_data_obj.get_X_lemma_np_array())

        if self.__WhichFeaturesToUse["postgs"] == True:
            ANN_INPUT.append (self.devel_data_obj.get_X_pos_np_array())
        
        return self.__model.predict (ANN_INPUT)

    def __evaluate__(self, PRED): #, predict_o_labels=0):
        import numpy as np
        from sklearn.metrics import f1_score

        true_threshold = 0.5
        y_predicted_np_array = PRED 

        bool_predicted_np_array = np.zeros(y_predicted_np_array.shape, dtype=np.int32)
        for i in range(0, y_predicted_np_array.shape[0]):
            for j in range (0, y_predicted_np_array.shape[1]):
                if y_predicted_np_array[i, j] >= true_threshold:
                    #if predict_o_labels or j + 1 != self.devel_data_obj.o_label_id:
                    bool_predicted_np_array[i, j] = 1

        #devel_gold_np_array = self.devel_data_obj.get_y_n_hot_np_array()
        #if not predict_o_labels:
        #    bool_predicted_np_array = np.delete(bool_predicted_np_array, self.devel_data_obj.o_label_id - 1, 1)
        #    devel_gold_np_array = np.delete(devel_gold_np_array, self.devel_data_obj.o_label_id - 1, 1)

        #assert bool_predicted_np_array.shape == devel_gold_np_array.shape

        #f1_macro = f1_score(devel_gold_np_array, bool_predicted_np_array, average='macro')
        #f1_micro = f1_score(devel_gold_np_array, bool_predicted_np_array, average='micro')
        #f1_weighted = f1_score(devel_gold_np_array, bool_predicted_np_array, average='weighted')
        #f1_samples  = f1_score(devel_gold_np_array, bool_predicted_np_array, average='samples')

        assert bool_predicted_np_array.shape == self.devel_data_obj.get_y_n_hot_np_array().shape

        f1_macro = f1_score(self.devel_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average='macro')
        f1_micro = f1_score(self.devel_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average='micro')
        f1_weighted = f1_score(self.devel_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average='weighted')
        f1_samples  = f1_score(self.devel_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average='samples')

        self.PredMetricLog.append ([self.EpochNoCntr,f1_macro,f1_micro,f1_weighted,f1_samples])
        
        MSG = self.CurrentArchName 
        MSG += "\tEpoch: "+str(self.EpochNoCntr)
        MSG += "\tf1-macro: " + GF.f_round(f1_macro)
        MSG += "\tf1-micro: " + GF.f_round(f1_micro)
        MSG += "\tf1-weighted: " + GF.f_round(f1_weighted)
        MSG += "\tf1_samples: "   + GF.f_round(f1_samples)
        print MSG 
        
        """
        f1_score_class_list = f1_score(self.devel_data_obj.get_y_n_hot_np_array(), bool_predicted_np_array, average=None)
        for i_class, class_f1_score in enumerate(f1_score_class_list):
            print('\t' + str(i_class + 1) + ': ' + str(class_f1_score))
        """
    def Run_Optimization_Pipeline(self, Archictectures):
        self.__LoadData__()
        for arch in Archictectures:
            #1-Reset Evaluation 
            #self.TrainMetricLog = [] 
            self.PredMetricLog = []
            self.CurrentArchName = arch 
            
            #1-Build model
            ARCBuilder = ARCHITECTURES.ANN_Architecture_Builder (self.PARAMS , self.lp , self.PROGRAM_Halt) ; 
            self.__model , self.__WhichFeaturesToUse = eval ("ARCBuilder." + arch);
            
            #2-Compile model             
            #self.lp ("-"*30 + " COMPILING MODEL:" + arch)
            self.__model.compile (loss="binary_crossentropy", optimizer="adam") ; 
            #self.lp ("-"*30 + " DONE MODEL BUILDING " + "-" *30) 
            
            for EpochNo in range(self.args.nb_epoch):
                self.EpochNoCntr = EpochNo + 1 #when we train for 1st epoch (for the first time), this should be 1
                self.__train__() 
                PRED = self.__predict__()
                self.__evaluate__(PRED)
            
            #Select best result of this particular architecture ... 
            BestMeasure = 2 #Micro-FScore
            BestResults = sorted (self.PredMetricLog , key = lambda x: x[BestMeasure] , reverse=True)[0]
            self.GLOBAL_BEST_DEVEL_PRED_RESULTS.append ([arch]+BestResults)

            self.lp (["-"*80 , "BEST RESULTS so far:" , "-"*80])
            for best_result in sorted (self.GLOBAL_BEST_DEVEL_PRED_RESULTS, key = lambda x: x[BestMeasure+1] , reverse=True):
                self.lp (str(best_result)) 
            
if __name__ == "__main__":
    default_logfile_address =os.path.dirname(os.path.realpath(__file__))+"/LOGS/"+GF.DATETIME_GetNowStr()+".txt"
    parser = argparse.ArgumentParser(description='keras_4_annotations.py')
    parser.add_argument('-data_folder', type=str, help='Location of the data folder', default='/home/hanmoe/annotation/text-classification/DATA')
    parser.add_argument('-ann_set', type=str, help='What annotation set to use, choices={"kipu", "sekavuus", "infektio"}', choices=['kipu', 'sekavuus', 'infektio'], required=True)
    parser.add_argument('-ann_type', type=str, help='Train on sentence or document level, choices={"sent", "doc"}', choices=['sent', 'doc'], required=True)
    parser.add_argument('-save_folder', type=str, help='A new folder with model and log file will be created here.', default='MODELS')
    parser.add_argument('-word_embeddings', type=str, help='Filename of the pre-created word embeddings to use for the X data.', default=None)
    parser.add_argument('-lemma_embeddings', type=str, help='Filename of the pre-created lemma embeddings to use for the X data.', default=None)
    parser.add_argument('-pos_embeddings', type=str, help='Filename of the pre-created pos embeddings to use for the X data.', default=None)
    parser.add_argument('-normalize_embeddings', type=int, help='Wether or not to normalize the loaded pre-created word embeddings; default=1 (True)', choices=[0, 1], default=1)
    parser.add_argument('-nb_epoch', type=int, help='Number of epochs, default=10', default=10)
    parser.add_argument('-batch_size', type=int, help='Size of batches; default=100', default=100)
    parser.add_argument('-fit_verbose', type=int, help='Verbose during training, 0=silent, 1=normal, 2=minimal; default=1', choices=[0, 1, 2], default=1)
    parser.add_argument('-padding_side', type=str, help='From what side to do the padding, choices={"right", "left"}; default="left"', choices=['right', 'left'], default='left')
    parser.add_argument('-logfileaddress', type=str, help='LogFile path and name.', default=default_logfile_address)
    
    args = parser.parse_args(sys.argv[1:])
    OP = Optimization_Pipeline (args) 

    Archs  = [] 
    Archs += [ "Hans_1 ({'wed': 300 , 'led': 300, 'ped': 300, 'lsd':300 , 'dw': 0.2 , 'du': 0.01})" ]
    Archs += [ "Hans_1 ({'wed': 300 , 'led': 300, 'ped': 200, 'lsd':300 , 'dw': 0.2 , 'du': 0.01})" ]
    Archs += [ "Hans_1 ({'wed': 400 , 'led': 400, 'ped': 200, 'lsd':400 , 'dw': 0.2 , 'du': 0.01})" ]
    Archs += [ "Hans_1 ({'wed': 400 , 'led': 400, 'ped': 100, 'lsd':200 , 'dw': 0.2 , 'du': 0.01})" ]
    Archs += [ "Hans_1 ({'wed': 400 , 'led': 400, 'ped': 100, 'lsd':200 , 'dw': 0.2 , 'du': 0.01})" ]
    Archs += [ "Hans_1 ({'wed': 300 , 'led': 300, 'ped': 300, 'lsd':300 , 'dw': 0.5 , 'du': 0.01})" ]

    """
    for vd in range(50,301,50):
        for lsd in range(50,301,50):
            for dv in [0.1,0.2,0.3,0.4,0.5]:
                arch = "Hans_1 ({" 
                arch+= "'wed':"+str(vd)+","
                arch+= "'led':"+str(vd)+","
                arch+= "'ped':"+str(vd)+","
                arch+= "'lsd':"+str(lsd)+","
                arch+= "'dw' :"+str(dv)+","
                arch+= "'du' :"+str(dv)+"}"
                arch+= ")" 
                Archs.append (arch)
   
    """
  
    OP.args.nb_epoch = 10
    OP.args.fit_verbose = 0 #for taito execution 
    OP.Run_Optimization_Pipeline(Archs)
    OP.__exit__()

