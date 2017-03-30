import os,sys,inspect,argparse,collections ; 

import GeneralFunctions as GF ; 
from data4keras import X_y_dataHandler

class Optimization_Pipeline ():
    def __init__ (self, args):
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
        train_data_obj = X_y_dataHandler()
        train_data_obj.load_data_set(self.PARAMS["train_filename"])
        # ----------------------------------
        devel_data_obj = X_y_dataHandler()
        devel_data_obj.load_data_set(self.PARAMS["devel_filename"])
        # ----------------------------------
        test_data_obj = X_y_dataHandler()
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
        train_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=args.padding_side)
        # ----------------------------------
        devel_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=args.padding_side)
        # ----------------------------------
        test_data_obj.make_numpy_arrays(X_used_row_len, y_max_value, padding_side=args.padding_side)
        # ----------------------------------
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
    
    def Run_Optimization_Pipeline(self):
        self.__LoadData__()

if __name__ == "__main__":
    default_logfile_address =os.path.dirname(os.path.realpath(__file__))+"/LOGS/"+GF.DATETIME_GetNowStr()+".txt"
    parser = argparse.ArgumentParser(description='keras_4_annotations.py')
    parser.add_argument('-data_folder', type=str, help='Location of the data folder', default='/home/hanmoe/text-classification/annotation/DATA')
    parser.add_argument('-ann_set', type=str, help='What annotation set to use, choices={"kipu", "sekavuus", "infektio"}', choices=['kipu', 'sekavuus', 'infektio'], required=True)
    parser.add_argument('-ann_type', type=str, help='Train on sentence or document level, choices={"sent", "doc"}', choices=['sent', 'doc'], required=True)
    parser.add_argument('-save_folder', type=str, help='A new folder with model and log file will be created here.', default='MODELS')
    parser.add_argument('-word_embeddings', type=str, help='Filename of the pre-created word embeddings to use for the X data.', default=None)
    parser.add_argument('-lemma_embeddings', type=str, help='Filename of the pre-created lemma embeddings to use for the X data.', default=None)
    parser.add_argument('-pos_embeddings', type=str, help='Filename of the pre-created pos embeddings to use for the X data.', default=None)
    parser.add_argument('-normalize_embeddings', type=int, help='Wether or not to normalize the loaded pre-created word embeddings; default=1 (True)', choices=[0, 1], default=1)
    parser.add_argument('-nb_epoch', type=int, help='Number of epochs, default=10', default=10)
    parser.add_argument('-fit_verbose', type=int, help='Verbose during training, 0=silent, 1=normal, 2=minimal; default=1', choices=[0, 1, 2], default=1)
    parser.add_argument('-padding_side', type=str, help='From what side to do the padding, choices={"right", "left"}; default="left"', choices=['right', 'left'], default='left')
    parser.add_argument('-logfileaddress', type=str, help='LogFile path and name.', choices=['right', 'left'], default=default_logfile_address)
    
    args = parser.parse_args(sys.argv[1:])
    OP = Optimization_Pipeline (args) 
    OP.Run_Optimization_Pipeline()
    OP.__exit__()

