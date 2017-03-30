""" By Farrokh ... General usage functions ... """

import datetime , shutil; 

def DATETIME_GetNowStr():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") ;  

def OS_IsDirectory (FolderAddress):
    return shutil.os.path.isdir (FolderAddress)

def FILE_CheckFileExists (fname):
    return shutil.os.path.isfile(fname); 

def NVLR (S, Cnt):
    if S == None:
        return " " * Cnt ;
    if len(S) < Cnt:
        return str(S) + (" " * (Cnt - len(S)));
    else:
        return S[0:Cnt]

def NVLL (S, Cnt):
    if S == None:
        return " " * Cnt ;
    if len(S) < Cnt:
        return (" " * (Cnt - len(S))) + str(S) ;
    else:
        return S[0:Cnt]

def f_round(number, NP=3):
    NP = str(NP)
    return ('{:.' + NP + 'f}').format (number)
