# Module with dataset preparation function


import pandas as pd
import numpy as np
# from collections import Counter
# import datetime
# import re


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def trainingset_preparation(DataFrame=pd.DataFrame()):
    
    '''
       This function prepares the training set for pre-processing.
       
       Input: 
             DataFrame = .xlsx file with all covariates as in the reference dataset.
             If no dataframe is passed to the function, the reference dataset
             will be loaded from '..URL to be added..'.
    '''
    
    
    # Load ref. dataset
    StrPath = '/Users/Riccardo/TriesteProject/Covid19/Datasets/SecondWave_Citofluorimetria/FlowCitometryData.xlsx' #URL or local path
    DataRef = pd.read_excel(StrPath, engine='openpyxl')
    ColumnsRef = DataRef.columns.values.copy()
    
    
    # Assign dataset to be used as training set
    if not DataFrame.empty:
        
        # create local copy
        Data = DataFrame.copy()
        Columns = Data.columns.values.copy()
        
        # check consistency with ref. dataset
        if np.array_equal(Columns, ColumnsRef):
            pass
        else:
            StrError1 = '; '.join(Columns)+';'
            StrError2 = '; '.join(ColumnsRef)+';'
            raise ValueError('Wrong column format.\nInserted format:\n%s\nExpected format:\n%s' %(StrError1, StrError2))
    
    else:
        
        # use ref. dataframe
        Data = DataRef.copy()
        Columns = ColumnsRef.copy()
        
    
    # Return prepared dataset
    return Data


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #