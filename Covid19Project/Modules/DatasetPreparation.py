# Module with dataset preparation function


import pandas as pd
import numpy as np
import datetime
# from collections import Counter
# import re


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def trainingset_preparation(DataFrame=pd.DataFrame(), print_info=False):
    
    '''
       This function prepares the training set for pre-processing.
       Input: 
             DataFrame = .xlsx file with all covariates as in the reference dataset.
             If no dataframe is passed to the function, the reference dataset
             will be loaded from '..URL to be added..'.
    '''
    
    
    ## Load ref. dataset
    StrPath = 'https://RiccardoGM.github.io/Covid19Project/Data/FlowCitometryData.xlsx' #URL or local path
    DataRef = pd.read_excel(StrPath, engine='openpyxl')
    ColumnsRef = DataRef.columns.values.copy()
    
    
    ## Assign dataset to be used as training set
    if not DataFrame.empty:
        print('Preprocessing not yet available for generic training set. Using reference dataset instead...\n')
        # Use ref. dataframe
        Data = DataRef.copy()
        Columns = ColumnsRef.copy()
#         # Create local copy
#         Data = DataFrame.copy()
#         Columns = Data.columns.values.copy()
#         # Check consistency with ref. dataset
#         if np.array_equal(Columns, ColumnsRef):
#             pass
#         else:
#             StrError1 = '; '.join(Columns)+';'
#             StrError2 = '; '.join(ColumnsRef)+';'
#             raise ValueError('Wrong column format.\nInserted format:\n%s\nExpected format:\n%s' %(StrError1, StrError2))
    else:
        # Use ref. dataframe
        Data = DataRef.copy()
        Columns = ColumnsRef.copy()
    
    
    ## Dimensions of raw data
    N_samples = Data.shape[0]
    N_covariates = Data.shape[1]

    
    ## Check and fix data types, save dates
    Dates = []
    for i, element in enumerate(Columns):
        if 'date' in element:
            Dates.append(element)
            Data[element] = pd.to_datetime(Data[element], errors='coerce')
        else:
            Data[element] = pd.to_numeric(Data[element], errors='coerce')
            
            
    ## Drop columns with too many nans
    N_nans_threshold = N_samples / 2.
    mask_columns = Data.count(axis=0).values < N_nans_threshold
    Data.drop(columns=Data.columns[mask_columns], inplace=True)
    Columns = Data.columns.values.copy()
    N_covariates = Data.shape[1]
    
    
    ## Convert dates to reals
    ReferenceTime = datetime.datetime(2100, 1, 1, 1, 0) # datetime.datetime(1970, 1, 1, 1, 0) -> 0
    ReferenceTime = ReferenceTime.timestamp()
    for i in range(N_samples):
        for date in Dates:
            datetime_obj = Data.loc[i, date]
            if datetime_obj:
                Data.loc[i, date] = float(ReferenceTime - datetime_obj.timestamp())
    
    
    ## Check and fix data values
    '''
        Add checks on e.g. dates (not from future), sex/death \in (0, 1), etc.
        To be completed...
    '''
    date1 = 'hospitalization_date'
    date2 = 'death_date'
    mask = Data.loc[:, date1].values < Data.loc[:, date2].values
    Data.loc[mask, date1] = Data.loc[mask, date2].values
    Data.loc[mask, 'OS_days'] = 0
    
    
    ## Collect ranges info
    MinMaxInfo = {}
    for i, element in enumerate(Columns):
        MinMaxInfo[element] = {}
        if 'date' in element:
            MinMaxInfo[element]['min'] = []
            MinMaxInfo[element]['max'] = []
        else:
            MinMaxInfo[element]['min'] = np.nanmin(Data[element])
            MinMaxInfo[element]['max'] = np.nanmax(Data[element])
            
            
    ## Collect means info
    Age_threshold = 70
    Age_range_1 = 'All ages'
    Age_range_2 = 'Age >= %d' % Age_threshold
    Age_range_3 = 'Age < %d' % Age_threshold
    #
    Mask_age_2 = Data.loc[:, 'age'].values >= Age_threshold
    Mask_age_3 = Mask_age_2 == False
    Mask_death = Data.loc[:, 'death'].values == 1
    Mask_no_death = Mask_death == False
    #
    MeansInfo = {}
    for i, element in enumerate(Columns):
        MeansInfo[element] = {Age_range_1: {'All': [],
                                           'Death = 0': [],
                                           'Death = 1': []
                                          },
                              Age_range_2: {'All': [],
                                           'Death = 0': [],
                                           'Death = 1': []
                                           },
                              Age_range_3:  {'All': [],
                                           'Death = 0': [],
                                           'Death = 1': []
                                          }}
        if 'date' in element:
            pass
        else:
            v = Data[element].values
            #
            MeansInfo[element][Age_range_1]['All'] = np.nanmean(v)
            MeansInfo[element][Age_range_1]['Death = 0'] = np.nanmean(v[Mask_no_death])
            MeansInfo[element][Age_range_1]['Death = 1'] = np.nanmean(v[Mask_death])
            #
            MeansInfo[element][Age_range_2]['All'] = np.nanmean(v[Mask_age_2])
            MeansInfo[element][Age_range_2]['Death = 0'] = np.nanmean(v[(Mask_age_2) & (Mask_no_death)])
            MeansInfo[element][Age_range_2]['Death = 1'] = np.nanmean(v[(Mask_age_2) & (Mask_death)])
            #
            MeansInfo[element][Age_range_3]['All'] = np.nanmean(v[Mask_age_3])
            MeansInfo[element][Age_range_3]['Death = 0'] = np.nanmean(v[(Mask_age_3) & (Mask_no_death)])
            MeansInfo[element][Age_range_3]['Death = 1'] = np.nanmean(v[(Mask_age_3) & (Mask_death)])
            
            
    ## Print info about covariates
    if print_info:
        print('Description of training set:\n')
        for i, element in enumerate(Columns):
            if 'date' not in element:
                N_data = Data[element].count()
                print(element)
                str_to_show = 'Availability:\n%d/%d (%.0f%%)' % (N_data, N_samples, 100*N_data/N_samples)
                print(str_to_show)
                #
                str_to_show = 'Range:\n[%.1f, %.1f]' % (MinMaxInfo[element]['min'], MinMaxInfo[element]['max'])
                print(str_to_show)
                #
                Age_range = Age_range_1
                val_1 = MeansInfo[element][Age_range]['All']
                val_2 = MeansInfo[element][Age_range]['Death = 0']
                val_3 = MeansInfo[element][Age_range]['Death = 1']
                str_to_show = 'Means:\n%s  %.2f (All), %.2f (Death = 0), %.2f (Death = 1)' % (Age_range, val_1, val_2, val_3)
                print(str_to_show)
                Age_range = Age_range_2
                val_1 = MeansInfo[element][Age_range]['All']
                val_2 = MeansInfo[element][Age_range]['Death = 0']
                val_3 = MeansInfo[element][Age_range]['Death = 1']
                str_to_show = '%s %.2f (All), %.2f (Death = 0), %.2f (Death = 1)' % (Age_range, val_1, val_2, val_3)
                print(str_to_show)
                Age_range = Age_range_3
                val_1 = MeansInfo[element][Age_range]['All']
                val_2 = MeansInfo[element][Age_range]['Death = 0']
                val_3 = MeansInfo[element][Age_range]['Death = 1']
                str_to_show = '%s %.2f  (All), %.2f (Death = 0), %.2f (Death = 1)\n\n' % (Age_range, val_1, val_2, val_3)
                print(str_to_show)
        
    
    ## Add index column
    ID_values = list(map(lambda x: str(x), np.arange(1, N_samples+1, 1)))
    Data.insert(loc=N_covariates, column='ID', value=ID_values)
    
    
    ## Return prepared dataset
    return Data


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #