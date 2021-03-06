## Module description

'''
    This file contains all functions needed to run 
    classifications based on flow citometry data as
    in the reference dataset 'FlowCitometryData.xlsx'
'''


## import statements
import pandas as pd
import numpy as np
import datetime
from datetime import date
import re
import sys

# sklearn
from scipy.stats import kurtosistest
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def trainingset_preparation(DataFrame=pd.DataFrame(), print_info=False):
    
    '''
       This function prepares the training set for pre-processing.
       Input: 
             1) DataFrame: .xlsx file with all covariates as in the reference dataset.
                           If no dataframe is passed to the function, the reference dataset
                           will be loaded from path in StrPath.
             2) print_info: boolean - whether to show basic info about training set (True) or not (False).
    '''
    
    
    ## Load ref. dataset
    StrPath = 'https://RiccardoGM.github.io/Covid19Project/Data/FlowCitometryData_NewNames.xlsx' #URL or local path
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
        print('Available data:\n')
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
    return Data, MinMaxInfo


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def preprocessing(Data, MinMaxInfo, Data_test=pd.DataFrame()):
    
    '''
       This function pre-processes data for training and testing.
       Input: 
             1) Data: pandas DataFrame with all covariates ready for preprocessing.
             2) MinMaxInfo: dictionary with {'covariate name': {'min': [] or value, 'max': [] or value}}
    '''
    
    
    ## Create local copy
    Data_local = Data.copy()
    
    
    ## Datasets from input data
    Features = ['age', 
                'sex', 
                'LINFO/uL', 
                'CD4%',
                'CD4/uL', 
                'CD8%', 
                'CD8/uL', 
                'CD4/CD8', 
                'NK%', 
                'NK/uL',
                'B%', 
                '%CD4 DR', 
                '%CD8 DR',
                'NK LIKE%', 
                'RTE (%CD4)',
                'MONO/uL', 
                'MONO DR%', 
                'MONO DR IFI',
                'WBC/uL']
    
    # Excluded features: 'CD3%', 
    #                    'CD3/uL', 
    #                    'CD3 DR%', 
    #                    'CD3 DR/uL',
    #                    'B/uL', 
    #                    'RTE/uL', 
    #                    'CD8 DR%', 
    #                    'CD4 DR%'
    
    Features_cat = [col for col in Features if (np.isin(Data_local[col].dropna().unique(), [0, 1]).all())]
    Features_noncat = [col for col in Features if col not in Features_cat]
    Target = ['death']
    Dates = [col for col in Features if 'date' in col]
    #
    Data_X = Data_local.loc[:, Features].astype(float)
    Data_Y = Data_local.loc[:, Target].astype(float)
    Data_dates = Data_local.loc[:, Dates].astype(float)
    Data_ID = Data_local.loc[:, ['ID']]
    Data_Age = Data_local.loc[:, ['age']]
    # Test set
    if not Data_test.empty:
        Data_X_test = Data_test.loc[:, Features].astype(float)
        #Data_dates_test = Data_test.loc[:, Dates].astype(float) # Currently not required
        Data_ID_test = Data_test.loc[:, ['ID']]
        Data_Age_test = Data_test.loc[:, ['age']]
        intersection = set.intersection(set(Data_test.columns), set(Target))
        flag_test_target = False
        if len(intersection)>0:
            flag_test_target = True
            Data_Y_test = Data_test.loc[:, Target].astype(float)
            
    
    ## Apply x->log(1+x) where kurtosis is above threshold
    kurtosis_threshold = 6
    skew_threshold = -1.5
    X_kurtosis = kurtosistest(Data_X.values, axis=0, nan_policy='omit').statistic
    X_skew = Data_X.skew(axis=0)
    Features_LogProcessing = {}
    for i, element in enumerate(Features_noncat):
        Features_LogProcessing[element] = {'Reflection': False,
                                           'Log': False}
        if (X_kurtosis[i]>kurtosis_threshold):
            Features_LogProcessing[element]['Log'] = True
            if (X_skew[element]<skew_threshold):
                Features_LogProcessing[element]['Reflection'] = True
                max_val = MinMaxInfo[element]['max']
                Data_X.loc[:, element] = max_val - Data_X.loc[:, element].values
                # Test set
                if not Data_test.empty:
                    Data_X_test.loc[:, element] = np.max(max_val - Data_X_test.loc[:, element].values, 0)
            Data_X.loc[:, element] = np.log(1+Data_X.loc[:, element].values)
            # Test set
            if not Data_test.empty:
                Data_X_test.loc[:, element] = np.log(1+Data_X_test.loc[:, element].values)
            
            
    ## Standardization (before imputation)
    X = Data_X.loc[:, Features_noncat].values.copy()
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X = (X - X_mean) / X_std
    Data_X.loc[:, Features_noncat] = X
    # Test set
    if not Data_test.empty:
        X = Data_X_test.loc[:, Features_noncat].values.copy()
        X = (X - X_mean) / X_std
        Data_X_test.loc[:, Features_noncat] = X
    
    
    ## Imputation
    Imputer_knn = KNNImputer(n_neighbors=5, weights='distance')
    Data_X = pd.DataFrame(Imputer_knn.fit_transform(Data_X), columns=Features)
    # Test set
    if not Data_test.empty:
        Data_X_test = pd.DataFrame(Imputer_knn.transform(Data_X_test), columns=Features)
        
    
    ## Reverse standardization
    X = Data_X.loc[:, Features_noncat].values.copy()
    X = X * X_std + X_mean
    Data_X.loc[:, Features_noncat] = X
    # Test set
    if not Data_test.empty:
        X = Data_X_test.loc[:, Features_noncat].values.copy()
        X = X * X_std + X_mean
        Data_X_test.loc[:, Features_noncat] = X
        # re-assign Age if was missing
        mask = Data_Age_test.isnull().values.ravel()
        Data_Age_test.loc[mask, 'age'] = Data_X_test.loc[mask, 'age'].values.copy()
    
    
    ## Fix tails
    if Data_test.empty:
        X = Data_X.loc[:, Features_noncat].values.copy()
        X = fix_outliers(X, Features_noncat)
        Data_X.loc[:, Features_noncat] = X
    # Test set
    else:        
        X = Data_X.loc[:, Features_noncat].values.copy()
        X_test = Data_X_test.loc[:, Features_noncat].values.copy()
        X, X_test = fix_outliers(X, Features_noncat, X_test=X_test)
        Data_X.loc[:, Features_noncat] = X
        Data_X_test.loc[:, Features_noncat] = X_test
    
    
    ## Standardization (final)
    X = Data_X.loc[:, Features_noncat].values.copy()
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    Data_X.loc[:, Features_noncat] = X
    # Test set
    if not Data_test.empty:
        X = Data_X_test.loc[:, Features_noncat].values.copy()
        X = (X - X_mean) / X_std
        Data_X_test.loc[:, Features_noncat] = X
        
        
    ## Return preprocessed datasets
    returns = {}
    returns['Training set'] = {'X': Data_X,
                               'Y': Data_Y,
                               'ID': Data_ID,
                               'Age': Data_Age}
    # Test set
    if not Data_test.empty:
        returns['Test set'] = {'X': Data_X_test,
                               'ID': Data_ID_test,
                               'Age': Data_Age_test}
        if flag_test_target:
            returns['Test set']['Y'] = Data_Y_test
    return returns


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def fix_outliers(X_train, features, X_test=np.array([])):

    '''
       This function fixes outliers detected as follows:
       - for upper tail check if maxvalue - qantile99>prefactor*(quantile98 - quantile99)
       - for lower tail check if qantile01 - minvalue)>prefactor*(quantile02 - quantile01)       
       Input: 
             1) X_train: 2D np.array (n_samples x n_features).
             2) features: list with features names.
             3) X_test: 2D np.array with same format as X_train.
    '''
    
    
    ## Create local copy
    X_train_c = X_train.copy()
    if len(X_test)>0:
        X_test_c = X_test.copy()
        
    
    ## Check input format
    if len(X_train_c.shape)==2:
        N_features = X_train_c.shape[1]
        if len(X_test)>0:
            X_test_c = X_test_c.reshape(-1, N_features)
    else:
        raise Exception("Wrong input format. Accepted format: n_samples x n_features (2D array).")
        
    
    ## Parameters
    prefactor = 10
    delta_q = 0.01
    q_top1 = 1 - delta_q
    q_top2 = 1 - 2*delta_q
    q_min1 = delta_q
    q_min2 = 2*delta_q
    
    
    ## Fix one feature at a time
    flag_outliers = False
    features_with_outliers = []
    #
    for idx in range(N_features):
        feature = features[idx]
        if feature == 'sex':
            str_to_add = 'Sesso'
        elif feature == 'age':
            str_to_add = 'Anno di nascita - et\u00E0'
        else:
            str_to_add = feature
        x = X_train_c[:, idx]
        mask_notnull = pd.notnull(x)
        x = x[mask_notnull]
        
        # fix upper tail
        max_val = float(max(x))
        quantile_top1 = float(np.quantile(x, q=q_top1))
        quantile_top2 = float(np.quantile(x, q=q_top2))
        if(quantile_top1!=quantile_top2):
            if max_val-quantile_top1>prefactor*(quantile_top1-quantile_top2):
                mask_q1 = x>quantile_top1
                n_new_values = sum(mask_q1)
                new_values = np.random.uniform(low=quantile_top2, high=quantile_top1, size=n_new_values)
                x[mask_q1] = new_values     

        # fix lower tail
        min_val = float(min(x))
        quantile_min1 = float(np.quantile(x, q=q_min1))
        quantile_min2 = float(np.quantile(x, q=q_min2))
        if(quantile_min1!=quantile_min2):
            if quantile_min1-min_val>prefactor*(quantile_min2-quantile_min1):
                mask_q1 = np.array(x<quantile_min1)
                n_new_values = sum(mask_q1)
                new_values = np.random.uniform(low=quantile_min1, high=quantile_min2, size=n_new_values)
                x[mask_q1] = new_values
                
        X_train_c[mask_notnull, idx] = x
        
        # Test set
        if len(X_test)>0:
            x = X_test_c[:, idx]
            mask_notnull = pd.notnull(x)
            x = x[mask_notnull]
            
            # fix upper tail
            max_val = float(max(x))
            if max_val-quantile_top1>prefactor*(quantile_top1-quantile_top2):
                flag_outliers = True
                features_with_outliers.append('\u2022 %s (+)' % (str_to_add))
                mask_q1 = x>quantile_top1
                n_new_values = sum(mask_q1)
                new_values = np.random.uniform(low=quantile_top2, high=quantile_top1, size=n_new_values)
                x[mask_q1] = new_values
                
            # fix lower tail
            min_val = float(min(x))
            if quantile_min1-min_val>prefactor*(quantile_min2-quantile_min1):
                flag_outliers = True
                features_with_outliers.append('\u2022 %s (-)' % (str_to_add))
                mask_q1 = np.array(x<quantile_min1)
                n_new_values = sum(mask_q1)
                new_values = np.random.uniform(low=quantile_min1, high=quantile_min2, size=n_new_values)
                x[mask_q1] = new_values
                
            X_test_c[mask_notnull, idx] = x
            
    # Print warning
    if flag_outliers:
        #print('Warning! Outliers found in test set at:\n', features_with_outliers, '\n')
        str_to_show = '\n'.join(features_with_outliers)
        print('\nAttenzione! Outliers individuati nelle seguenti variabili:')
        print(str_to_show)
    
    if len(X_test)>0:
        return X_train_c, X_test_c
    else:
        return X_train_c


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def prediction(X_train, y_train, X_test=np.array([]), y_test=np.array([])):

    '''
       This function trains several models on training set and applies/shows the best one.
       Corrige: currently only a 2D SVC is implemented (data are projected onto plane 
       defined by a LR model (axis v_1) and first pca component on hyperplane perp. to v_1).
       Input: 
             1) X_train: 2D np.array (n_samples x n_features).
             2) y_train: 1D np.array (n_samples).
             3) X_test: 2D np.array (n_samples x n_features).
    '''
    
    
    ## Initialize models
    LR_1 = LogisticRegression(C=1.0, class_weight='balanced', penalty='l2')
    LR_2 = LogisticRegression(C=1.0, class_weight='balanced', penalty='l2')
    SVC_2D = SVC(C=1.0, class_weight='balanced', gamma='auto', kernel='rbf')
    
    
    ## Find first axis v_1 (LR_1 projection)
    y_LR_1 = LR_1.fit(X_train, y_train).predict(X_train)
    v_1 = LR_1.coef_.reshape(-1, 1)
    v_1_norm = np.sqrt(sum(v_1**2))
    v_1_matrix = np.repeat(v_1.reshape((1, -1)), X_train.shape[0], axis=0)
    X1D_1 = np.dot(X_train, v_1)
    X1D_1_matrix = np.repeat(X1D_1.reshape((-1, 1)), X_train.shape[1], axis=1)
    
    
    ## Data projection onto hyperplane perpendicular to v_1
    X_parallel = v_1_matrix * X1D_1_matrix / v_1_norm**2
    X_perpendicular = X_train - X_parallel
    
    
    ## Find second axis v_2 on hyperplane from misclassifications (LR_2 projection)
    mask_misclass = y_train != y_LR_1
    y_LR_2 = LR_2.fit(X_perpendicular[mask_misclass, :], y_train[mask_misclass]).predict(X_train)
    v_2 = LR_2.coef_.reshape(-1, 1)
    v_2_norm = np.sqrt(sum(v_2**2))
    X1D_2 = np.dot(X_train, v_2)
    
    
    ## Find first pca component on hyperplane from misclassified
    DR_pca = PCA(n_components=1)
    #X1D_2_pca = DR_pca.fit(X_perpendicular[mask_misclass, :], y_train[mask_misclass]).transform(X_train) # use missclassified data
    X1D_2_pca = DR_pca.fit(X_perpendicular, y_train).transform(X_train) # use all data
    
    
    ## SVC on both axis
    X_1 = X1D_1.reshape(-1, 1)
    X_2 = X1D_2_pca.reshape(-1, 1)
    X2D_train = np.concatenate((X_1, X_2), axis=1)
    SVC_2D.fit(X2D_train, y_train)
    y_SVC_2D = SVC_2D.predict(X2D_train)
    value_SVC_2D = SVC_2D.decision_function(X2D_train)
    
    
    ## Test set
    if len(X_test)>0:
        X1D_1_test = np.dot(X_test, v_1)
        X1D_2_pca_test = DR_pca.transform(X_test)
        X_1_test = X1D_1_test.reshape(-1, 1)
        X_2_test = X1D_2_pca_test.reshape(-1, 1)
        X2D_test = np.concatenate((X_1_test, X_2_test), axis=1)
        y_SVC_2D_test = SVC_2D.predict(X2D_test)
    else:
        X_1_test = np.array([])
        X_2_test = np.array([])
    
    
    ## Show scatterplot and info
    print('\nMappa di classificazione:')
    print(u'\u2022', 'variabile target: decesso')
    print(u'\u2022', 'zone blu e blu scuro: rischio elevato e molto elevato')
    print(u'\u2022', 'punti blu: pazienti deceduti')
    print(u'\u2022', 'zone rosse: rishio moderato')
    print(u'\u2022', 'punti rossi: pazienti non deceduti')
    print(u'\u2022', 'x: posizione del paziente in esame')
    classification_plot2D(X_1, X_2, y_train, SVC_2D, X_1_test, X_2_test, y_test)
    #
    if len(X_test)>0:
        flag = False
        if max(X_1_test) < min(X_1) or min(X_1_test) > max(X_1):
            flag = True
        if max(X_1_test) < min(X_1) or min(X_1_test) > max(X_1):
            flag = True
        if flag:
            print('Test fuori dall\' intervallo mostrato: predizione non attendibile\n')
        
    
    ## Print info
    if len(X_test)>0:
        prediction_str = ' '.join(map(str, y_SVC_2D_test.astype(int)))
        print('Predizione: %s' % prediction_str, '\n')
    n_msc = np.sum(y_SVC_2D + y_train == 1)
    #print('N misclassified =', n_msc, '(%.2f%%)'%(100*n_msc/len(y_train)), '\n')
    #print('F1 score: %.2f' % f1_score(y_train, y_SVC_2D), '\n')
    print(u'Sensibilit\u00E0: %.2f' % recall_score(y_train, y_SVC_2D), '\n')
    print('Valore predittivo positivo: %.2f' % precision_score(y_train, y_SVC_2D), '\n')
    #
    if len(y_test)>10 and np.sum(y_test)>0:
        print(u'Sensibilit\u00E0 test: %.2f' % recall_score(y_test, y_SVC_2D_test), '\n')
        print('Valore predittivo positivo test: %.2f' % precision_score(y_test, y_SVC_2D_test), '\n')
    
# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def age_masking(X_train, y_train, age_train, age_test):
    
    '''
       This function returns data masked by age. The lower bound is set according to min(age_test). 
       Input: 
             1) X_train: 2D np.array (n_samples x n_features).
             2) y_train: 1D np.array (n_samples).
             3) age_train: 1D np.array (n_samples).
             4) age_test: model trained (n_samples).
    '''
    
    
    ## Create local copies
    X_train_c = X_train.copy()
    y_train_c = y_train.copy()
    
    
    ## Define lower bound
    min_test_age = min(age_test)
    if min_test_age>90:
        lower_bound = 85
    elif min_test_age>=65:
        lower_bound = min_test_age - 5
    elif min_test_age>=60:
        lower_bound = 60
    else:
        lower_bound = min_test_age 
      
    
    ## Apply masking
    age_mask = age_train >= lower_bound
    X_train_c = X_train_c[age_mask, :]
    y_train_c = y_train_c[age_mask]
    
    
    return X_train_c, y_train_c


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def classification_plot2D(X_1, X_2, y, classifier, X_1_test=np.array([]), X_2_test=np.array([]), y_test=np.array([])):
    
    '''
       This function produces a scatterplot of the (2D) training data, 
       with a heatmap of the passed classifier.
       Input: 
             1) X_1: 1D np.array with feature 1 values.
             2) X_2: 1D np.array with feature 2 values.
             3) y: 1D np.array with target values.
             4) classifier: model trained on X_1, X_2 and y.
             5) X_1_test: 1D np.array with feature 1 test values.
             6) X_2_test: 1D np.array with feature 2 test values.
             7) y: 1D np.array with target test values.
    '''
    
    #SetPlotParams()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    
    ## Define grid
    h = .1 # .02  # step size in the mesh
    x_1_min, x_1_max = X_1.min() - 3., X_1.max() + 3.
    x_2_min, x_2_max = X_2.min() - 3., X_2.max() + 3.
    xx_1, xx_2 = np.meshgrid(np.arange(x_1_min, x_1_max, h),
                         np.arange(x_2_min, x_2_max, h))
    X = np.concatenate((X_1.reshape(-1, 1), X_2.reshape(-1, 1)), axis=1)


    ## Compute values for heatmap
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(np.c_[xx_1.ravel(), xx_2.ravel()])
    else:
        Z = classifier.predict_proba(np.c_[xx_1.ravel(), xx_2.ravel()])[:, 1]
    
    
    ## Put the result into a color plot
    Z = Z.reshape(xx_1.shape)
    min_lev = Z.min()
    max_lev = Z.max()
    ax.contourf(xx_1, xx_2, Z, cmap=cm, levels=[-2, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 2], alpha=.5) #levels=[-2, -0.2, -0.1, 0.1, 0.2, 2], 

    
    ## Plot the training points
    ax.scatter(X_1, X_2, c=y, cmap=cm_bright, edgecolors='face', s=7, alpha=0.7)

    ## Plot the test points
    if len(X_1_test)>0 and len(X_2_test)>0:
        if len(y_test)>0:
            ax.scatter(X_1_test, X_2_test, c=y_test, vmin=0., vmax=1., cmap=cm_bright, marker='x', edgecolors='face', s=80, alpha=1)
        else:
            ax.scatter(X_1_test, X_2_test, color='black', marker='x', s=80, linewidth=2., alpha=1)
    
    ax.set_xlim(x_1_min, x_1_max)
    ax.set_ylim(x_2_min, x_2_max)
    #ax.set_xticks(())
    #ax.set_yticks(())
    
    plt.tight_layout()
    plt.show() 
    

# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def run_classification(Data_test_input=pd.DataFrame()):
    
    
    ## Training set preparation
    Data_train, MinMaxInfo = trainingset_preparation(print_info=False)
    
    
    ## Test set
    input_list = ['LINFO/uL', 
                  'CD4%',
                  'CD4/uL', 
                  'CD8%', 
                  'CD8/uL', 
                  'CD4/CD8', 
                  'NK%', 
                  'NK/uL',
                  'B%', 
                  '%CD4 DR', 
                  '%CD8 DR',
                  'NK LIKE%', 
                  'RTE (%CD4)',
                  'MONO/uL', 
                  'MONO DR%', 
                  'MONO DR IFI',
                  'WBC/uL']
    # ID, Sex and Age are excluded from this list
    
    if Data_test_input.empty:
        
        input_data = []
        inserted_data = []

        ## ID
        inserted_value = input('Inserire ID\n')
        input_data.append(inserted_value)
        inserted_data.append(inserted_value)


        ## Sex
        flag = True
        while flag:
            inserted_value = input('Inserire sesso (M, F)\n')
            if inserted_value in ('M', 'F'):
                if inserted_value == 'M':
                    value = 0
                else:
                    value = 1
                flag = False
        input_data.append(value)
        inserted_data.append(inserted_value)


        ## Age
        flag = True
        while flag:
            inserted_value = input('Inserire anno di nascita (esempio: 1940)\n')
            if isfloat(inserted_value):
                year = int(inserted_value)
                if year>1889 and year<2021:
                    datetime_str = '01/07/' + inserted_value
                    datetime_obj = datetime.datetime.strptime(datetime_str, '%d/%m/%Y')
                    value = calculate_age(datetime_obj)
                    flag = False
                else:
                    print('Valore inserito fuori dal range [1890, 2020]')
            elif inserted_value == '':
                value = np.nan
                inserted_value = np.nan
                flag = False
        input_data.append(value)
        inserted_data.append(inserted_value)
        
        
        ## All remaining CitoFluo values
        for element in input_list:
            flag = True
            while flag:
                str_to_show = element
                min_val = MinMaxInfo[element]['min']
                max_val = MinMaxInfo[element]['max']
                inserted_value = input(u'Inserire %s - intervallo osservato: [%.2f, %.2f]\n' % (str_to_show, min_val, max_val))
                #
                if isfloat(inserted_value) or inserted_value=='':
                    if isfloat(inserted_value):
                        value = float(inserted_value)
                        if value>=0:
                            if '%' in element:
                                if value <= 100:
                                    flag = False
                            else:
                                flag = False
                    else:
                        value = np.nan
                        inserted_value = np.nan
                        flag = False
                #
            input_data.append(value)
            inserted_data.append(inserted_value)
        Data_test = pd.DataFrame([input_data], columns=['ID', 'sex', 'age', *input_list], index=[1])
    else:
        Data_test = Data_test_input.copy()
        Data_test = reshape_datafromfile(Data_test)
        Data_test, birth_year, sex = get_patient_data(Data_test)
        Data_test = Data_test.loc[:, ['ID', 'sex', 'age', *input_list]]
        inserted_data = list(Data_test.values.ravel())
        inserted_data[1] = sex # not a great solution, but does the job
        inserted_data[2] = birth_year # not a great solution, but does the job
    
    # Print insterted values
    print('\nDati paziente:')
    for i, element in enumerate(Data_test.columns):
        element_name = element
        #
        if element == 'sex':
            element_name = 'Sesso'
        elif element == 'age':
            element_name = 'Anno di nascita'
        #
        if pd.isnull(inserted_data[i]):
            str_to_show = '--'
        else:
            str_to_show = str(inserted_data[i])
        str_to_show = '%s: %s' % (element_name, str_to_show)
        print(u'\u2022', str_to_show)
        
        
    ## Preprocessing
    Preprocessed_data_dict = preprocessing(Data_train, MinMaxInfo, Data_test=Data_test)
    X_train = Preprocessed_data_dict['Training set']['X'].values
    y_train = Preprocessed_data_dict['Training set']['Y'].values.ravel()
    age_train = Preprocessed_data_dict['Training set']['Age'].values.ravel()
    X_test = Preprocessed_data_dict['Test set']['X'].values
    age_test = Preprocessed_data_dict['Test set']['Age'].values.ravel()
    if 'Y' in Preprocessed_data_dict['Test set'].keys():
        y_test = Preprocessed_data_dict['Test set']['Y'].values.ravel()
    else:
        y_test = np.array([])

        
    ## Age masking - before or after preprocessing?
    X_train, y_train = age_masking(X_train, y_train, age_train, age_test)
    
    
    ## Prediction
    prediction(X_train, y_train, X_test=X_test, y_test=y_test)
    

# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def reshape_datafromfile(Data_test):
    
    '''
       This function checks if the uploaded file (assigned to Data_test) containes the features used to make predictions, then organizes Data_test as a pandas dataframe with: n_rows = n_patients, n_col = n_features.
       NB: this reshaping is not strictly required to make predictions about patients.
       Input: 
             1) Data_test: pandas DataFrame with no header, n_rows = n_features (including ID), n_col = n_patients.
    '''
    
    Features = ['ID',
                'LINFO/uL',
                'CD3%',
                'CD3/uL',
                'CD4%',
                'CD4/uL', 
                'CD8%', 
                'CD8/uL', 
                'CD4/CD8', 
                'NK%', 
                'NK/uL',
                'B%',
                'B/uL',
                'CD3 DR%',
                'CD3 DR/uL',
                'CD4 DR%',
                '%CD4 DR',
                'CD8 DR%',
                '%CD8 DR',
                'NK LIKE%', 
                'RTE (%CD4)',
                'RTE/uL',
                'MONO/uL', 
                'MONO DR%', 
                'MONO DR IFI',
                'WBC/uL']
    
    
    ## Create local copy
    Data_test_local = Data_test.copy()
    
    
    ## Check features
    if not set(Data_test_local.index).issubset(Features):
        error_message = 'Formato file non valido. Caricare file Excel Workbook (.xlsx) con le seguenti righe:\n'+'\n'.join(Features)
        raise Exception(error_message)
      
    
    ## Reshape data
    try:
        Data_test_local.loc['ID', :] = Data_test_local.loc['ID', :].values.astype(int).astype(str)
    except:
        Data_test_local.loc['ID', :] = Data_test_local.loc['ID', :].values.astype(str)
    ID_test = Data_test_local.loc['ID', :].values
    Data_test_local.drop(index='ID', inplace=True)
    Data_test_local = pd.DataFrame(Data_test_local.values.transpose(), columns=Data_test_local.index.values, index=ID_test)
    
    
    ## Return data
    return Data_test_local


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def get_patient_data(Data_test):
    
    '''
       This function asks the ID of the patient to test, then looks for ID in Data_test and checks the associated values.
       It throws an error if any of the values is not numeric, positive or equal to zero.
       It returns the data associated with ID if no errors occurred.
       Input: 
             1) Data_test: pandas DataFrame with n_rows = n_patients (indexed by ID), n_col = n_features.
             2) The user inserts the ID string.
    '''
    
    ## Create local copy
    Data_test_local = Data_test.copy()
    
    
    ## Check ID
    ID = input('Inserire ID\n')
    flag = True
    while flag:        
        if ID in Data_test_local.index:
            x_testing = Data_test_local.loc[[ID], :]
            flag = False
        elif ID in ['exit', '\'exit\'', 'Exit']:
            sys.exit('Sessione terminata.')
            flag = False
        else:
            message = 'Inserire nuovo ID o inserire \'exit\' per terminare la sessione\n'
            ID = input(message)
            
    
    ## Check values
    error_message = 'Input non valido: controllare valori citofluorimetrici del paziente con ID='+ ID \
    + '.\nSono ammessi solo valori numerici positivi (gli zeri vengono interpretati come valori mancanti).'
    for element in x_testing.values.ravel():
        if isfloat(element) or pd.isnull(element):
            if isfloat(element):
                if element<0:
                    raise ValueError(error_message)
        else:
            raise ValueError(error_message)
                
    
    ## Convert to float
    x_testing = x_testing.astype(float)
    
    
    ## Sex
    flag = True
    while flag:
        sex = input('Inserire sesso (M, F)\n')
        if sex in ('M', 'F'):
            flag = False
            if sex == 'M':
                value = 0
            else:
                value = 1
    x_testing['sex'] = value

        
    ## Age
    flag = True
    while flag:
        #value = input('Inserire data di nascita gg/mm/aaaa (esempio: 25/12/1960)\n')        
        #match = re.match('^[0-9]{2}/[0-9]{2}/[0-9]{4}$', value)
        #if match or value=='':
        #    if value=='':
        #        value = np.nan
        #        flag = False
        #    else:
        #        datetime_obj = datetime.datetime.strptime(value, '%d/%m/%Y')
        #        day, month, year = datetime_obj.day, datetime_obj.month, datetime_obj.year
        #        if day>0 and day<32:
        #            if month>0 and month<13:
        #                if year>1890 and year<2020:
        #                    value = calculate_age(datetime_obj)
        #                    flag = False
        inserted_value = input('Inserire anno di nascita (esempio: 1940)\n')
        if isfloat(inserted_value):
            year = int(inserted_value)
            if year>1889 and year<2021:
                datetime_str = '01/07/' + inserted_value
                datetime_obj = datetime.datetime.strptime(datetime_str, '%d/%m/%Y')
                value = calculate_age(datetime_obj)
                flag = False
            else:
                print('Valore inserito fuori dal range [1890, 2020]')
        elif inserted_value == '':
            value = np.nan
            flag = False
                
    x_testing['age'] = value
    
    
    ## ID
    x_testing['ID'] = ID
        
    ## Return test set as pandas dataframe
    return x_testing, year, sex
    

# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def calculate_age(birth_date):
    
    '''
       This function calculates current age from datetime object birth_date.
    '''
    
    born = birth_date
    today = date.today()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    
    return age


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def SetPlotParams(magnification=1.0, ratio=float(2.2/2.7), fontsize=11., ylabelsize=None, xlabelsize=None):
    
    plt.style.use('ggplot')

    if (ylabelsize==None):
        ylabelsize = fontsize
    if (xlabelsize==None):
        xlabelsize = fontsize

    ratio = ratio  # usually this is 2.2/2.7
    fig_width = 2.9 * magnification # width in inches
    fig_height = fig_width*ratio  # height in inches
    fig_size = [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True

    plt.rcParams['lines.linewidth'] = 0.7
    #plt.rcParams['lines.markeredgewidth'] = 0.25
    #plt.rcParams['lines.markersize'] = 1
    plt.rcParams['lines.markeredgewidth'] = 1.
    plt.rcParams['errorbar.capsize'] = 1 #1.5

    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.markerscale'] = 1
    plt.rcParams['legend.handlelength'] = 1.5
    plt.rcParams['legend.labelspacing'] = 0.3
    plt.rcParams['legend.columnspacing'] = 0.3
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0.0'
    plt.rcParams['axes.linewidth'] = '0.7'

    plt.rcParams['grid.color'] = '0.85'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = '0.7'
    plt.rcParams['grid.alpha'] = '1.'

    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['xtick.labelsize'] = xlabelsize
    plt.rcParams['ytick.labelsize'] = ylabelsize
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'

    plt.rcParams['xtick.major.size'] = 3.
    plt.rcParams['xtick.major.width'] = 0.7
    plt.rcParams['xtick.minor.size'] = 0
    plt.rcParams['ytick.major.size'] = 3.
    plt.rcParams['ytick.major.width'] = 0.7
    plt.rcParams['ytick.minor.size'] = 0
    plt.rcParams['xtick.major.pad']= 5.
    plt.rcParams['ytick.major.pad']= 5.

    #plt.rcParams['font.sans-serif'] = 'Arial'
    #plt.rcParams['font.serif'] = 'mc'
    #plt.rcParams['text.usetex'] = False # set to True for TeX-like fonts
    #plt.rc('font', family='serif')

    #sn.set_context("paper", rc={"font.size":11, "axes.titlesize":11, "axes.labelsize":11}) # uncomment for seaborn


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #