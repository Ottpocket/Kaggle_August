import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

def cross_val(train, test, FEATURES, model, TARGET, probabilities = False,
              cross_val_type = StratifiedKFold, cross_val_repeats = 3,
              mean_encode=False, FEATURES_TO_ME = [], n_folds = 5):
    '''
    Trains a model on FEATURES from train to predict TARGET to get out of fold
    predictions for the train and predictions for the test.

    ARGUMENTS
    ___________
    train: (pd.DataFrame) df containing FEATURES and TARGET
    test: (pd.DataFrame) test df containing FEATURES
    model: (sklearn compatable model) model to predict
    TARGET: (str ) the target to be predicted
    probabilities: (bool) whether to get probabilities from the model.
                   This is for CLASSIFICATION only!
                   Wrap your model with Prob_Model class or will get error.
    cross_val_type: (sklearn model selection) how to split the data
    cross_val_repeats: (int) how many times to repeat the cross validation
    mean_encode: (bool) Do you want to mean encode?
    FEATURES_TO_ME: (list of cols) columns to be mean encoded
    n_folds: (int) number of folds for getting the out of fold preds

    OUTPUTS
    __________
    scores: (list) list of scores of oof predictions for each run
    oof: (pd.DataFrame) out of frame predictions.  Columns are named after the
         number of the cross_val.
    preds: (pd.DataFrame) mean of the predictions from each model
    '''
    if probabilities:
        predictions = [f'preds_{i}' for i in train[TARGET].unique()]
        predictions.sort()
    else:
        predictions = ['preds']
    oof = pd.DataFrame(np.zeros(shape = (train.shape[0], len(predictions)) )).rename(columns={i:feat for i, feat in enumerate(predictions)})
    preds = pd.DataFrame(np.zeros(shape = (test.shape[0], len(predictions)) )).rename(columns={i:feat for i, feat in enumerate(predictions)})

    FEATURES_ALL = FEATURES

    #Just to get the column names for later.  All values will be reassigned in the xval part
    if mean_encode:
        encodings = Mean_Encoding(train, FEATURES_TO_ME, TARGET = TARGET, STAT=['mean','var'])
        ME_COL_NAMES = list(encodings.columns)
        for col in ME_COL_NAMES:
            if col not in train.columns:
                train[col] = 0
            if col not in test.columns:
                test[col] = 0
        FEATURES_ALL= FEATURES + ME_COL_NAMES

    for random_state in range(cross_val_repeats):
        skf = cross_val_type(n_splits=n_folds, shuffle=True, random_state=random_state)

        for f, (t_idx, v_idx) in enumerate(skf.split(X=train, y=train[TARGET])):
            start = time()
            train_fold = train[list(set(FEATURES_TO_ME +FEATURES))+[TARGET]].iloc[t_idx].reset_index(drop=True).copy()
            val_fold =   train[list(set(FEATURES_TO_ME +FEATURES))+[TARGET]].iloc[v_idx].reset_index(drop=True).copy()

            #Mean Encoding
            if mean_encode:
                #Getting the mean encoded columns
                train_encoded_cols = Mean_Encoding(train_fold, FEATURES_TO_ME, TARGET = TARGET, STAT=['mean','var'])
                val_encoded_cols = Mean_Encoding_Val(train_fold, val_fold, FEATURES_TO_ME, TARGET = TARGET, STAT=['mean','var'])
                test_encoded_cols= Mean_Encoding_Val(train_fold, test, FEATURES_TO_ME, TARGET = TARGET, STAT=['mean','var'])

                #Adding the encoded columns to the train, val, and test
                train_fold[ME_COL_NAMES] = train_encoded_cols
                val_fold[ME_COL_NAMES] = val_encoded_cols
                test[ME_COL_NAMES] = test_encoded_cols


            #The model
            model.fit(train_fold[FEATURES_ALL], train_fold[TARGET])
            oof.iloc[v_idx, :] += np.reshape(model.predict(val_fold[FEATURES_ALL]) / cross_val_repeats, (len(v_idx), oof.shape[1]))
            preds[predictions] += np.reshape(model.predict(test[FEATURES_ALL]) / (cross_val_repeats * n_folds), (preds.shape))

            #Deleting excess
            if mean_encode:
                del train_fold, val_fold, train_encoded_cols, val_encoded_cols, test_encoded_cols; gc.collect()

            print(f'{time() - start :.2f}', end=', ')
        print()

        scores = [mean_squared_error(train[TARGET].values, oof[col].values, squared=False) for col in oof.columns]
    return scores, oof, preds

class Prob_Model :
    '''
    Creates a sklearn compatable model whose predict only gives probabilities.
    '''
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict_proba(X)


def to_bin(df, num_bins, binned_features, test = None):
    '''
    In place adds binned columns to df and test, if test dataframe is wanted.
    Old column 'feat' is binned as column 'feat_bin'.
    ARGUMENTS
    ___________
    df: (pd.DataFrame) dataframe to have binned columns added
    num_bins: (int) the min number of elements to fit into a bin
    binned_features: (list of columns) the columns to be binned
    test: (pd.DataFrame) the test set to have binned columns
    '''
    for feat in binned_features:
        bins = np.quantile(a=df[feat], q=np.linspace(start=0, stop=1, num=num_bins+1))
        unique = list(set(bins))
        bins_copy = bins.tolist()

        #Ensure no duplicates in binning process
        for uniq_el in unique:
            first_idx = -1
            to_replace = uniq_el
            for idx, element in enumerate(bins_copy):
                if (first_idx == -1) and (bins[idx] == uniq_el):
                    first_idx = idx
                    continue

                elif bins[idx] == uniq_el:
                    bins[idx] = to_replace + 1e-03
                    to_replace += 1e-03

        df[f'{feat}_bin'] = pd.cut(df[feat], bins, labels=False, include_lowest=True)

        if test is not None:
            test[f'{feat}_bin'] = pd.cut(test[feat], bins, labels=False, include_lowest=True)
