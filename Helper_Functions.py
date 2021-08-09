import pandas as import pd
import numpy as np

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
