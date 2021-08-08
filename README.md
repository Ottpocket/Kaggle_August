# Kaggle_August
Ideas for competing in the August Tabular comp.

## Feature Engineering ideas to try
1. Turn columns to ordinal (1,2,...,n) instead of real valued.
2. Likelihood encode the columns 
3. Use rf to find most common column interactions and combine
4. DAE features.  Consider taking the mean of n models features.
5. Predict 0s
6. Predict levels (0,1,2,3,4) where each level corresponds to a range of TARGET values
7. PseudoLabelling

## Fancy Models 
1. TabNet
2. Muddling Regularization Networks
3. NN on DAE features
4. 1D CNN
5. 2D CNN


## Kernels to look at 
1. DAE https://www.kaggle.com/cdeotte/dae-book1c/comments
2. pseudolabels https://www.kaggle.com/alexryzhkov/tps-lightautoml-baseline-with-pseudolabels

## TODO
1. Deotte style MultilabelStratifiedKFold baseline.  Used in Kernel 1.
2. Enable WandB for team.
3. 
