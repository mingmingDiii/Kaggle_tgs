import pandas as pd
from sklearn.model_selection import KFold
import os
all_list = open('/mnt/sda1/don/documents/tgs/data/split_list/train_all.txt').read().splitlines()
print(len(all_list))
folds_num = 10
kf = KFold(n_splits=10)
f=0
for train_l,test_l in kf.split(all_list):

    fold_path = '/mnt/sda1/don/documents/tgs/data/fold_split2/f{}/'.format(f)
    if not os.path.exists(fold_path):
        os.mkdir(fold_path)

    train_list = []
    test_list = []
    for i in range(len(train_l)):
        print(all_list[train_l[i]])
        train_list+=[all_list[train_l[i]]]

    for i in range(len(test_l)):
        test_list+=[all_list[test_l[i]]]

    train_list = pd.Series(train_list)
    train_list.to_csv(fold_path+'train.csv'.format(f),index=False)

    test_list = pd.Series(test_list)
    test_list.to_csv(fold_path+'val.csv'.format(f),index=False)

    f+=1

