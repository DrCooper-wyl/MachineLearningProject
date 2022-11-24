import os
import sys
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, ShuffleSplit
from src.data import Data

data = Data()
x, y = data.get_data(data.CROP_DIR)
def train_test(x, y):
    print("Start training")
    rfc = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=90, verbose=1)
    #shuffle = ShuffleSplit(train_size=.7, test_size=.2, n_splits=8)
    scores = cross_val_score(rfc, x, y, cv=3)
    print("Cross validation scores:{}".format(scores))
    print("Mean cross validation score:{:2f}".format(scores.mean()))
    print("Finish training")

def test():

    x_test_crop,y_test_crop=data.get_data(data.TEST_CROP_DIR)
    print("Start training")
    rfc = RandomForestClassifier(verbose=1,n_jobs=-1)
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                    'class 6', 'class 7', 'class 8', 'class 9', 'class 10',
                    'class 11', 'class 12', 'class 13', 'class 14', 'class 15']
    rfc.fit(x,y)
    y_pred_crop=rfc.predict(x_test_crop)
    print("With crop")
    print(classification_report(y_test_crop, y_pred_crop, target_names=target_names))

if __name__ == "__main__":
    start = time.time()
    test()
    train_test(x, y)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
