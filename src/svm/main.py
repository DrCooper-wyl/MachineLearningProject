from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC
from src.data import Data
import time

data=Data()
x,y=data.get_data(data.CROP_DIR)
def train_test(x, y):
    clf = SVC(C=40.0, verbose=1,decision_function_shape='ovr')
    #shuffle = ShuffleSplit(train_size=.7, test_size=.2, n_splits=5)
    scores = cross_val_score(clf, x, y, cv=3)
    print("Cross validation scores:{}".format(scores))
    print("Mean cross validation score:{:2f}".format(scores.mean()))
    print("Finish training")

def test():
    x_test_crop,y_test_crop=data.get_data(data.TEST_CROP_DIR)
    print("Start training")
    rfc = SVC(verbose=1)
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                    'class 6', 'class 7', 'class 8', 'class 9', 'class 10',
                    'class 11', 'class 12', 'class 13', 'class 14', 'class 15']
    rfc.fit(x,y)
    y_pred_crop=rfc.predict(x_test_crop)
    print(classification_report(y_test_crop, y_pred_crop, target_names=target_names))

# 6 9
if __name__ == "__main__":
    start = time.time()
    test()
    train_test(x,y)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
