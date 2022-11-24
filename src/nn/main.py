import time

from sklearn import neural_network
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import ShuffleSplit, cross_val_score

from src.data import Data


def train_test(x, y):
    nn = neural_network.MLPClassifier(verbose=1)
    # shuffle = ShuffleSplit(train_size=.7, test_size=.2, n_splits=5)
    scores = cross_val_score(nn, x, y, cv=3)
    print("Cross validation scores:{}".format(scores))
    print("Mean cross validation score:{:2f}".format(scores.mean()))
    print("Finish training")


# def test():
#
#     x_test_crop,y_test_crop=data.get_data(data.TEST_CROP_DIR)
#     print("Start training")
#     rfc = neural_network.MLPClassifier(verbose=1)
#     target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
#                     'class 6', 'class 7', 'class 8', 'class 9', 'class 10',
#                     'class 11', 'class 12', 'class 13', 'class 14', 'class 15']
#     rfc.fit(x,y)
#     y_pred_crop=rfc.predict(x_test_crop)
#     print(classification_report(y_test_crop, y_pred_crop, target_names=target_names))

def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))  # print classification report
    return accuracy_score(y_true, y_pred)  # return accuracy score


if __name__ == "__main__":
    start = time.time()
    # test()
    data = Data()
    x, y = data.get_data(data.CROP_DIR)
    rfc = neural_network.MLPClassifier(verbose=1)
    nested_score = cross_val_score(rfc, X=x, y=y, cv=3, scoring=make_scorer(classification_report_with_accuracy_score))
    print(nested_score)
    # train_test(x, y)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
