from sklearn.metrics import classification_report
from SoundClassifier import *
from SoundDataManager import get_dataset_from_wavfile, get_train_test, pre_process
import requests
from datetime import datetime
from joblib import dump, load


def get_trained_classifier(X_train, y_train, X_test, y_test, algorithm):
    X_test, X_train = pre_process(X_test, X_train)
    clf = SoundClassifier(algorithm)
    clf.train_classifier(X_train, y_train)
    predictions = clf.get_predictions(X_test)
    print(f"Accuracy Train =  {str(clf.get_accuracy(X_train, y_train))}")
    print(f"Accuracy Test  =  {str(clf.get_accuracy(X_test, y_test))}")
    print(classification_report(y_test, predictions))
    a = datetime.now()
    dump(clf, str(datetime.now() + algorithm))
    return clf


def split_uav_criteria(data, target, index):
    X_loaded_unloaded = []
    y_loaded_unloaded = []
    X_uav_noise = []
    y_uav_noise = []
    for index in index:
        if target[index] == 'Unloaded':
            X_loaded_unloaded.append(data[index])
            y_loaded_unloaded.append('Unloaded')
            X_uav_noise.append(data[index])
            y_uav_noise.append('UAV')
        elif target[index] == 'Loaded':
            X_loaded_unloaded.append(data[index])
            y_loaded_unloaded.append('Loaded')
            X_uav_noise.append(data[index])
            y_uav_noise.append('UAV')
        elif target[index] == 'Noise':
            X_uav_noise.append(data[index])
            y_uav_noise.append('Noise')
    return np.vstack(X_loaded_unloaded), y_loaded_unloaded, np.vstack(X_uav_noise), y_uav_noise


def main():
    # feature_type: 'mfcc' or 'filter_banks'
    data, target, filenames = get_dataset_from_wavfile('wavfiles/realuav/', 'labels_new.csv', 1.5, 'mfcc', 'class2')
    X_test, X_train, y_test, y_train, train_index, test_index = get_train_test(data, target)
    print("Final Report All Together")
    get_trained_classifier(X_train, y_train, X_test, y_test, "gnb")


if __name__ == '__main__':
    load_json()
