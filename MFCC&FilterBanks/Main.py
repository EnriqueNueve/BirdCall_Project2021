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
    # X_test_processed, X_train_processed = pre_process(X_test, X_train)
    # X_train_loaded_unloaded, y_train_loaded_unloaded, X_train_uav_noise, y_train_uav_noise = split_uav_criteria(data, target, train_index)
    # X_test_loaded_unloaded, y_test_loaded_unloaded, X_test_uav_noise, y_test_uav_noise = split_uav_criteria(data, target, test_index)
    # print("Report UAV vs Noise")
    # clf_uav_noise = get_trained_classifier(X_train_uav_noise, y_train_uav_noise, X_test_uav_noise, y_test_uav_noise,
    #                                        "gnb")
    # predictions = np.array(clf_uav_noise.get_predictions(X_test_processed), dtype="object")
    # print("Report Loaded vs Unloaded")
    # clf_loaded_unloaded = get_trained_classifier(X_train_loaded_unloaded, y_train_loaded_unloaded,
    #                                              X_test_loaded_unloaded, y_test_loaded_unloaded, "gnb")
    #
    # for index in range(predictions.size):
    #     if predictions[index] == 'UAV':
    #         pred_loaded_unloaded = clf_loaded_unloaded.get_predictions(X_test_processed[index].reshape(1, -1))
    #         predictions[index] = pred_loaded_unloaded[0]
    #
    # print("Final Report in Sequence")
    # print(classification_report(y_test, predictions))

    print("Final Report All Together")
    get_trained_classifier(X_train, y_train, X_test, y_test, "gnb")


def load_json():
    addr = "https://www.xeno-canto.org/api/2/recordings?query=troglodytes+troglodytes"
    URLs = []
    ids = []
    files = []
    for i in range(1, 1000):
        URLs.append(addr + "&page=" + str(i))

    for url in URLs:
        results = requests.get(url)
        print(results.status_code)
        if results.status_code != 200:
            break
        data = results.json()
        for r in data["recordings"]:
            files.append(r["file-name"])
            ids.append(r["id"])
    print(len(files), len(ids))


if __name__ == '__main__':
    load_json()