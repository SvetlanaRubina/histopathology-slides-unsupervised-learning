import numpy as np


def save_results(file_names_test, predicted_labels_test, output_name_test,
                 file_names_train, predicted_labels_train, output_name_train):
    test_results = list(zip(file_names_test, predicted_labels_test))
    np.savetxt(output_name_test, test_results, delimiter=',', fmt='%s', header='<file_name>,<class>', comments='')
    train_results = list(zip(file_names_train, predicted_labels_train))
    np.savetxt(output_name_train, train_results, delimiter=',', fmt='%s', header='<file_name>,<class>', comments='')