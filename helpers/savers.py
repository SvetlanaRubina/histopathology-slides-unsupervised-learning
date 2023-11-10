import numpy as np


def save_results(file_names, predicted_labels_test, output_name):
    test_results = list(zip(file_names, predicted_labels_test))
    np.savetxt(output_name, test_results, delimiter=',', fmt='%s', header='<file_name>,<class>', comments='')
