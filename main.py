# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

from Data.DataOperations import DataOperations
from SVM import SVM


def __main():
    directory = os.getcwd()

    if not os.path.exists(f'{directory}/Data/HearthDiseaseDataSet.csv'):
        print('Hearth Disease dataset file was not found (HearthDiseaseDataSet.csv).')
        return

    if not os.path.exists(f'{directory}/Data/Diabetics.csv'):
        print('Diabetics dataset file was not found (Diabetics.csv).')
        # return

    if not os.path.exists(f'{directory}/Data/CarClassifierDataset.csv'):
        print('Car classifier dataset file was not found (CarClassifierDataset.csv).')
        # return

    index_dataset = GetInputNumber(
        "1. Hearth disease dataset\r\n2. Diabetics dataset\r\n3. Car classifier dataset\r\nEnter dataset index: ", 1, 3,
        True)

    # pobranie danych z pliku i zamienienie ich w DataFrame
    if index_dataset == 1:
        path_to_data = f'{directory}/Data/HearthDiseaseDataSet.csv'
    elif index_dataset == 2:
        path_to_data = f'{directory}/Data/Diabetics.csv'
    else:
        path_to_data = f'{directory}/Data/CarClassifierDataset.csv'

    data = DataOperations.read_file(path_to_data)

    # normalizacja danych

    data = DataOperations.normalize_data(data)

    train_size = GetInputNumber("Enter % size of train dataset: ", 0, 99, True) / 100.0
    random_state = GetInputNumber("Enter random state: ", 0, 99, True)

    # zdefiniowanie klasy od svm oraz wykonanie podziału w konstruktorze na set terningowy i testowy
    svm = SVM(data.iloc[:, :-1], data.iloc[:, -1], train_size, random_state)

    while True:
        print()
        kernel_type = GetInputNumber("1. Linear Kernel type\r\n2. RBF Kernel type\r\nEnter kernel type: ", 1, 2,
                                     True)
        c = GetInputNumber("Enter value of C: ", 0, 10, True)
        svm.svm_algorithm('linear' if kernel_type == 1 else 'rbf', c)


def GetInputNumber(title: str, min: int, max: int, clearConsole: bool):
    index = min - 1
    while index < min or index > max:
        try:
            index = int(input(title))
            if clearConsole:
                clear = lambda: os.system('cls')
                clear()
            break
        except ValueError:
            index = min - 1
    return index


__main()
