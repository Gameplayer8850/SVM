# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pandas as pd

from Data.DataOperations import DataOperations
from SVM import SVM


def __main():
    directory = os.getcwd()

    if not os.path.exists(f'{directory}/Data/HearthDiseaseDataSet.csv'):
        print('File with data was (HearthDiseaseDataSet) not found.')
        return

    if not os.path.exists(f'{directory}/Data/DummyName.csv'):
        print('File with data was (DummyName) not found.')
        #return

    index_dataset = GetInputNumber("1. HearthDiseaseDataSet\r\n2.DummyName\r\nEnter dataset index: ", 1, 2, True)

    # pobranie danych z pliku i zamienienie ich w DataFrame
    data=DataOperations.read_file(f'{directory}/Data/HearthDiseaseDataSet.csv' if index_dataset == 1 else f'{directory}/Data/DummyName.csv')

    # normalizacja danych
    data=DataOperations.normalize_data(data)

    train_size = GetInputNumber("Enter % size of train dataset: ", 0, 99, True)/100.0
    random_state = GetInputNumber("Enter random state: ", 0, 99, True)

    # zdefiniowanie klasy od svm oraz wykonanie podziału w konstruktorze na set terningowy i testowy
    svm = SVM(data.iloc[:, :-1], data.iloc[:, -1], train_size, random_state)

    while True:
        c = GetInputNumber("Enter value of C: ", 0, 10, True)
        svm.svm_algorithm('linear', c)

def GetInputNumber(title: str, min: int, max: int, clearConsole:bool):
    index = min-1
    while index < min or index>max:
        try:
            index = int(input(title))
            if clearConsole:
                clear = lambda: os.system('cls')
                clear()
            break
        except ValueError:
            index = min-1
    return index


__main()





