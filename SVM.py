from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd


class SVM:
    def __init__(self, arguments, decisions, size, random):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(arguments, decisions, test_size=size,
                                                                                random_state=random)

    def cm_print(self, cm, labels):
        df = pd.DataFrame()
        for i, row_label in enumerate(labels):
            rowdata = {}
            for j, col_label in enumerate(labels):
                rowdata[col_label] = cm[i, j]
            df = df._append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
        return df[labels]

    def svm_algorithm(self, kernel: str, c: int):
        classifier = SVC(kernel=kernel, C=c, decision_function_shape='ovr')
        classifier.fit(self.X_train, self.y_train)

        y_prediction = classifier.predict(self.X_test)
        # dokładność klasyfikatora
        accuracy = accuracy_score(self.y_test, y_prediction)
        print("Classification accuracy: ", "{:.2f}".format(accuracy))

        # macierz pomyłek
        cm = confusion_matrix(self.y_test, y_prediction, labels=classifier.classes_)

        print(self.cm_print(cm, classifier.classes_))
