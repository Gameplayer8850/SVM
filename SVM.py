from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class SVM:
    def __init__(self, arguments, decisions, size, random):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(arguments, decisions, test_size=size,
                                                                                random_state=random)

    def svm_algorithm(self, kernel: str, c: int):
        classifier = SVC(kernel=kernel, C=c, decision_function_shape='ovr')
        classifier.fit(self.X_train, self.y_train)

        y_prediction = classifier.predict(self.X_test)

        # dokładność klasyfikatora
        accuracy = accuracy_score(self.y_test, y_prediction)
        print("Classification accuracy: ", accuracy)

        # macierz pomyłek
        cm = confusion_matrix(self.y_test, y_prediction)
        print(f"Confusion matrix: \r\n{cm}")
        # print(cm)
