import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataOperations:
    @staticmethod
    def read_file(path: str):
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        column_names = lines[0].split(';')
        data = [line.split(';') for line in lines[1:]]
        return pd.DataFrame(data, columns=column_names)

    @staticmethod
    def normalize_data(data: pd.DataFrame):
        label_encoder = LabelEncoder()
        scaler = StandardScaler()

        columns = data.columns
        for column in columns:
            if "[NO]" in column or "[OR]" in column:
                data[column] = label_encoder.fit_transform(data[column])
            elif "[NU]" in column:
                data[[column]] = scaler.fit_transform(data[[column]])
            elif "[BIS]" in column or "[BIA]" in column:
                data[column] = data[column].replace('true', 1).replace('false', 0)
        return data
