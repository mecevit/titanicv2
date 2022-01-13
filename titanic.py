from layer.v2.assertions import *
from layer.v2.decorators import dataset, model
from layer.v2.dependencies import File, Directory
from layer import Dataset, Model
from layer.client import Dataset
from layer.v2 import LayerProject
import layer
import sys

layer.login("https://dev-judgment-day.layer.co/")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import pandas as pd

def clean_sex(sex):
    result = 0
    if sex == "female":
        result = 0
    elif sex == "male":
        result = 1
    return result


def clean_age(data):
    age = data[0]
    pclass = data[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age

data_file = './titanic.csv'

# @greatexpectations("expect_column_stdev_to_be_between", "Fare" ,10, 200)
# @assert_valid_values('Sex', ['male', 'female'])
@assert_unique("PassengerId")
@dataset('passengers_abc', dependencies=[File(data_file)])
def read_and_clean_dataset():
    df = pd.read_csv(data_file)
    return df

# @assert_valid_values('Sex', [0,1])
@dataset('features_abc', dependencies=[Dataset("passengers_abc")])
def extract_features():
    # df = layer.get_dataset("passengers_abc").to_pandas()
    df = read_and_clean_dataset()
    df['Sex'] = df['Sex'].apply(clean_sex)
    df['Age'] = df[['Age', 'Pclass']].apply(clean_age, axis=1)
    df = df.drop(["PassengerId", "Name", "Cabin", "Ticket", "Embarked"], axis=1)
    return df

@model(name='survival_model_abc', dependencies=[Dataset("features_abc")])
def train():
    # df = layer.get_dataset("passenger_features").to_pandas()
    df = extract_features()
    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    layer.log_metric("accuracy", acc)
    return random_forest


# ++ init Layer
layer_project = LayerProject(name="titanic-mecevit-local", requirements=File("requirements.txt"), debug=False)

# ++ To run the whole project on Layer Infra
# layer_project.run([read_and_clean_dataset, extract_features, train])
# layer_project.run([train, read_and_clean_dataset, extract_features])

# ++ To build individual assets on Layer infra
# layer_project.run([read_and_clean_dataset])
# layer_project.run([extract_features])
# layer_project.run([train])

#
# ++ To debug the code locally, just call the function:
train()
# df = extract_features()
# df = read_and_clean_dataset()
# df.head()