import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LearningML:
    def __init__(self) -> None:
        self.path = "housing.csv"
        self.test_data_ratio = 0.2
        self.data = pd.read_csv(self.path)
        self.trainingData = None
        self.testData = None
        np.random.seed(42)

    def look(self, dataToExplore: pd.DataFrame):
        print(dataToExplore.head())
        print (dataToExplore.info())
        print(dataToExplore['ocean_proximity'].value_counts())
        dataToExplore.hist(bins=50, figsize=(20,15))
        plt.show()

    def split_into_test_and_training_set(self):
        data_length = self.data.shape[0]
        test_length = int(self.test_data_ratio*data_length)
        indices = np.random.randint(0,data_length,data_length)
        self.testData = self.data.iloc[indices[:test_length]]
        self.trainingData = self.data.iloc[indices[test_length:]]
        print ("Split data into Test set : ", self.testData.shape[0]," rows, and Training Set: ",self.trainingData.shape[0]," rows")

    def splitUsingScikit(self):
        self.trainingData, self.testData = train_test_split(self.data,test_size=self.test_data_ratio, random_state=42)
        print("SKLEARN: Split data into Test set : ", self.testData.shape[0], " rows, and Training Set: ",
              self.trainingData.shape[0], " rows")


def main():
    learn = LearningML()
    learn.look(learn.data)
    learn.splitUsingScikit()


main()