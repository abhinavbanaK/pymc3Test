import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class LearningML:
    def __init__(self) -> None:
        self.path = "housing.csv"
        self.test_data_ratio = 0.2
        self.data = pd.read_csv(self.path)
        self.trainingData = None
        self.testData = None
        self.originalTrainingData = None
        np.random.seed(42)

    def look(self, dataToExplore: pd.DataFrame):
        print(dataToExplore.head())
        print (dataToExplore.info())
        dataToExplore.hist(bins=50)

        plt.legend()
        plt.show()

    def split_into_test_and_training_set(self):
        data_length = self.data.shape[0]
        test_length = int(self.test_data_ratio*data_length)
        indices = np.random.randint(0,data_length,data_length)
        self.testData = self.data.iloc[indices[:test_length]]
        self.trainingData = self.data.iloc[indices[test_length:]]
        print ("Split data into Test set : ", self.testData.shape[0]," rows, and Training Set: ",self.trainingData.shape[0]," rows")

    def splitUsingScikit(self):
        """
        Split a dataset into a test data and training data, in a proportion given by test_data_ratio parameter.
        :return:
        """
        self.trainingData, self.testData = train_test_split(self.data,test_size=self.test_data_ratio, random_state=42)
        print("SKLEARN: Split data into Test set : ", self.testData.shape[0], " rows, and Training Set: ",
              self.trainingData.shape[0], " rows")

    def stratifiedSamplingUsingIncome(self):
        """
        Creates a test and training set, stratified on the "income" category of the original dataset
        This ensures that the test and training sets have the same proportion of various income categories as the original
        :return:
        """
        self.data['income'] = pd.cut(self.data['median_income'],bins=[0,1.5,3,4.5,6,np.inf], labels=[1,2,3,4,5])

        split = StratifiedShuffleSplit(n_splits=1,test_size=self.test_data_ratio,random_state=42)
        for train,test in split.split(self.data, self.data['income']):
            self.trainingData = self.data.loc[train]
            self.testData = self.data.loc[test]

        self.trainingData.drop('income', axis=1,inplace=True)
        self.testData.drop('income', axis=1, inplace=True)
        self.originalTrainingData = self.trainingData.copy()

    def createCombinedAttributes(self):
        self.trainingData['rooms_per_household'] = self.trainingData['total_rooms']/self.trainingData['households']
        self.trainingData['bedrooms_per_room'] = self.trainingData['total_bedrooms']/self.trainingData['total_rooms']
        self.trainingData['population_per_household'] = self.trainingData['population']/self.trainingData['households']

    def dataCleaning(self):
        self.trainingData = self.originalTrainingData.drop('median_house_value', axis=1)
        self.house_labels=self.originalTrainingData['median_house_value'].copy()
        # take care of missing values in data, by either
        #1. drop the rows with missing data
        #2. drop the whole column which has missing data
        #3. set the missing values to 0, mean, median etc.. BUT this needs to be done for BOTH test/Training sets AND to any real-world data that the ML Algorithm
        # needs to work with
        # we are using the strategy to replace missing values with median

        imputer = SimpleImputer(strategy="median")

        # the imputer only works with numeric data.. so we create a copy of data withou the "ocean proximity" text column
        housing_num = self.trainingData.drop('ocean_proximity', axis=1)

        # calculate the median values for each attribute/column
        imputer.fit(housing_num)

        # use the imputer to replace the missing values with learned/calculated medians for that attribute/column
        X = imputer.transform(housing_num)
        # transformed data set
        housing_tr = pd.DataFrame(X,columns=housing_num.columns, index=housing_num.index)
        print(housing_tr)

        # we can also convert a text attribute to numeric using a transformation
        oneHot = OneHotEncoder()
        housing_cat= self.trainingData[['ocean_proximity']]
        housing_cat_encoded=oneHot.fit_transform(housing_cat)
        print(housing_cat_encoded[:10])

def main():
    learn = LearningML()

    learn.stratifiedSamplingUsingIncome()
    learn.trainingData.plot(kind="scatter", x='longitude', y='latitude', alpha=0.1,
                           s=learn.trainingData['population'] / 100,
                           label='population', figsize=(10, 7), c='median_house_value',
                           cmap=plt.get_cmap('jet'), colorbar=True)
    plt.legend()
    learn.createCombinedAttributes()

    corr_matrix= learn.trainingData.corr()

    print(corr_matrix['median_house_value'].sort_values(ascending=False))
    scatter_matrix(learn.trainingData[['median_house_value','median_income','total_rooms', 'housing_median_age']], figsize=(12,8))
    plt.show()

    learn.dataCleaning()


main()