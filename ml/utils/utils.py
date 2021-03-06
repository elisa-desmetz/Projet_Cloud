import pandas as pd
import numpy as np

class DataHandling:
    """
        Getting data from local csv file
    """
    def __init__(self):
        """
            Initializing the dataset handling
        """
        print("DataHandling intialization")
        self.data = None
        print("Intialization done !")

    def get_data(self):
        print("Loading data from local file...")
        self.data = pd.read_csv("./static/data/data.csv")
        self.data=self.data.rename(columns={'Engine HP':'HP','Engine Cylinders':'Cylinders','Transmission Type':'Transmission','Driven_Wheels':'Drive Mode','highway MPG':'MPG-H','city mpg':'MPG-C','MSRP':'Price'})
        print("Dataset shape : {} lines, {} columns".format(self.data.shape[0],self.data.shape[1]))
        print("Data loaded from local file !")

class FeatureRecipe:
    """
        Cleaning the dataset
    """
    def __init__(self,data:pd.DataFrame):
        """
            Initialazing the FeatureRecipe
        """
        print("FeatureRecipe intialization...")
        self.data = data
        self.categorial = []
        self.continuous = []
        self.discrete = []
        self.drop = []
        print("Initialization done !")

    def convert_and_encode(self):
        def getrange(Price):
            if (Price >= 0 and Price < 25000):
                return '0 - 25000'
            if (Price >= 25000 and Price < 50000):
                return '25000 - 50000'
            if (Price >= 50000 and Price < 75000):
                return '50000 - 75000'
            if (Price >= 75000 and Price < 100000):
                return '75000 - 100000'
        self.data['Price Range'] = self.data.apply(lambda x:getrange(x['Price']),axis = 1)

        label_encoder = preprocessing.LabelEncoder()
        for col in ['Make','Model','Engine Fuel Type','Transmission','Drive Mode','Vehicle Size','Vehicle Style','Price']:
            self.data[col] = self.data[col].astype('category') 
            self.data[col] = label_encoder.fit_transform(self.data[col])

    def separate_variable_types(self) -> None:
        """
            Separating dataset features by type
        """
        print("Separating features by type...")
        for col in self.data.columns:
            if self.data[col].dtypes == 'int64':
                self.discrete.append(self.data[col])
            elif self.data[col].dtypes == 'float64':
                self.continuous.append(self.data[col])
            else:
                self.categorial.append(self.data[col])
        print ("Dataset columns : {} \nNumber of discrete features : {} \nNumber of continuous features : {} \nNumber of categorical features : {} \nTotal number of features : {}\n".format(len(self.data.columns),len(self.discrete),len(self.continuous),len(self.categorial),len(self.discrete)+len(self.continuous)+len(self.categorial) ))

    def drop_useless_features(self) :
        """
            Droping useless features and observations
        """
        print("Dropping useless features and observations...")
        # Dropping observations with price == 0
        self.data.drop(self.data[self.data['Price'] == 0].index, inplace=True)
        # Dropping Market Cetegory feature
        self.data.drop(['Market Category'], axis=1, inplace=True)
        # Dropping duplicates observations and null features observations
        self.data.drop_duplicates(keep=False,inplace=True)
        self.data.dropna(inplace=True,axis=0)
        # Dropping extreme observations
        self.data.drop(self.data[self.data['Price'] >= 500000].index,inplace=True)
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        self.data = self.data[~((self.data < (Q1 - 1.5*IQR))|(self.data > (Q3 + 1.5*IQR))).any(axis = 1)]
        print("Useless features and observations dropped !\n")

    def drop_duplicate(self) :
        """
            Dropping duplicates features
        """
        def get_duplicate(data:pd.DataFrame): 
              # Create an empty set 
            duplicateColumnNames = set() 
      
            # Iterate through all the columns  
            # of dataframe 
            for x in range(data.shape[1]): 
          
                # Take column at xth index. 
                col = data.iloc[:, x] 
          
                # Iterate through all the columns in 
                # DataFrame from (x + 1)th index to 
                # last index 
                for y in range(x + 1, data.shape[1]): 
              
                    # Take column at yth index. 
                    otherCol = data.iloc[:, y] 
              
                    # Check if two columns at x & y 
                    # index are equal or not, 
                    # if equal then adding  
                    # to the set 
                    if col.equals(otherCol): 
                            duplicateColumnNames.add(data.columns.values[y]) 
                  
            # Return list of unique column names  
            # whose contents are duplicates. 
            return list(duplicateColumnNames) 
        self.data = self.data.drop(get_duplicate(self.data), axis=1)
   
    def drop_na_prct(self,threshold : float):
        """
            Threshold between 0 and 1.
            Dropping features with NaN ratio > threshold
            Counting dropped features
            params: threshold : float
        """
        count = 0
        print("Dropping features with more than {} % NaN...".format(threshold*100))
        for col in self.data.columns:
            if self.data[col].isna().sum()/self.data.shape[0] >= threshold:
                self.drop.append( self.data.drop([col], axis='columns', inplace=True) )
                count+=1
        print("Dropped {} features.\n".format(count))

    def prepare_data(self,threshold : float):
        self.drop_useless_features()
        self.drop_na_prct(threshold)
        self.drop_duplicate()
        self.separate_variable_types()
        self.convert_and_encode()
        print("Processed dataset shape : {} lines, {} columns".format(self.data.shape[0],self.data.shape[1]))
        print("FeatureRecipe processing done !\n")


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

class FeatureExtractor:
    """
    Feature Extractor class
    """
    def __init__(self, data: pd.DataFrame, keep_list: list):
        """
            Input : pandas.DataFrame, features list to keep
            Output : X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        """
        print("FeatureExtractor initialization...")
        self.X_train, self.X_test, self.y_train, self.y_test = None,None,None,None
        self.data = data
        self.klist = keep_list
        print("Intialization done !\n")

    def extractor(self):
        print("Extracting selected columns... ")
        self.data=self.data[self.klist]
        print("Selected columns extracted ! \n")

    def splitting(self, size:float,rng:int, y:str):
        """
            size : part of the testing set, between 0 and 1
            rng : randomizer for the splitting
            y : result feature
        """
        print("Splitting dataset in training and testing sets...")
        x = self.data.loc[:,self.data.columns != y]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, self.data[y], test_size=size, random_state=rng)
        print("Splitting done !\n")

    def split_data(self,split:float):
        self.extractor()
        self.splitting(split,42,'Price')
        print("FeatureExtractor processing done !\n")
        return self.X_train, self.X_test, self.y_train, self.y_test

from joblib import dump,load

class ModelBuilder:
    """
        Training and printing machine learning model
    """
    def __init__(self, model_path:str=None, save:bool=None):

        print("ModelBuilder initialization...")
        self.model_path = model_path
        self.save = save
        self.reg = RandomForestRegressor()
        print ("Initialization done !")

    def train(self, X, y):
        print("Training the algorithm...")
        self.reg.fit(X,y)
        print("Algorithm trained !")


    def __repr__(self):
        pass

    def predict_test(self, X) -> np.ndarray: 
        return self.reg.predict(X.iloc[:1])

    def save_model(self, path:str):
        print('Saving file...')
        dump(self.reg, '{}/model.joblib'.format(path))
        self.save = True
        print('File saved !')

    def predict_from_dump(self, X) -> np.ndarray:
        return self.reg.predict(X)

    def print_accuracy(self,X_test,y_test):
        score=self.reg.score(X_test,y_test)*100
        print('Regression got an accuracy of {}%'.format(score))

    def load_model(self):
        print('Loading file...')
        try:
            self.reg = joblib.load("{}/model.joblib".format(self.path))
            print("File loaded !")
        except:
            print("File not found !")