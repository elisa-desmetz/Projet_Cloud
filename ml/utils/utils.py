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

    def separate_variable_types(self) -> None:
        """
            Separating dataset features by type
        """
        print("Separating features by type...")
        for col in self.data.columns:
            if self.data[col].dtypes == int:
                self.discrete.append(self.data[col])
            elif self.data[col].dtypes == float:
                self.continuous.append(self.data[col])
            else:
                self.categorial.append(self.data[col])
        print ("Dataset columns : {} \nNumber of discrete features : {} \nNumber of continuous features : {} \nNumber of categorical features : {} \nTotal number of features : {}".format(len(self.data.columns),len(self.discrete),len(self.continuous),len(self.categorial),len(self.discrete)+len(self.continuous)+len(self.categorial) ))

    def drop_useless_features(self) :
        """
            Droping useless features and observations
        """
        print("Dropping useless features and observations...")
        # Dropping observations with price == 0
        self.data.drop(self.data[self.data['MSRP'] == 0].index, inplace=True)
        # Dropping Market Cetegory feature
        self.data.drop(['Market Category'], axis=1, inplace=True)
        print("Useless features and observations dropped !")

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
        print("Dropped {} features.".format(count))

    def prepare_data(self,threshold : float):
        self.drop_useless_features()
        self.drop_na_prct(threshold)
        self.drop_duplicate()
        self.separate_variable_types()
        print(self.data.shape)
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
    def __init__(self, data: pd.DataFrame, flist: list):
        """
            Input : pandas.DataFrame, feature list to drop
            Output : X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        """
        print("FeatureExtractor intialisation")
        self.X_train, self.X_test, self.y_train, self.y_test = None,None,None,None
        self.data = data
        self.flist = flist
        print("intialisation done")

    def extractor(self):
        print("extracting unwanted columns")
        for col in self.flist:
            if col in self.data:
                self.df.drop(col, axis=1, inplace=True)
        print("done extracting unwanted columns")

    def splitting(self, size:float,rng:int, y:str):
        print("splitting dataset for train and test")
        x = self.data.loc[:,self.data.columns != y]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, self.data[y], test_size=size, random_state=rng)
        print("splitting done")

    def extract_data(self):
        self.extractor()
        self.splitting(0.3,42,'FEATURE')
        print("done processing Feature Extractor")
        return self.X_train, self.X_test, self.y_train, self.y_test


class ModelBuilder:
    """
        Training and printing machine learning model
    """
    def __init__(self, model_path: str = None, save: bool = None):

        print("ModelBuilder initialization...")
        self.model_path = model_path
        self.save = save
        self.line_reg = LinearRegression()
        print ("Initialization done !")

    def train(self, X, y):
        self.line_reg.fit(X,y)

    def __repr__(self):
        pass

    def predict_test(self, X) -> np.ndarray:
        # on test sur une ligne
        return self.line_reg.predict(X)

    def save_model(self, path:str):

        #with the format : 'model_{}_{}'.format(date)
        #joblib.dump(self, path, 3)
        pass

    def predict_from_dump(self, X) -> np.ndarray:
        pass

    def print_accuracy(self,X_test,y_test):
        self.line_reg.predict(X_test)
        self.line_reg.score(X_test,y_test)*100

    def load_model(self):
        try:
            #load model
            pass
        except:
            pass