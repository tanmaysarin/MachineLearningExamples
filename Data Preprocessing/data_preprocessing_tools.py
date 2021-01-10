# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt # used for plotting graphs
import pandas as pd # used for importing data


# Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values #select all rows, the first three columns
Y = dataset.iloc[:, -1].values #select all rows, and just the last column as
                               #it is the dependent variable
                               
print(X)
print(Y)


# Taking care of missing data
from sklearn.impute import SimpleImputer # import this class to fill 
                                         # missing data by taking averages

# create an object of the class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# add the imputer to our matrix of values "x"
imputer.fit(X[:, 1:3]) # connect
# call transform to apply the changes to the dataset
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)
 

# Encoding categorical data

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)


# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

print(Y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("Feature Scaling - ")
print(X_train)
print(X_test)