import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

class BusinessProblem:

  dataset = None
  X = X_train = X_test = None
  y = y_train = y_test = y_pred = None
  classifier = conf_mat = None
  best_params = best_accuracy = None


  def __init__(self):
    # Importing the dataset
    self.dataset = pd.read_csv('data.csv')
    self.X = self.dataset.iloc[:,3:13].values
    self.y = self.dataset.iloc[:,13].values

    # Encoding categorical data
    label_encoder_X_1 = LabelEncoder()
    self.X[:,1] = label_encoder_X_1.fit_transform(self.X[:,1])
    label_encoder_X_2 = LabelEncoder()
    self.X[:,2] = label_encoder_X_2.fit_transform(self.X[:,2])
    one_hot_encoder = OneHotEncoder(categorical_features=[1])
    self.X = one_hot_encoder.fit_transform(self.X).toarray()
    self.X = self.X[:,1:]
    
    # Splitting the dataset into the training set and test set
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
      self.X, self.y,
      test_size=0.2,
      random_state=0)


  def scale_features(self):
    '''
    Feature scaling
    '''
    sc = StandardScaler()
    self.X_train = sc.fit_transform(self.X_train)
    self.X_test = sc.transform(self.X_test)
  

  def build_model(self):
    '''
    Use grid search with k-Fold cross-validation to find the best
    hyperparameters and accuracy
    '''
    classifier = KerasClassifier(build_fn=_build_classifier)
    params = {'batch_size': [25, 32], 
                  'epochs': [100, 500],
                  'optimizer': ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=params,
                               scoring='accuracy',
                               cv=10)
    grid_search = grid_search.fit(self.X_train, self.y_train)
    self.best_params = grid_search.best_params_
    self.best_accuracy = grid_search.best_score_

  
  def get_accuracy(self):
    '''
    Retrieve the best accuracy after grid search
    '''
    print("Accuracy: " + str(self.best_accuracy))
    print("Params: ", self.best_params)


def _build_classifier(optimizer):
    classifier = Sequential()

    # Add the input layer and the first hidden layer with dropout
    classifier.add(Dense(input_dim=11, units=6,
                         kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))

    # Add the second hidden layer with dropout
    classifier.add(Dense(units=6, kernel_initializer='uniform',
                         activation='relu'))
    classifier.add(Dropout(rate=0.1))

    # Add the output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform',
                         activation='sigmoid'))

    # Compile the artificial neural network
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy',
                       metrics=['accuracy'])

    return classifier

def run():
  '''
  Based on a sample of 10.000 bank customers' details from within the past
  6 months, decide whether or not a user is likely to leave the bank for 
  a competitor
  '''
  business_problem = BusinessProblem()
  business_problem.scale_features()
  business_problem.build_model()
  business_problem.get_accuracy()

if __name__ == '__main__':
  run()
