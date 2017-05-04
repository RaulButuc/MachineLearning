import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

class BusinessProblem:

  dataset = None
  X = X_train = X_test = None
  y = y_train = y_test = y_pred = None
  classifier = conf_mat = None


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
    Build the artificial neural network
    '''
    self.classifier = Sequential()
    self.classifier.add(Dense(input_dim=11, units=6,
                              kernel_initializer='uniform', activation='relu'))
    self.classifier.add(Dense(units=6, kernel_initializer='uniform',
                              activation='relu'))
    self.classifier.add(Dense(units=1, kernel_initializer='uniform',
                              activation='sigmoid'))
    self.classifier.compile(optimizer='adam', loss='binary_crossentropy',
                            metrics=['accuracy'])
    self.classifier.fit(self.X_train, self.y_train,
                        batch_size=10, epochs=100)


  def predict(self):
    '''
    Predicting the test set result
    '''
    self.y_pred = self.classifier.predict(self.X_test)
    self.y_pred = (self.y_pred > 0.5)
    

  def make_confusion_matrix(self):
    '''
    Making the confusion matrix
    '''
    self.conf_mat = confusion_matrix(self.y_test, self.y_pred)

  
  def get_accuracy(self):
    '''
    Divide the number of correct predictions by the total number of entries
    '''
    print("Test accuracy: " + 
          str(np.trace(self.conf_mat) / self.y_pred.shape[0]))


def run():
  '''
  Based on a sample of 10.000 bank customers' details from within the past
  6 months, decide whether or not a user is likely to leave the bank for 
  a competitor
  '''
  business_problem = BusinessProblem()
  business_problem.scale_features()
  business_problem.build_model()
  business_problem.predict()
  business_problem.make_confusion_matrix()
  business_problem.get_accuracy()

if __name__ == '__main__':
  run()
