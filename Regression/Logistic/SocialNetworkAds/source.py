import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class SocialNetworkAds:

  dataset = None
  X = X_train = X_test = None
  y = y_train = y_test = y_pred = None
  classifier = conf_mat = None


  def __init__(self):
      # Importing the dataset
      self.dataset = pd.read_csv('data.csv')
      self.X = self.dataset.iloc[:,[2,3]].values
      self.y = self.dataset.iloc[:,4].values

      # Splitting the dataset into the training set and test set
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
          self.X, self.y,
          test_size=0.25,
          random_state=0)


  def scale_features(self):
      '''
      Feature scaling
      '''
      sc = StandardScaler()
      self.X_train = sc.fit_transform(self.X_train)
      self.X_test = sc.transform(self.X_test)
    

  def fit_classifier(self):
      '''
      Fitting classifier to the training set and predicting
      the test set result
      '''
      self.classifier = LogisticRegression(random_state=0)
      self.classifier.fit(self.X_train, self.y_train)
      self.y_pred = self.classifier.predict(self.X_test)
    

  def make_confusion_matrix(self):
      '''
      Making the confusion matrix
      '''
      self.conf_mat = confusion_matrix(self.y_test, self.y_pred)
    

  def plot_training(self):
      '''
      Visualizing the training set results
      '''
      X_set, y_set = self.X_train, self.y_train
      X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, 
                                     stop=X_set[:,0].max()+1,
                                     step=0.01),
                           np.arange(start=X_set[:,1].min()-1,
                                     stop=X_set[:,1].max()+1,
                                     step=0.01))
      plt.contourf(X1, X2, self.classifier.predict(
                   np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                   alpha=0.75, cmap=ListedColormap(('red', 'green')))
      plt.xlim(X1.min(), X1.max())
      plt.ylim(X2.min(), X2.max())
      for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
      plt.title('Classifier (Training set)')
      plt.xlabel('Age')
      plt.ylabel('Estimated Salary')
      plt.legend()
      plt.show()
    

  def plot_testing(self):
      '''
      Visualizing the testing set results
      '''
      X_set, y_set = self.X_test, self.y_test
      X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,
                                     stop=X_set[:,0].max()+1,
                                     step=0.01),
                           np.arange(start=X_set[:,1].min()-1,
                                     stop=X_set[:,1].max()+1,
                                     step=0.01))
      plt.contourf(X1, X2, self.classifier.predict(
                   np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                   alpha=0.75, cmap=ListedColormap(('red', 'green')))
      plt.xlim(X1.min(), X1.max())
      plt.ylim(X2.min(), X2.max())
      for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
      plt.title('Classifier (Test set)')
      plt.xlabel('Age')
      plt.ylabel('Estimated Salary')
      plt.legend()
      plt.show()


def run():
  social_network_ads = SocialNetworkAds()
  social_network_ads.scale_features()
  social_network_ads.fit_classifier()
  social_network_ads.make_confusion_matrix()
  social_network_ads.plot_training()
  social_network_ads.plot_testing()

if __name__ == '__main__':
  run()
