#importing part
import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class backend():

    def load_dataset(self):
        #load dataset
        self.dataset = pd.read_csv('diabetes.csv')
        #print( len(self.dataset) )
        #print( self.dataset.head())

    def remove_zeros(self):
        #remove zeros
        remove_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin' ]

        for column in remove_zero:
            self.dataset[column] = self.dataset[column].replace(0,np.NaN)
            mean= int( self.dataset[column].mean(skipna=True) )
            self.dataset[column] = self.dataset[column].replace(np.NaN, mean)

    def split(self):
        #split dataset

        self.X = self.dataset.iloc[:,0:8]
        self.Y = self.dataset.iloc[:, 8]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y, random_state=0, test_size=0.2)

    def Feature_Scaling(self):
        #Feature scaling

        self.sc_X = StandardScaler()
        self.X_train = self.sc_X.fit_transform(self.X_train)
        self.X_test = self.sc_X.transform(self.X_test)

    def model(self):
        # define the model : Init K-NN

        self.classifier = KNeighborsClassifier(n_neighbors=42, p=2, metric='euclidean')

        #fit Model

        self.classifier.fit(self.X_train, self.Y_train)
        KNeighborsClassifier(algorithm= 'auto', leaf_size=30, metric='euclidean', metric_params=None, n_jobs=1, n_neighbors=11, p=2, weights='uniform')

        # Predict the test set results
        self.Y_pred = self.classifier.predict(self.X_test)
        #self.Y_pred
        #print(self.Y_pred)

        #Evaluate the model
        self.cm = confusion_matrix(self.Y_test,self.Y_pred)
        print( self.cm )
        #self.cr=classification_report(self.Y_test,self.Y_pred)
        #print(self.cr)

    def scores(self):
        print("F1 Score : " , f1_score(self.Y_test, self.Y_pred) )
        print("Accuracy Score : " , accuracy_score(self.Y_test, self.Y_pred) )

    def predictor(self, newtext):
        self.sc_X1 = StandardScaler()
        self.X_test1 = newtext
        self.X_test1 = self.sc_X.transform(self.X_test1)
        #print(self.X_test)

        # print(self.X_test)
        self.Y_pred1 = self.classifier.predict(self.X_test1)
        # self.Y_pred
        print(self.Y_pred1)
        if (self.Y_pred1 == 0):
            print("Diabetes is Negative")
        else:
            print("Diabetes is Positive")

    def knn(self):

        d.load_dataset()
        d.remove_zeros()
        d.split()
        d.Feature_Scaling()
        d.model()
        d.scores()
        #d.predictor()
        #print(self.dataset.head(3))


d=backend()
#d.knn()

