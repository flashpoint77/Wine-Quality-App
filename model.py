import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy import stats
from sklearn import metrics
from sklearn import preprocessing
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle


## Importar librerías
"""

# import de librerías
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy import stats
from sklearn import metrics
from sklearn import preprocessing
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

"""## Adquisicion - Lectura datasets"""
df_wines = pd.read_csv('wine-quality-white-and-red.csv',sep=',')

"""##  Feature Engineering-  Pre procesamiento de variables categoricas"""
df_wines = df_wines.drop('total sulfur dioxide', axis=1)
#One Hot Encoding de variable "Type"
# OPCION 1: df_wines.replace({'white': 1, 'red': 0}, inplace=True)
#OPCION 2:
df_wines['type'] = pd.get_dummies(df_wines['type'], drop_first = True)

#One Hot Encoding de variable "Quality"
df_wines['best quality'] = [1 if x > 5 else 0 for x in df_wines.quality]
df_wines

## Separamos en datos de entrenamiento y datos de prueba - Train & Test Split

features = df_wines.drop(['type','quality', 'best quality'], axis=1)
target = df_wines['best quality'] 
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=40)
print('Division de los datos en train y test (80%/20%)\n')
print ("Conjunto de entrenamiento:", x_train.shape, y_train.shape)
print ("Conjunto de prueba       : ", x_test.shape, y_test.shape)

"""## Escalado de datos con MinMaxScaler"""
features = df_wines.drop(['type','quality', 'best quality'], axis=1)
target = df_wines['best quality']
# # Creamos objeto 
#escalado = MinMaxScaler()
# # fit data
#x_train = escalado.fit_transform(x_train)
#x_test = escalado.transform(x_test)

"""### Random Forest"""
rf = RandomForestClassifier(n_estimators=200)
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)
print(classification_report(y_test, pred_rf))
cross_val = cross_val_score(estimator=rf, X=x_train, y=y_train, cv=10)
print("Matriz de confusion\n",confusion_matrix(y_test, pred_rf))
print("Cross validation \n",cross_val.mean())

Reporte_modelos= pd.DataFrame({'Modelos': ["Random Forest"],
                           'Precision_del_modelo': [  accuracy_score(y_test,pred_rf)]})
Reporte_modelos



# Make pickle file of our model
pickle.dump(rf, open("model.pkl", "wb"))