from IPython import get_ipython
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,BaggingClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline


 

get_ipython().run_line_magic('matplotlib', 'qt')

names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df = pd.read_csv('breast-cancer-wisconsin.data.csv',names=names)

df.hist()
df.plot(kind='density',subplots=True,layout=(4,3),sharex=False,sharey=False)

print(df.corr(method='pearson'))

df.drop(['Sample code number'],axis=1,inplace=True)    
df.drop(['Uniformity of Cell Shape'],axis=1,inplace=True) #High Correlation between Cell Shape and Cell Size
df.drop(df.loc[df['Bare Nuclei']=='?'].index, inplace=True)



array = df.values
x = array[:,0:8]
y = array[:,8]

encoder = LabelEncoder().fit(y) #Encode Output into binary values
new_y = encoder.transform(y)


x_train,x_test,y_train,y_test = train_test_split(x,new_y,test_size=0.20,random_state=7)



models=[]

models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('KN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

results=[]
names=[]

for name,model in models:
    
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result = cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy') 
    results.append(cv_result)
    names.append(name)
    
    msg = "%s: %f %f"%(name,cv_result.mean()*100,cv_result.std())
    print(msg)
    

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM',
SVC(gamma='auto'))])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)
    

solver = ['newton-cg','lbfgs','liblinear','sag','saga']
param_grid = dict(solver=solver)
model = LogisticRegression()
scaler = StandardScaler().fit(x_train)
rescaled_x = scaler.transform(x_train)
kfold = KFold(n_splits=10,shuffle=True,random_state=7)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=kfold)
grid_result = grid.fit(rescaled_x,y_train)
print(grid_result.best_params_,grid_result.best_score_)

scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
model=SVC(gamma='auto')
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values,kernel=kernel_values)
kfold = KFold(n_splits=10,random_state=7,shuffle=True)
grid  = GridSearchCV(estimator=model,param_grid=param_grid,cv=kfold,scoring='accuracy')
grid_result = grid.fit(rescaledx,y_train)
print(grid_result.best_score_ , grid_result.best_params_)


model = LogisticRegression(solver='newton-cg')
scaler = StandardScaler().fit(x_train)
rescaled_x = scaler.transform(x_train)
rescaled_xt = scaler.transform(x_test)
model.fit(rescaled_x,y_train)
predictions = model.predict(rescaled_xt)
print(accuracy_score(y_test,predictions)*100)


model = SVC(C=0.3,kernel='linear')
scaler = StandardScaler().fit(x_train)
rescaled_x = scaler.transform(x_train)
rescaled_xt = scaler.transform(x_test)
model.fit(rescaled_x,y_train)
predictions = model.predict(rescaled_xt)
print(accuracy_score(y_test,predictions)*100)
