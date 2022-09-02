# linear-regression
data.shape
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X=data.loc[0:896,['Items_Available','Daily_Customer_Count']]
Y=data.loc[0:896,['Store_Sales']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, train_size = .75)
#Creating object
regressor=LinearRegression()
#training the model
regressor.fit(X_train,Y_train)
data.isnull().sum()
pred=regressor.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error
mse= mean_squared_error (Y_test,pred)
r2=r2_score(Y_test,pred) 
#Results
print("Mean Squared Error:",mse)
print("R-Squared:",r2)
print("Y-intercept:",regressor.intercept_)
print("Slope:",regressor.coef_)
plt.scatter(Y_test,pred);
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.regplot(x=Y_test,y=pred,ci=None,color='Red')




#logistic regression
data.tail()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X=dataset.loc[0:768,['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Dia
betesPedigreeFunction','Age']] 
Y=dataset.loc[0:768,['Outcome']] 
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,random_state=0,train_size=0.75) 
logreg= LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred_test = logreg.predict(X_test) 
Y_pred_train = logreg.predict(X_train)
from sklearn.metrics import confusion_matrix 
cnf_matrix_test = confusion_matrix(Y_test, Y_pred_test) 
cnf_matrix_train = confusion_matrix(Y_train, Y_pred_train) 
print('Confusion Matrix For Test Data = ', cnf_matrix_test)
print('Confusion Matrix For Train Data = ', cnf_matrix_train)
 
#Test Data
class_names=[0,1] # name of classes
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix_test), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout()
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#Train Data
class_names=[0,1] # name of classes
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix_train), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
from sklearn.metrics import accuracy_score,precision_score,recall_score
#Test Data
print('Precision_Score_test = ', precision_score(Y_test, Y_pred_test)) 
print('Recall_Score_test = ', recall_score(Y_test, Y_pred_test)) 
print('Accuracy_Score_test = ', accuracy_score(Y_test, Y_pred_test)) 
# Train Data
print('Precision_Score_train = ', precision_score(Y_train, Y_pred_train))
print('Recall_Score_train = ', recall_score(Y_train, Y_pred_train)) 
print('Accuracy_Score_train = ', accuracy_score(Y_train, Y_pred_train))
logreg.score(X_train,Y_train)


#knn
dataset.tail()
from sklearn.model_selection import train_test_split
X=dataset.loc[0:768,['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Dia
betesPedigreeFunction','Age']] 
Y=dataset.loc[0:768,['Outcome']] 
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,random_state=0,train_size=0.70) 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) 
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
cm = confusion_matrix(Y_test, Y_pred) 
ac = accuracy_score(Y_test,Y_pred)
pc= precision_score(Y_test,Y_pred) 
print(ac)
print(pc) 
#Train Data
class_names=[0,1] # name of classes
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout()
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#svm
dataset.isnull().sum()
from sklearn.model_selection import train_test_split
X=dataset.loc[0:918,['Age','RestingBP’,'Cholesterol', 'FastingBS', 'MaxHR','Oldpeak']] 
Y=dataset.loc[0:918,'HeartDisease']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,
train_size=0.75) 
from sklearn.svm import SVC
SVCClf = SVC(kernel = 'linear',gamma = 'scale', shrinking = False)
SVCClf.fit(X_train,Y_train)
Y_pred=SVCClf.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score 
cm = confusion_matrix(Y_test, Y_pred) 
ac = accuracy_score(Y_test,Y_pred)
pc= precision_score(Y_test,Y_pred) 
print(ac)
print(pc) 
class_names=[0,1] 
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" )
ax.xaxis.set_label_position("top") 
plt.tight_layout()
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#nvb
dataset.tail()
dataset.shape
from sklearn.model_selection import train_test_split
X=dataset.loc[0:918,['Age','RestingBP' ,'Cholesterol', 'FastingBS', 'MaxHR','Oldpeak']] 
Y=dataset.loc[0:918,'HeartDisease']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,train_size=0.75) 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
Y_pred = gnb.fit(X_train,Y_train).predict(X_test) 
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
pc=precision_score(Y_test,Y_pred) 
ac=accuracy_score(Y_test,Y_pred)
print("Acccuracy Score:",ac)
print("Precison Score:",pc)
cm = confusion_matrix(Y_test, Y_pred) 
#Train Data
class_names=[0,1] # name of classes
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names) 
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout()
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#descision
dataset=pd.read_csv("heart.csv")
dataset.tail()
dataset.shape
from sklearn.model_selection import train_test_split
X=dataset.loc[0:918,['Age','RestingBP' ,'Cholesterol', 'FastingBS', 'MaxHR','Oldpeak']] 
Y=dataset.loc[0:918,'HeartDisease']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,train_size=0.75) 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
clf_en =DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, Y_train)
Y_pred = clf_en.predict(X_test) 
print('Training set score: {:.4f}'.format(clf_en.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(clf_en.score(X_test, Y_test)))
plt.figure(figsize=(12,8)) 
from sklearn import tree
tree.plot_tree(clf_en.fit(X_train, Y_train))
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
pc=precision_score(Y_test,Y_pred) 
ac=accuracy_score(Y_test,Y_pred)
rc=recall_score(Y_test,Y_pred)
print("Acccuracy Score:",ac)
print("Precison Score:",pc)
print("Recall Score",rc)
cm = confusion_matrix(Y_test, Y_pred) 
#Train Data
cm = confusion_matrix(Y_test, Y_pred) 
class_names=[0,1] # name of classes
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top") 
plt.tight_layout()
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#hill climbing
import random
def randomSolution(tsp):
 cities = list(range(len(tsp)))
 solution = []
 for i in range(len(tsp)):
 randomCity = cities[random.randint(0, len(cities) - 1)]
 solution.append(randomCity)
 cities.remove(randomCity)
 return solution
def routeLength(tsp, solution):
 routeLength = 0 
 for i in range(len(solution)):
 routeLength += tsp[solution[i - 1]][solution[i]]
 return routeLength
def getNeighbours(solution):
 neighbours = []
 for i in range(len(solution)):
 for j in range(i + 1, len(solution)):
 neighbour = solution.copy()
 neighbour[i] = solution[j]
 neighbour[j] = solution[i]
 neighbours.append(neighbour)
 return neighbours
def getBestNeighbour(tsp, neighbours):
 bestRouteLength = routeLength(tsp, neighbours[0])
 bestNeighbour = neighbours[0]
 for neighbour in neighbours:
 currentRouteLength = routeLength(tsp, neighbour)
 if currentRouteLength < bestRouteLength:
 bestRouteLength = currentRouteLength
 bestNeighbour = neighbour
 return bestNeighbour, bestRouteLength
def hillClimbing(tsp):
 currentSolution = randomSolution(tsp)
 currentRouteLength = routeLength(tsp, currentSolution)
 neighbours = getNeighbours(currentSolution)
 bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)
 while bestNeighbourRouteLength < currentRouteLength:
 currentSolution = bestNeighbour
 currentRouteLength = bestNeighbourRouteLength
 neighbours = getNeighbours(currentSolution)
 bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)
 return currentSolution, currentRouteLength
def main():
 tsp = [ 
 [0, 400, 500, 300],
 [400, 0, 300, 500],
 [500, 300, 0, 400],
 [300, 500, 400, 0]
 ] 
 print(hillClimbing(tsp))
if __name__ == "__main__":
 main(if __name__ == "__main__":
 main()
 
 
 #mean
 import matplotlib.pyplot as plt 
runs_list = []
name_list = ["RAHUL", "AJAY"]
std_dev_list = []
for j in range(2):
 print(name_list[j])
 main_list = eval(input("Enter a list:"))
 runs_list.append(main_list) 
 main_list.sort()
 
 # Mean
 sum = 0 
 list_length = len(main_list)
 for i in main_list:
 sum += i 
 mean = sum/list_length
 print("Mean = ", mean)
 # Median
 if list_length % 2 == 0: 
 median = (main_list[(list_length // 2) - 1] + main_list[list_length // 2]) / 2 
 else:
 median = main_list[(list_length // 2)] 
 print("Median = ", median) 
 # Mode 
 max_count = 0 
 mode = 0 
 for i in main_list:
 count = main_list.count(i) 
 if count > max_count: 
 max_count = count 
 mode = i 
 if max_count == 1: 
 print("Mode Doesnt Exist.")
 else:
 print("Mode = ", mode) 
 # Variance
 var_list = []
 for l in main_list:
 var_list.append((l - mean)**2) 
 variance = 0 
 for k in range(list_length):
 variance += var_list[k]
 print("Variance = ", variance)
 # Standard Deviation
 std_dev = (variance / 5)**0.5 
 print("Standard Deviation = ", std_dev) 
 std_dev_list.append(std_dev) 
 print("=============================================================")
if std_dev_list[0] > std_dev_list[1]: 
 print(name_list[1], "will be selected.")
else:
 print(name_list[0], "will be selected.")
print("=============================================================")
 
print()
print()
print()
x = runs_list[0] 
plt.hist(x, ec = 'black', color = 'red')
plt.xlabel("Runs") 
plt.ylabel("Count") 
plt.title("Rahul")
plt.show() 
print()
print()
print()
y = runs_list[1] 
plt.hist(y, ec = 'black', color = 'green')
plt.xlabel("Runs") 
plt.ylabel("Count") 
plt.title("Ajay")
plt.show() 
