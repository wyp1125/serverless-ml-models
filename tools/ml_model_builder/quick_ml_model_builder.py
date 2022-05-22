import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='This program quickly builds a binary machine learning model using predefined parameters for testing purpose')
parser.add_argument('-i', '--inprefix', type=str, required=True, help="prefix of feature and response files (.codedx & .codedy)")
parser.add_argument('-m', '--mlmodel', type=int, required=True, help="ml model choice - 0:Nearest Neighbors, 1:Linear SVM, 2:RBS SVM, 3:Decision Tree, 4:Random Forest, 5:Neural Net, 6:AdaBoost, 7:Naive Bayes")
parser.add_argument("-t", '--testsize', type=float, required=False,  help="fraction for testing data (default: 0.2)")
args = parser.parse_args()
testratio=0.2
if args.testsize:
    testratio=args.testsize

clsf_names=["Nearest Neighbors", "Linear SVM", "RBF SVM",
            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            "Naive Bayes"]

classifiers = [KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]

if args.mlmodel<0 or args.mlmodel>7:
    print("Classifier ID should be between 0 and 7")
    exit()

clsf = classifiers[args.mlmodel]
print("Chosen ML model: "+str(clsf))
feafile=args.inprefix+".codedx"
rspfile=args.inprefix+".codedy"
fea=pd.read_csv(feafile)
rsp=pd.read_csv(rspfile).iloc[:,0]
x_trn, x_tst, y_trn, y_tst = train_test_split(fea, rsp, test_size=testratio)
model=clsf.fit(x_trn,y_trn)
y_pred=model.predict(x_tst)
acc=float((y_pred==y_tst).sum())/float(len(y_tst))
print("\nModel accuracy: {0: .3f}%\n".format(acc*100))
print(classification_report(y_tst,y_pred))
