import os
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json,load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

parser = argparse.ArgumentParser(description='This program quickly builds a deep neural network (multi-layer perceptron) model for testing purpose')
parser.add_argument('-i', '--inprefix', type=str, required=True, help="prefix of feature & response files (.codedx & .codedy)")
parser.add_argument('-m', '--model', type=str, required=True, help="directory and prefix for saving the best model")
parser.add_argument('-st', '--testing_size', type=float, required=False, help="testing size (default: 0.2)")
parser.add_argument('-sv', '--valid_size', type=float, required=False, help="validation size (default: 0.2)")
parser.add_argument('-r', '--rate', type=float, required=False, help="learning rate (default: 0.001)")
parser.add_argument('-c', '--epochs', type=int, required=False, help="number of epochs (default: 20)")
parser.add_argument('-b', '--batch_size', type=int, required=False, help="batch size (default: 400)")
parser.add_argument('-d1', '--dense1', type=int, required=False, help="units of first dense layer (default: 100)")
parser.add_argument('-d2', '--dense2', type=int, required=False, help="units of second dense layer (default: 100)")
parser.add_argument('-d3', '--dense3', type=int, required=False, help="unites of thind dense layer (default: no third layer)")
parser.add_argument('-dp', '--dropout', type=float, required=False, help="ratio for dropout layers (default: 0.25)")
args = parser.parse_args()

feafile=args.inprefix+".codedx"
if not os.path.isfile(feafile):
    print("Could not find the feature file!")
    exit()
rspfile=args.inprefix+".codedy"
if not os.path.isfile(rspfile):
    print("Could not find the response file!")
    exit()
fea=pd.read_csv(feafile)
rsp=pd.read_csv(rspfile)
num_classes=len(rsp.columns)
saved_model=args.model
validation_ratio=0.2
if args.valid_size:
    validation_ratio=args.valid_size
testing_ratio=0.2
if args.testing_size:
    testing_ratio=args.testing_size
learning_rate = 0.001
if args.rate:
    learning_rate=args.rate
training_epochs = 20
if args.epochs:
    training_epochs=args.epochs
batch_size = 256
if args.batch_size:
    batch_size=args.batch_size
dropout_fr = 0.25
if args.dropout:
    dropout_fr=args.dropout
unit_1=512
unit_2=512
having_unit_3=False
if args.dense1:
    unit_1=args.dense1
if args.dense2:
    unit_2=args.dense2
if args.dense3:
    having_unit_3=True
    unit_3=args.dense3

x_trnval, x_tst, y_trnval, y_tst = train_test_split(fea.to_numpy(), rsp.to_numpy(), test_size=testing_ratio)
x_trn,x_val,y_trn,y_val = train_test_split(x_trnval, y_trnval, test_size=validation_ratio)

model=Sequential()
model.add(Dense(unit_1,activation='relu'))
model.add(Dropout(dropout_fr))
model.add(Dense(unit_2,activation='relu'))
model.add(Dropout(dropout_fr))
if having_unit_3:
    model.add(Dense(unit_3,activation='relu'))
    model.add(Dropout(dropout_fr))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=tf.optimizers.Adam(learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath=saved_model+'.h5', verbose=1, monitor='val_loss',save_best_only=True,save_weights_only=True)  
model.fit(x_trn, y_trn, validation_data=(x_val,y_val), epochs=training_epochs, verbose=1, callbacks=[checkpoint], batch_size=batch_size)
model_json=model.to_json()
with open(saved_model+'.json',"w") as of:
    of.write(model_json)
    of.close()
json_file=open(saved_model+'.json','r')
loaded_model_json=json_file.read()
json_file.close() 
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights(saved_model+'.h5')
loaded_model.compile(optimizer=tf.optimizers.Adam(learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
y_pred=np.argmax(loaded_model.predict(x_tst), axis=1)
print(classification_report(np.argmax(y_tst,axis=1),y_pred))
acc=float((y_pred==y_tst[:,1]).sum())/float(len(y_tst))
print("\nModel accuracy: {0: .3f}%\n".format(acc*100))
