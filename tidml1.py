import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from tensorflow.keras import initializers
from numpy import mean
from numpy import std
import numpy as np
import threading
import time
import csv

from numba import jit, cuda


accuracies, models, model_optimizers, model_epochs= list(), list(), list(), list()
hidden_lay_numbers, hidden_layer_neuron_numbers1, hidden_layer_neuron_numbers2= list(), list(),list()
model_folds,testratios = list(), list()
mutex = threading.Lock()

@cuda.jit
def evaluate_kfold_model(hidL,TraX,TeY,TsX,TsY,loss_func, optimizer_func,actf1,actf2,actfout,epo,hidlay1nc,hidlay2nc,tesr):
	# prepare the k-fold cross-validation configuration
	selectedmodels = list()
	selectedkfold  = list()
	selectedacc    = list()
	for n_folds in (3,5,7,10):
		if (n_folds==3):
			kfold = KFold(n_folds) #, True, 1)
			# cross validation estimation of performance
			scores, members = list(), list()
			#hafizam = list()
			print(n_folds,"_folds is started")
			validation_setnumber=0
			for train_ix, test_ix in kfold.split(TraX):
				validation_setnumber += 1
				#Create new Model INITIALIZATION of MODEL
				model = tf.keras.models.Sequential()
				model.add(tf.keras.layers.Dense(hidlay1nc, input_shape=(featurescount,) , activation = actf1,
							kernel_initializer=tf.keras.initializers.GlorotNormal(),
							#tf.keras.initializers.Orthogonal(),
							#tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
							bias_initializer=tf.keras.initializers.he_normal()))
				if (hidL==2):
					model.add(tf.keras.layers.Dense(hidlay2nc, activation = actf2,
							kernel_initializer=tf.keras.initializers.GlorotNormal(),
							#tf.keras.initializers.Orthogonal(),
							#tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
							bias_initializer=tf.keras.initializers.he_normal()))	
				model.add(tf.keras.layers.Dense(out_layer_neurons, activation=actfout))
				model.compile(loss = loss_func , optimizer = optimizer_func , metrics = ['accuracy'])
				# select samples
				trainX, trainy  = TraX[train_ix], TeY[train_ix]
				testX , testy   = TraX[test_ix] , TeY[test_ix]
				# evaluate model
				hafiza=model.fit( trainX, trainy,
				                batch_size	=128,
				                validation_data	=(testX, testy),
				                epochs		=epo,
				                verbose		=0)
				test_acc=hafiza.history["accuracy"][epo-1]
				print(validation_setnumber,". of",n_folds,"_fold validation accuracy",test_acc)
				scores.append(test_acc)
				members.append(model)
			print(n_folds,'_folds Mean Training Accuracy %.3f (%.3f)' % (mean(scores), std(scores)))
			#print("Infra selected model is ",scores.index(max(scores)))
			#print("Infra selected models score is ",scores[scores.index(max(scores))])
			#print("Infra selected models fold is ",n_folds)
			selectedmodels.append(members[scores.index(max(scores))])
			selectedkfold.append(n_folds)
			selectedacc.append(mean(scores))

	# max accuracy shows the selected model and its kfold
	indis = selectedacc.index(max(selectedacc))
	print("--------------------------------------")
	print("All kfolds are tested for the model...")
	print("Selected Model Fold=", selectedkfold[indis], "(indis)",indis)
	print("Selected Model Training Accuracy=",selectedacc[indis])
	print("--------------------------------------")
	# The model Predicted Classes and Confusions and the related metrics can be found with the followin;
	Predicted_Y_class = selectedmodels[indis].predict(TsX)
	Predicted_Y= np.argmax(Predicted_Y_class ,1)
	Predicted_Y=Predicted_Y.reshape(len(Predicted_Y),1)
	print("shape of Predicted Y_class:", Predicted_Y_class.shape, "shape of Predicted_Y:",Predicted_Y.shape, "shape of Test Y:",TsY.shape)
	## To see the each Predicted value with default Evaluated value use the following for checking
	#for i in range (0,len(Predicted_Y)):
	#	print(Predicted_Y[i], TsY[i])
	matched_Ys= np.sum(np.array(Predicted_Y)==np.array(TsY)) #how many of the classes are correctly predicted
	manuel_accuracy =matched_Ys/len(Predicted_Y)
	R2_Skoru  = r2_score(Predicted_Y, TsY)
	MSE       = mean_squared_error(Predicted_Y, TsY)
	MAE       = mean_absolute_error(Predicted_Y, TsY)
	MedAE     = median_absolute_error(Predicted_Y, TsY)
	# RAE -> Relative Absolute Error
	RAE		  = np.sum(np.abs(np.subtract(TsY,Predicted_Y))) / np.sum(np.abs(np.subtract(TsY, np.mean(TsY))))
	# RRSE-> Root Relative Squared Error
	RRSE	  = np.sqrt(np.sum(np.square(np.subtract(TsY,Predicted_Y))) / np.sum(np.square(np.subtract(TsY, np.mean(TsY)))))
	ConFMtrx  = confusion_matrix(TsY,Predicted_Y)
	PrecisS   = precision_score(TsY,Predicted_Y,average="weighted",zero_division=1)
	RecallS   = recall_score(TsY,Predicted_Y,average="weighted",zero_division=1)
	F1S   	  = f1_score(TsY,Predicted_Y,average="weighted",zero_division=1)
	CoKaS     = cohen_kappa_score(TsY,Predicted_Y )
	print("R2_Skor:",R2_Skoru, "MSE:",MSE, "MAE:",MAE,"MedAE:",MedAE,"Precision:",PrecisS,"RecallS:",RecallS,"F1:",F1S,"CohenKappa:",CoKaS)
	print("Confusion Matrix:",ConFMtrx)
	### print("PAUSED")
	### wait = input("Press Enter to continue.")
	# The evaluation results (accuracy and loss) can be directly found by evaluate function
	realloss, realacc = selectedmodels[indis].evaluate(TsX,TsY)
	print("Manual Accuracy:",manuel_accuracy,"Evaluated Accuracy:",realacc)
	add_model_to_list(optimizer_func,selectedkfold[indis],epo,hidL,hidlay1nc,hidlay2nc,realacc,tesr)
	#time.sleep(1)
	mutex.acquire()
	try:
		with open('balancerecords.log', 'a', newline='') as csvfile:
			babo_writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			#org1:babo_writer.writerow([tesr, realacc,manuel_accuracy, optimizer_func,epo,selectedkfold[indis],hidL,hidlay1nc,hidlay2nc])
			#org2:babo_writer.writerow([tesr, realacc,manuel_accuracy, optimizer_func,epo,selectedkfold[indis],hidL,hidlay1nc,hidlay2nc,R2_Skoru,MSE,MAE,MedAE,PrecisS,RecallS,F1S,CoKaS])
			babo_writer.writerow([tesr, realacc,manuel_accuracy, optimizer_func,epo,selectedkfold[indis],hidL,hidlay1nc,hidlay2nc])
		with open('confusions.log','a',newline='') as csvfile2:
			confm_writer = csv.writer(csvfile2,delimiter=';',quotechar='|',quoting=csv.QUOTE_MINIMAL)
			confm_writer.writerow(["TesTR:", tesr, "RealAcc:",realacc, "ManuelAcc:",manuel_accuracy,"R2 Skor:",R2_Skoru])
			confm_writer.writerow(["MSE:",MSE,"MAE:",MAE,"MedAE:",MedAE,"RAE:",RAE,"RRSE:",RRSE])
			confm_writer.writerow(["Precision:",PrecisS,"Recall:",RecallS,"F1:",F1S,"CohenKappa:",CoKaS])
			# org: confm_writer.writerow([tesr,realacc,manuel_accuracy,R2_Skoru,MSE,MAE,MedAE,PrecisS,RecallS,F1S,CoKaS])
			confm_writer.writerow(["Confusion Matrix"])
			confm_writer.writerow([ConFMtrx])
			confm_writer.writerow(["************************************"])
		csvfile.close()
		csvfile2.close()
	finally:
		mutex.release()
	return selectedkfold[indis], realloss, realacc

#def add_model_to_list(modl,optimf,kfinfo,epo,hidLc,hid1n,hid2n,racc,testra):
@cuda.jit
def add_model_to_list(optimf,kfinfo,epo,hidLc,hid1n,hid2n,racc,testra):
    #models.append(modl)
    model_optimizers.append(optimf)
    model_folds.append(kfinfo)
    model_epochs.append(epo)
    hidden_lay_numbers.append(hidLc)
    hidden_layer_neuron_numbers1.append(hid1n)
    hidden_layer_neuron_numbers2.append(hid2n)
    accuracies.append(racc)
    testratios.append(testra)
    return True



import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-set", "--data_set", required = False, default="./tidml.csv", help = "path to where Athlets balance board recordings are in")
ap.add_argument("-epoc", "--epochcount", required = False, default=10, help = "ML training of whole dataset loop count")
args = vars(ap.parse_args())


start_time = time.time()

# You can read data with Header Row like a single read_csv
# datasetoriginal = pd.read_csv(args["dataset"])
# then 
# dataset = datasetoriginal.iloc[0:].values
# Now, all values are as 2D matrix
# Dont forget now that, 
# dataset[0] means row. To reach 0.column use dataset[:,0]

dataset     = pd.read_csv(args["data_set"],skiprows=1,header=None)
r,c 		= dataset.shape  # i.e. r=870, c=15 0-13 features 14.target
featurescount = c-1
targetcol     = c-1
#dataset[4]  = pd.to_numeric(dataset[4],errors='coerce')  #dataset["SICRAMA"]
#dataset[0]  = dataset[0].astype(float)   # dataset["YAS"]
#dataset[4]  = dataset[4].astype(float)
#dataset[11] = dataset[11].astype(float) # dataset["0:L/1:R"]
####dataset["Class"]=lb_make.fit_transform(dataset["Class"])
# fit_transform convers target category to number i.e. p3,p7,p2 ==> 3,7,2
lb_make 	      = LabelEncoder()
dataset[targetcol]    = lb_make.fit_transform(dataset[c-1])
dataset[targetcol]    = dataset[c-1].astype(int)

X_input 	    = dataset.iloc[:,0:c-1].values
y	            = dataset.iloc[:,c-1:c].values

# Model : DNN
# Model parameters are adjusted here
epochnos = (100,200,300,500) 	    #org:(50,150,200,300,500)
activ_func1="relu"          #Alternatives "relu", "linear", "lrelu", "sigmoid", "tanh"
activ_func2="relu"          #Alternatives "relu", "linear", "lrelu", "sigmoid", "tanh"
activ_funcout="softmax"     #Alternative 
optimizers =("adam","rmsprop") #org ("rmsprop","adam")
loss_function = "sparse_categorical_crossentropy" #Alternative "sparse_categorical_crossentropy","binary_crossentropy","categorical_crossentropy"
testset_ratio = (0.2, 0.3)  # org (0.2, 0.3, 0.4)
hidlayns1= (32,64,128,256)  #org: (32,64,128,256,512)
hidlayns2= (32,64)   #org: (32,64,128,256,512)
out_layer_neurons=2
hidden_layersc=2        # org: (1,2)
kackeremean = 50	# org: 50

jobs = list()
for tsr in testset_ratio:
	for optf in optimizers:
		for epos in epochnos:
			for hidlynrs in hidlayns1:
				for hidlynrs2 in hidlayns2:
					if ((tsr==0.2)&(hidlynrs==128)&(hidlynrs2==64)&(epos==200)&(optf=="adam")):
						X_train, X_test, y_train, y_test = train_test_split(X_input, y, shuffle=True,test_size=tsr)
						print("====> Starting to write TRAINING and TEST SPLIT DATA on disk <======")
						original_dataset = pd.read_csv("tidml.csv")
						print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
						#x_tra=X_train.transpose()
						#y_tra=y_train.transpose()
						#Training_Matrix=np.concatenate((x_tra,y_tra))
						Training_Matrix=np.concatenate((X_train.transpose(),y_train.transpose()))
						Training_Csv	= pd.DataFrame(Training_Matrix.transpose())
						Training_Csv.columns=original_dataset.columns
						Training_Csv.to_csv("Data_Training_Split.csv",index=False) #,header=None) 
						
						#x_tes=X_test.transpose()
						#y_tes=y_test.transpose()
						#Testing_Matrix	= np.concatenate((x_tes,y_tes))
						Testing_Matrix	= np.concatenate((X_test.transpose(),y_test.transpose()))
						Testing_Csv		= pd.DataFrame(Testing_Matrix.transpose())
						Testing_Csv.columns=original_dataset.columns
						Testing_Csv.to_csv("Data_Testing_Split.csv",index=False) #,header=None) 
						print("====> Successfully TRAINING and TEST SPLIT DATA written on disk <======")
						
						for iteration in range (0,kackeremean):
							thread = threading.Thread(target=evaluate_kfold_model,args= 
												(hidden_layersc,X_train, y_train,
												X_test,y_test,
												loss_function,optf,
												activ_func1,activ_func2,activ_funcout,
												epos,hidlynrs,hidlynrs2,tsr ))
							jobs.append(thread)
							print("+")


print("Number of threads in Job List:", len(jobs))
acikthread=0 
for j in range (0,len(jobs)):
    jobs[j].start()
    print("Thread ", j, " is started")
    acikthread +=1
    if (acikthread==10):
        print("Son 10 Threadin kapanmasi bekleniyor")
        for i in range(j-9,j+1):
            print("Thread",i," is stopped")
            jobs[i].join()
            acikthread-=1
        print(j,".Thread kapanis Zamani")
        with open('balancetime.log', 'a', newline='') as timefile:
        	babo_writer = csv.writer(timefile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        	babo_writer.writerow([j, (time.time()-start_time)])
        timefile.close()
    
#for j in jobs:
#   j.join()

import csv
with open('balance.log', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for i in range (0,len(models)):

        print("******Model ",i,"*********")
        print("Split Ratio(test):",testratios[i])
        print("Test Accuracy    :",accuracies[i])
        print("Optimizer        :",model_optimizers[i])
        print("Epoch            :",model_epochs[i])
        print("kFold            :",model_folds[i])
        print("Hidden Layers    :",hidden_lay_numbers[i])
        print("1.HL Neurons     :",hidden_layer_neuron_numbers1[i])
        print("1.HL Neurons     :",hidden_layer_neuron_numbers2[i])
        
        spamwriter.writerow([testratios[i],accuracies[i],model_optimizers[i],model_epochs[i],model_folds[i],
                      hidden_lay_numbers[i],hidden_layer_neuron_numbers1[i],
                      hidden_layer_neuron_numbers2[i]])
    spamwriter.writerow(['All results are written....'])
