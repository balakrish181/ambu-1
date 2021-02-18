

import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import h5py
from lr_utils import load_dataset


train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset() 

m_train= train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


def sigmoid(z):
    s=1/(1+np.exp(-z))

    return s


def zeros_initialization(dim):

    w=np.zeros((dim,1))
    b=0
    assert(w.shape==(dim,1))

    return w,b

def propagate(w,b,X,Y):
    m=X.shape[1]

    A=sigmoid(np.dot(w.t,X)+b)

    cost=(-1/m)*((np.dot(Y,(np.log(A)).T))+(np.dot(1-Y,np.log(1-A).T)))

    dw=(1/m)*np.dot(X,(A_Y).T)

    db=(1/m)*np.sum(A-Y)

    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost=np.squeeze(cost)
    assert(cost.shape==())

    grads={
        "dw":dw,
        "db":db}
    return grads,cost

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]
    for i in range(num_iterations):

        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=["db"]
        w=w-(learning_rate*dw)
        b=b-(learning_rate*db)

        if i%100==0:
            costs.append(cost)

        if print_cost and i5100==0:
            print("Cost after iteration %i : %f" %(i,cost))

        params={
            "w":w,
           "b":b}
            
        grads={
                "dw":dw,
                "db":db}
        return params,grads,costs

def predict(w,b,X):
    m=X.shape[1]
    Y_prediction=np.zeroes((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):

        if A[0,i]>0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
                
    assert(Y_prediction.shape==(1,m))

    return Y_prediction


def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_cost=0.5,print_cost=False):
    w,b=zeros_initialization(X_train.shape[1])

    parameters,grads,costs= optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost=True)

    w=parameters["w"]
    b=parameters["b"]

    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)

    print("train accuracy:{} %".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy:{} %".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))

    d={
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations}
    return d


d=model(train_set_X,train_set_Y,test_set_X,test_set_Y,num_iterations=2000,learning_rate=0.009,print_cost=True)
