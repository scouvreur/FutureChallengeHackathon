# import tutorial
import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

def readData():
    '''
    This function reads in the hdf5 file - it takes
    around 3s on average to run on a
    dual processor workstation
    '''
    # read h5 format back to numpy array
    global citydata
    global train
    global test
    h5f = h5py.File('METdata.h5', 'r')
    # citydata = h5f['citydata'][:]
    train = h5f['train'][:]
    test = h5f['test'][:]
    h5f.close()

readData()

sum = 0
model_MAE = {}
for i in range(10):
	model_MAE[i] = mean_absolute_error(train[:,:,i,:,:].flatten(),
									   train[:,:,10,:,:].flatten())
	print(model_MAE[i])

min_model_key = min(model_MAE, key=model_MAE.get)
min_model_MAE = 1-(min(model_MAE.values())/train[:,:,10,:,:].mean())

print('Model {} on average performs best with an accuracy of {:0.2f}%.'.format(min_model_key,100*min_model_MAE))

# Taken the average of 10 models as prediction value

train_set = train[:3,:,:10,:,:]
valid_set = train[3:,:,:10,:,:]
train_label = train[:3,:,10,:,:]
valid_label = train[3:,:,10,:,:]

test_set = test[:,:,:,:,:]

# Gradient boosting regression model

train_set = np.column_stack((train_set[:,:,i,:,:].flatten() for i in range(10)))
valid_set = np.column_stack((valid_set[:,:,i,:,:].flatten() for i in range(10)))
test_set = np.column_stack((test_set[:,:,i,:,:].flatten() for i in range(10)))

train_label = train_label.flatten()
valid_label = valid_label.flatten()

# model = XGBRegressor()
model = SGDRegressor(loss='huber', max_iter=1000, tol=1e-3)

model.fit(train_set,train_label)
MAE = mean_absolute_error(model.predict(valid_set),valid_label)
error = 1 - (MAE/train[:,:,10,:,:].mean())

print('The model has an accuracy of {:0.2f}%.'.format(100*error))

GBReg = model.predict(test_set)
GBReg = GBReg.reshape(5,18,548,421)

h5f = h5py.File('GBReg.h5', 'w')
h5f.create_dataset('GBReg', data=GBReg)
h5f.close()
