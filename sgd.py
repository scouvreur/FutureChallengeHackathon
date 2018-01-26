# import tutorial
import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.model_selection import cross_val_score

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

# Stochastic gradient descent regression model

train_set_1 = np.column_stack((train_set[:,:,i,:,:].flatten() for i in range(10)))
valid_set_1 = np.column_stack((valid_set[:,:,i,:,:].flatten() for i in range(10)))
test_set_1 = np.column_stack((test_set[:,:,i,:,:].flatten() for i in range(10)))

train_label_1 = train_label.flatten()
valid_label_1 = valid_label.flatten()

sgd = SGDRegressor(fit_intercept=True, loss="huber", penalty="l2",
				   tol=1e-6, max_iter=10000)
sgd.fit(train_set_1,train_label_1)
sgdreg_MAE = mean_absolute_error(sgd.predict(valid_set_1),valid_label_1)
sgdreg_percent_error = 1 - (sgdreg_MAE/train[:,:,10,:,:].mean())

print('The stochastic gradient descent regressor model has an accuracy of {:0.2f}%.'.format(100*sgdreg_percent_error))

SGDReg = sgd.predict(test_set_1)
SGDReg = SGDReg.reshape(5,18,548,421)

h5f = h5py.File('SGDReg.h5', 'w')
h5f.create_dataset('SGDReg', data=SGDReg)
h5f.close()