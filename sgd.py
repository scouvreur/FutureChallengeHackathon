import tutorial
import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

tutorial.readData()

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

# Linear regression model

train_set_1 = np.column_stack((train_set[:,:,i,:,:].flatten() for i in range(10)))
valid_set_1 = np.column_stack((valid_set[:,:,i,:,:].flatten() for i in range(10)))

test_set_1 = test.flatten()
train_label_1 = train_label.flatten()
valid_label_1 = valid_label.flatten()

lr = LinearRegression(fit_intercept=True, normalize=True)
lr.fit(train_set_1,train_label_1)
linreg_MAE = mean_absolute_error(lr.predict(valid_set_1),valid_label_1)
linreg_percent_error = 1 - (linreg_MAE/train[:,:,10,:,:].mean())

print('The linear regression model has an accuracy of {:0.2f}%.'.format(100*linreg_percent_error))

lr.predict(train_set_1)
lr.predict(valid_set_1)
lr.predict(train_set_1).shape
lr.predict(valid_set_1).shape

# linreg = np.concatenate((lr.predict(train_set_1), lr.predict(valid_set_1)), axis=0)
Linreg = lr.predict(test_set_1)
linreg = linreg.reshape(5,18,548,421)

h5f = h5py.File('LinReg.h5', 'w')
h5f.create_dataset('linreg', data=linreg)
h5f.close()