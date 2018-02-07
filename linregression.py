# import tutorial
import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def readData():
    '''
    This function reads in the hdf5 file - it takes
    around 3s on average to run on a
    dual processor workstation
    '''
    # read h5 format back to numpy array
    global citydata
    global train_wind
    global train_rain
    global test_wind
    global test_rain
    h5f = h5py.File('METdata.h5', 'r')
    citydata = h5f['citydata'][:]
    train_wind = h5f['train_wind'][:]
    train_rain = h5f['train_rain'][:]
    test_wind = h5f['test_wind'][:]
    test_rain = h5f['test_rain'][:]
    print("--- Data read in successfully ---")
    h5f.close()

readData()

wind_model_MAE = {}
for i in range(10):
	wind_model_MAE[i] = mean_absolute_error(train_wind[:,:,i,:,:].flatten(),
									   train_wind[:,:,10,:,:].flatten())

min_wind_model_key = min(wind_model_MAE, key=wind_model_MAE.get)
min_wind_model_MAE = 1-(min(wind_model_MAE.values())/train_wind[:,:,10,:,:].mean())

rain_model_MAE = {}
for i in range(10):
    rain_model_MAE[i] = mean_absolute_error(train_rain[:,:,i,:,:].flatten(),
                                       train_rain[:,:,10,:,:].flatten())

min_rain_model_key = min(rain_model_MAE, key=rain_model_MAE.get)
min_rain_model_MAE = 1-(min(rain_model_MAE.values())/train_rain[:,:,10,:,:].mean())

print('Model {} on average performs best on wind prediction with an accuracy of {:0.2f}%.'.format(min_wind_model_key,100*min_wind_model_MAE))
print('Model {} on average performs best on rainfall prediction with an accuracy of {:0.2f}%.'.format(min_rain_model_key,100*min_rain_model_MAE))

train_set = train_wind[:3,:,:10,:,:]
valid_set = train_wind[3:,:,:10,:,:]
train_label = train_wind[:3,:,10,:,:]
valid_label = train_wind[3:,:,10,:,:]

test_set = test_wind

# Linear regression model

train_set = np.column_stack((train_set[:,:,i,:,:].flatten() for i in range(10)))
valid_set = np.column_stack((valid_set[:,:,i,:,:].flatten() for i in range(10)))
test_set = np.column_stack((test_set[:,:,i,:,:].flatten() for i in range(10)))

train_label = train_label.flatten()
valid_label = valid_label.flatten()

lr = LinearRegression(fit_intercept=True, normalize=True)
lr.fit(train_set,train_label)
linreg_MAE = mean_absolute_error(lr.predict(valid_set),valid_label)
linreg_percent_error = 1 - (linreg_MAE/train_wind[:,:,10,:,:].mean())

print('The linear regression model has an accuracy of {:0.2f}% on wind prediction.'.format(100*linreg_percent_error))

linreg_wind = lr.predict(test_set)
linreg_wind = linreg_wind.reshape(5,18,548,421)

train_set = train_rain[:3,:,:10,:,:]
valid_set = train_rain[3:,:,:10,:,:]
train_label = train_rain[:3,:,10,:,:]
valid_label = train_rain[3:,:,10,:,:]

test_set = test_rain

# Linear regression model

train_set = np.column_stack((train_set[:,:,i,:,:].flatten() for i in range(10)))
valid_set = np.column_stack((valid_set[:,:,i,:,:].flatten() for i in range(10)))
test_set = np.column_stack((test_set[:,:,i,:,:].flatten() for i in range(10)))

train_label = train_label.flatten()
valid_label = valid_label.flatten()

lr = LinearRegression(fit_intercept=True, normalize=True)
lr.fit(train_set,train_label)
linreg_MAE = mean_absolute_error(lr.predict(valid_set),valid_label)
linreg_percent_error = 1 - (linreg_MAE/train_rain[:,:,10,:,:].mean())

print('The linear regression model has an accuracy of {:0.2f}% on rainfall prediction.'.format(100*linreg_percent_error))

linreg_rain = lr.predict(test_set)
linreg_rain = linreg_rain.reshape(5,18,548,421)

h5f = h5py.File('LinReg.h5', 'w')
h5f.create_dataset('linreg_wind', data=linreg_wind)
h5f.create_dataset('linreg_rain', data=linreg_rain)
h5f.close()
