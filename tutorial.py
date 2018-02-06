import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import mean_absolute_error

def loadData():
    '''
    This function reads in all the CSV data and saves it to an
    hdf5 file - it takes around 23mins on average to run on a
    dual processor workstation
    '''
    global citydata
    global train_wind
    global train_rain
    global test_wind
    global test_rain
    citydata = pd.read_csv('cityData.csv')

    # initialize an empty 5D tensor
    train_wind = np.zeros((5,19,11,548,421))
    train_rain = np.zeros((5,19,11,548,421))

    print('start processing traindata')
    with open('forecastDataforTraining.csv') as trainfile:
        for index,line in enumerate(trainfile):
            # traindata format
            # xid,yid,date_id,hour,model,wind,rainfall
            # 1,1,1,3,1,13.8,41.4

            traindata = line.split(',')
            try:
                x = int(traindata[0])
                y = int(traindata[1])
                d = int(traindata[2])
                h = int(traindata[3])
                m = int(traindata[4])
                w = float(traindata[5])
                r = float(traindata[6])
                train_wind[d-1,h-3,m-1,x-1,y-1] = w
                train_rain[d-1,h-3,m-1,x-1,y-1] = r

                if index%1000000==0:
                    print(index,"lines have been processed")
            except ValueError:
                print("found line with datatype error! skip this line")
                continue

    print('start processing labeldata')
    with open('insituMeasurementforTraining.csv') as labelfile:
        for index,line in enumerate(labelfile):
            # labeldata format
            # xid,yid,date_id,hour,wind,rainfall
            # 1,1,1,3,12.8,1.1
            labeldata = line.split(',')
            try:
                lx = int(labeldata[0])
                ly = int(labeldata[1])
                ld = int(labeldata[2])
                lh = int(labeldata[3])
                lw = float(labeldata[4])
                lr = float(labeldata[5])
                train_wind[ld-1,lh-3,10,lx-1,ly-1] = lw
                train_rain[ld-1,lh-3,10,lx-1,ly-1] = lr

                if index%1000000==0:
                    print(index,"lines have been processed")
            except ValueError:
                print("found line with datatype error! skip this line")
                continue

    test_wind = np.zeros((5,18,10,548,421))
    test_rain = np.zeros((5,18,10,548,421))

    print('start processing testdata')
    with open('forecastDataforTesting.csv') as testfile:
        for index,line in enumerate(testfile):
            # testdata format
            # xid,yid,date_id,hour,model,wind,rainfall
            # 1,1,1,3,1,13.8,1.9

            testdata = line.split(',')
            try:
                x = int(testdata[0])
                y = int(testdata[1])
                d = int(testdata[2])
                h = int(testdata[3])
                m = int(testdata[4])
                w = float(testdata[5])
                r = float(testdata[6])
                test_wind[d-6,h-3,m-1,x-1,y-1] = w
                test_rain[d-6,h-3,m-1,x-1,y-1] = r

                if index%1000000==0:
                    print(index,"lines have been processed")
            except ValueError:
                print("found line with datatype error! skip this line")
                continue

def saveData():
    '''
    This function writes all the data to an hdf5 file
    '''
    # write numpy arrary tensor into h5 format
    h5f = h5py.File('METdata.h5', 'w')
    h5f.create_dataset('citydata', data=citydata)
    h5f.create_dataset('train_wind', data=train_wind)
    h5f.create_dataset('train_rain', data=train_rain)
    h5f.create_dataset('test_wind', data=test_wind)
    h5f.create_dataset('test_rain', data=test_rain)
    h5f.close()

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
    h5f.close()

loadData()
saveData()
readData()

def plotWindMap(day, hour, model):
    '''
    This function takes as input from the user the day,
    hour, and model number and plots a map of the wind
    speed
    '''
    if (0 <= day <= 4) and (3 <= hour <= 20) and (0 <= model <= 10):
        plt.title('Wind speed on Day {} at {:02d}:00\nModel {}'.format(day,hour,model))
        plt.imshow(train[int(day),int(hour),int(model),:,:].T, cmap='bone')
        plt.colorbar(ticks=[0, 10, 20], orientation='vertical')
        for cid,xid,yid in citydata:
            if cid == 0:
                plt.plot(xid-1,yid-1,'yo')
            else:
                plt.plot(xid-1,yid-1,'ro')
        # plt.savefig('windMaps/Day-{}-Hour-{}-Model-{}.pdf'.format(day,hour,model), format='pdf')
        plt.show()
    else:
        print('Please enter a day between 0 and 4')
        print('Please enter an hour between 3 and 20')
        print('Please enter a model between 0 and 10')
