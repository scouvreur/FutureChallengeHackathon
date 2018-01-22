import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py

citydata = pd.read_csv('cityData.csv')
citydata

train = np.zeros((5,20-3+1,11,548,421)) #initialize an empty 5D tensor
print 'start processing traindata'
with open('forecastDataforTraining.csv') as trainfile:
    for index,line in enumerate(trainfile):
        #traindata format
        #xid,yid,date_id,hour,model,wind
        #1,1,1,3,1,13.8

        traindata = line.split(',')
        try:
            x = int(traindata[0])
            y = int(traindata[1])
            d = int(traindata[2])
            h = int(traindata[3])
            m = int(traindata[4])
            w = float(traindata[5])
            train[d-1,h-3,m-1,x-1,y-1] = w # write values into tensor

            if index%1000000==0:
                print '%i lines has been processed' %(index)
        except ValueError:
            print "found line with datatype error! skip this line"
            continue

print 'start processing labeldata'
with open('insituMeasurementforTraining.csv') as labelfile:
    for index,line in enumerate(labelfile):
        #labeldata format
        #xid,yid,date_id,hour,wind
        #1,1,1,3,12.8
        labeldata = line.split(',')
        try:
            lx = int(labeldata[0])
            ly = int(labeldata[1])
            ld = int(labeldata[2])
            lh = int(labeldata[3])
            lw = float(labeldata[4])
            train[ld-1,lh-3,10,lx-1,ly-1] = lw
            if index%1000000==0:
                print '%i lines has been processed' %(index)
        except ValueError:
            print "found line with datatype error! skip this line"
            continue


test = np.zeros((5,20-3+1,10,548,421))
print 'start processing testdata'
with open('forecastDataforTesting.csv') as testfile:
    for index,line in enumerate(testfile):
        #testdata format
        #xid,yid,date_id,hour,model,wind
        #1,1,1,3,1,13.8

        testdata = line.split(',')
        try:
            x = int(testdata[0])
            y = int(testdata[1])
            d = int(testdata[2])
            h = int(testdata[3])
            m = int(testdata[4])
            w = float(testdata[5])
            test[d-6,h-3,m-1,x-1,y-1] = w

            if index%1000000==0:
                print '%i lines has been processed' %(index)
        except ValueError:
            print "found line with datatype error! skip this line"
            continue


#In[8]:

train.shape


#In[9]:

train


#In[ ]:




#In[22]:

# write numpy arrary tensor into h5 format
h5f = h5py.File('METdata.h5', 'w')
h5f.create_dataset('train', data=train)
h5f.create_dataset('test', data=test)
h5f.close()


#In[5]:

# read h5 format back to numpy array
h5f = h5py.File('METdata.h5', 'r')
train = h5f['train'][:]
test = h5f['test'][:]
h5f.close()


#In[4]:

print 548*421*18*5*10
print train.shape


#In[ ]:




#In[7]:

plt.figure(figsize=(20,10))
plt.imshow(train[0,0,0,:,:].T)
for c,x,y in zip(citydata.cid,citydata.xid,citydata.yid):
    if c == 0:
        plt.plot(x-1,y-1,'yo')
    else:
        plt.plot(x-1,y-1,'ro')
plt.show()


#In[8]:

plt.figure(figsize=(20,10))
plt.imshow(train[0,0,10,:,:].T)
for c,x,y in zip(citydata.cid,citydata.xid,citydata.yid):
    if c == 0:
        plt.plot(x-1,y-1,'yo')
    else:
        plt.plot(x-1,y-1,'ro')
plt.show()


#In[10]:

train[0,0,10,:,:].T-train[0,0,0,:,:].T


#In[11]:

plt.figure(figsize=(20,10))
plt.imshow(test[0,0,1,:,:].T)
for c,x,y in zip(citydata.cid,citydata.xid,citydata.yid):
    if c == 0:
        plt.plot(x-1,y-1,'yo')
    else:
        plt.plot(x-1,y-1,'ro')
plt.show()


#In[ ]:




# # Result submission
# <img src="../submission1.png" alt>
#
# #### After evaluation
# <img src="../submission2.png" alt>
#
# ## Evaluation time
# # <span style="color:red">One chance for very single hour between 12:00-18:00, 7 times in total</span>


# # <span style="color:red"> Requirement!</span>
# ### <li>No column name!</li>
# ### <li>City_id, Date_id, Time, Coordinate_x, Coordinate_y</li>
# ### <li>Always include the starting time/location and end time/location</li>
# ### <li>Time must be continuous for every 2 minutes, no repeated time</li>
# ### <li>No coordinates outside (548,421)</li>
# ### <li>Starting time is always 03:00 and no time later than 21:00</li>
# ### <li>Conservative route for first submission just check your format, if you got 72000 after 12:00, come with me and I will help you, each team will have only one chance for offine check</li>

#In[16]:

citydata


#In[ ]:




# # Machine Learning Example

#In[29]:

from sklearn.metrics import mean_absolute_error


#In[54]:

train.shape


#In[6]:

train[:,:,10,:,:]


#In[8]:

for i in range(10):
    print mean_absolute_error(train[:,:,i,:,:].flatten(),train[:,:,10,:,:].flatten())


#In[153]:

1-(1.698811266478261/train[:,:,10,:,:].mean())


# # Taken the average of 10 models as prediction value

#In[27]:

train_set = train[:4,:,:10,:,:]
valid_set = train[4,:,:10,:,:]
train_label = train[:4,:,10,:,:]
valid_label = train[4,:,10,:,:]


#In[30]:

mean_absolute_error(valid_set.mean(axis=1).flatten(),valid_label.flatten())


#In[31]:

1-(2.0349824438973374/valid_label.mean())


# # Linear regression model

#In[99]:

train_set_1 = np.column_stack((train_set[:,:,i,:,:].flatten() for i in range(10)))
valid_set_1 = np.column_stack((valid_set[:,i,:,:].flatten() for i in range(10)))


#In[106]:

train_label_1 = train_label.flatten()
valid_label_1 = valid_label.flatten()


#In[109]:

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr = LinearRegression(fit_intercept=True, normalize=True)
lr.fit(train_set_1,train_label_1)
mean_absolute_error(lr.predict(valid_set_1),valid_label_1)

from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K


#In[5]:

548*421


#In[148]:

# this is the size of our encoded representations

input_img = Input(shape=(28,28,10))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')


#In[141]:

train_set = train[:4,:,:10,130:158,130:158]/50.0
valid_set = train[4,:,:10,130:158,130:158]/50.0
train_label = train[:4,:,10,130:158,130:158]/50.0
valid_label = train[4,:,10,130:158,130:158]/50.0


#In[142]:

print train_set.shape
print valid_set.shape
print train_label.shape
print valid_label.shape


#In[145]:

train_set = train_set.reshape(4*18,28,28,10)
valid_set = valid_set.reshape(18,28,28,10)
train_label = train_label.reshape(4*18,28,28,1)
valid_label = valid_label.reshape(18,28,28,1)


#In[146]:

print train_set.shape
print valid_set.shape
print train_label.shape
print valid_label.shape


#In[149]:

autoencoder.fit(train_set, train_label,
                epochs=1000,
                batch_size=18,
                shuffle=True,
                validation_data=(valid_set, valid_label))


# ## Autoencoder error

#In[150]:

1-(0.0309/valid_label.mean())


# ## Average model error

#In[143]:

mean_absolute_error(valid_set.mean(axis=1).flatten(),valid_label.flatten())


#In[144]:

1-(0.031123083900226757/valid_label.mean())


#In[ ]:




# # Route Planning

#In[159]:

import datetime


#In[157]:

citydata


#In[306]:

def manhatten_distance(a,b):
    #a,b should be a tuple with (x,y) as coordinate
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def move(pos,command):
    if command == 'stay':
        outpos = pos
    elif command == 'up':
        outpos = (pos[0],pos[1]+1)
    elif command == 'down':
        outpos = (pos[0],pos[1]-1)
    elif command == 'left':
        outpos = (pos[0]-1,pos[1])
    elif command == 'right':
        outpos = (pos[0]+1,pos[1])
    else:
        print 'unknown command'
        raise
    return outpos


#In[328]:

start_pos = (142,328)
end_pos = (199,371)
weather_matrix = train[1,:,10,:,:]
start_time = datetime.datetime(2017,1,1,hour=3,minute=0)
current_time = start_time
current_pos = start_pos
state_dict = {'stay':0,'up':0,'down':0,'left':0,'right':0}

# greedy policy, choose move that minimize the manhatten_distance, if cant make it, then wait
while True:
    if weather_matrix[start_time.hour-3,start_pos[0]-1,start_pos[1]-1] >= 15:
        print 'can not start, just crush'
        break
    else:
        print 'current pos is now at %s'%(current_pos,)
        print 'current time is now at %s'%(current_time)
        # update the dict using the manhatten_distance
        for k in state_dict.keys():
            state_dict[k] = manhatten_distance(end_pos,move(current_pos,k))
        # sorted the dict using the manhatten_distance
        sorted_dict = sorted(state_dict.items(), key=lambda x: x[1])

        # check weather is good
        if weather_matrix[current_time.hour-3,move(current_pos,sorted_dict[0][0])[0]-1,move(current_pos,sorted_dict[0][0])[1]-1] < 15:
            current_pos = move(current_pos,sorted_dict[0][0])
            print 'action %s is executed!'%(sorted_dict[0][0])
            print 'current pos is now at %s'%(current_pos,)

        elif weather_matrix[current_time.hour-3,move(current_pos,sorted_dict[1][0])[0]-1,move(current_pos,sorted_dict[1][0])[1]-1] < 15:
            current_pos = move(current_pos,sorted_dict[1][0])
            print 'action %s is executed!'%(sorted_dict[1][0])
            print 'current pos is now at %s'%(current_pos,)

        else:
            current_pos = move(current_pos,sorted_dict[2][0])
            print 'action %s is executed!'%(sorted_dict[2][0])
            print 'current pos is now at %s'%(current_pos,)

        # check weather the balloon is at the end pos
        if current_pos == end_pos:
            print 'Successfully arrived at end_pos %s'%(end_pos,)
            print 'total time consumed %s'%(current_time-start_time)
            break
        current_time += datetime.timedelta(minutes=2)


#In[200]:

citydata

#In[338]:

start_pos = (142,328)
end_pos_list = [(84,203),(199,371),(140,234),(236,241),(315,281),(358,207),(363,237),(423,266),(125,375),(189,274)]
weather_matrix = test[:,:,:10,:,:].mean(axis=2)
start_time = datetime.datetime(2017,1,1,hour=3,minute=0)
end_time = datetime.datetime(2017,1,1,hour=21,minute=0)
current_time = start_time
current_pos = start_pos
state_dict = {'stay':0,'up':0,'down':0,'left':0,'right':0}

# greedy policy, choose move that minimize the manhatten_distance, if cant make it, then wait
with open('log2.csv', 'a') as log:
    for day in range(5):
        print 'day %i is started!'%(day)

        for pos_index,end_pos in enumerate(end_pos_list):
            print 'Now end pos is %s'%(end_pos,)

            log.write(str(pos_index+1)+','+str(day+6)+','+str(current_time.time())[:-3]+','+str(current_pos[0])+','+str(current_pos[1])+'\n')

            while True:
                if weather_matrix[day,start_time.hour-3,start_pos[0]-1,start_pos[1]-1] >= 15:
                    print 'can not start, just crush'
                    break
                else:
                    #print 'current time is now at %s'%(current_time)
                    # update the dict using the manhatten_distance
                    for k in state_dict.keys():
                        state_dict[k] = manhatten_distance(end_pos,move(current_pos,k))
                    # sorted the dict using the manhatten_distance
                    sorted_dict = sorted(state_dict.items(), key=lambda x: x[1])
                    #print sorted_dict

                    # check weather is good
                    if weather_matrix[day,current_time.hour-3,move(current_pos,sorted_dict[0][0])[0]-1,move(current_pos,sorted_dict[0][0])[1]-1] < 14:
                        current_pos = move(current_pos,sorted_dict[0][0])
                        #print 'action %s is executed!'%(sorted_dict[0][0])
                        #print 'current pos is now at %s'%(current_pos,)

                    elif weather_matrix[day,current_time.hour-3,move(current_pos,sorted_dict[1][0])[0]-1,move(current_pos,sorted_dict[1][0])[1]-1] < 14:
                        current_pos = move(current_pos,sorted_dict[1][0])
                        #print 'action %s is executed!'%(sorted_dict[1][0])
                        #print 'current pos is now at %s'%(current_pos,)

                    else:
                        current_pos = move(current_pos,sorted_dict[2][0])
                        #print 'action %s is executed!'%(sorted_dict[2][0])
                        #print 'current pos is now at %s'%(current_pos,)


                    # check weather the balloon is at the end pos
                    if current_pos != end_pos:
                        current_time += datetime.timedelta(minutes=2)
                        log.write(str(pos_index+1)+','+str(day+6)+','+str(current_time.time())[:-3]+','+str(current_pos[0])+','+str(current_pos[1])+'\n')
                    else:
                        print 'Successfully arrived at end_pos %s'%(end_pos,)
                        print 'total time consumed %s'%(current_time-start_time)
                        current_time += datetime.timedelta(minutes=2)
                        log.write(str(pos_index+1)+','+str(day+6)+','+str(current_time.time())[:-3]+','+str(current_pos[0])+','+str(current_pos[1])+'\n')
                        current_time = start_time
                        current_pos = start_pos

                        break
                    # check weather time is up
                    if current_time == end_time:
                        print 'time is up and I cant make it'
                        current_time = start_time
                        current_pos = start_pos
                        break
                continue



#In[264]:

test_binary = weather_matrix < np.ones(weather_matrix.shape)*14


#In[265]:

test_binary


#In[274]:

test_binary[0,0,141,347]


#In[276]:

plt.figure(figsize=(20,10))
plt.imshow(test_binary[0,0,:,:].T)
for c,x,y in zip(citydata.cid,citydata.xid,citydata.yid):
    if c == 0:
        plt.plot(x-1,y-1,'bo')
    else:
        plt.plot(x-1,y-1,'ro')
plt.show()