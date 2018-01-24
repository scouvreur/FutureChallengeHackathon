from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
# from tutorial import readData, saveData
import h5py
from sklearn.metrics import mean_absolute_error

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
    citydata = h5f['citydata'][:]
    train = h5f['train'][:]
    test = h5f['test'][:]
    h5f.close()

readData()

548*421

# This is the size of our encoded representations

input_img = Input(shape=(28,28,10))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# At this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

train_set = train[:4,:,:10,130:158,130:158]/50.0
valid_set = train[4,:,:10,130:158,130:158]/50.0
train_label = train[:4,:,10,130:158,130:158]/50.0
valid_label = train[4,:,10,130:158,130:158]/50.0

print(train_set.shape)
print(valid_set.shape)
print(train_label.shape)
print(valid_label.shape)

train_set = train_set.reshape(4*18,28,28,10)
valid_set = valid_set.reshape(18,28,28,10)
train_label = train_label.reshape(4*18,28,28,1)
valid_label = valid_label.reshape(18,28,28,1)

print(train_set.shape)
print(valid_set.shape)
print(train_label.shape)
print(valid_label.shape)

autoencoder.fit(train_set, train_label,
                epochs=1000,
                batch_size=18,
                shuffle=True,
                validation_data=(valid_set, valid_label))

# Autoencoder error

print(1-(0.0309/valid_label.mean()))

# Average model error

print(mean_absolute_error(valid_set.mean(axis=1).flatten(),valid_label.flatten()))
print(1-(0.031123083900226757/valid_label.mean()))

autoencoder = np.zeros((5,18,548,421))

# writeData()