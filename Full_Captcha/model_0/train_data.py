import numpy as np
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from google.colab import drive

# drive.mount("/content/gdrive")

# tf.test.gpu_device_name()

# location = "/content/gdrive/My Drive/Colab Notebooks/MultiDigit_CNN/"
location = "./"

train_data0 = np.load(location + 'Data/data0.npy')
train_lab0 = np.load(location + 'Label/lab0.npy')

train_data1 = np.load(location + 'Data/data1.npy')
train_lab1 = np.load(location + 'Label/lab1.npy')

train_data2 = np.load(location + 'Data/data2.npy')
train_lab2 = np.load(location + 'Label/lab2.npy')

train_data = np.vstack((train_data0, train_data1, train_data2))
train_lab = np.hstack((train_lab0, train_lab1, train_lab2))
print("Original dataset sizes:", train_data.shape, train_lab.shape)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_lab, test_size=0.2, random_state=42)

img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
input_shape = (img_rows, img_cols, 1)

X_train = X_train.reshape(-1, img_rows, img_cols, 1)
X_test = X_test.reshape(-1, img_rows, img_cols, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
print("Train Test split:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print("Label counts y_train:", np.unique(y_train, return_counts=True))

#set number of categories
num_category = 37

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)
print("One-hot encoded label size:", y_train.shape, y_test.shape)

print("Building Model")

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding = "same", activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', padding = "same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), padding = "same", activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding = "same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), padding = "same", activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding = "same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_category, activation='softmax'))

print("Compiling Model")
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

print("Training Model")
batch_size = 100
num_epoch = 12
model_log = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print("Model Metrics\n%s: %.2f%%, %s: %.2f" %(model.metrics_names[1], score[1]*100, model.metrics_names[0], score[0]))

fig = plt.figure()
plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('Accuracy vs Epoch')
plt.ylabel('Accuracy')
plt.xlabel('# Epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(1,2,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('Loss vs Epoch')
plt.ylabel('Loss')
plt.xlabel('# Epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.savefig("Plots.png")

file_name = location + 'model_1'

# Save the model
# serialize model to JSON
model_json = model.to_json()
with open(file_name + ".json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(file_name +".h5")
print("Saved model to disk")

# choice = int(input("Do you want to load any saved model? Enter 1 for Yes and 0 for No: "))
# if(choice == 1):

#   acc_tr = int(input("Enter train accuracy rounded to the nearest integer: "))
#   acc_va = int(input("Enter validation accuracy rounded to the nearest integer: "))
#   file_name = location + 'model_' + str(acc_tr) + "_" + str(acc_va)

#   # load json and create model
#   json_file = open(file_name + '.json', 'r')
#   loaded_model_json = json_file.read()
#   json_file.close()
#   loaded_model = model_from_json(loaded_model_json)
#   # load weights into new model
#   loaded_model.load_weights(file_name + ".h5")
#   print("Loaded model from disk")

#   # evaluate loaded model on test data
#   loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
#   score = loaded_model.evaluate(X_test, y_test, verbose=0)
#   print("Loaded Model Metrics\n%s: %.2f%%, %s: %.2f" % (loaded_model.metrics_names[1], score[1]*100, loaded_model.metrics_names[0], score[0]))
