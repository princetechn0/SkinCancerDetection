
import json
import csv
import cv2
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
import tensorflow as tf
import tensorflow.keras
import io
import requests
import numpy as np
from sklearn import metrics
from glob import glob
import os
from collections.abc import Sequence
from sklearn import preprocessing
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.callbacks import EarlyStopping
import sklearn.feature_extraction.text as sk_text
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure, show
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from os.path import isfile
from PIL import Image as pil_image

import skimage.io
from skimage.io import imread_collection



## Helpful Functions for Tensorflow
# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)



# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)


# Convert all missing values in the specified column to the default
def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.
    target_type = df[target].dtypes
    target_type = target_type[0] if isinstance(target_type, Sequence) else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df[result].values.astype(np.float32), df[target].values.astype(np.float32)

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# Regression chart.
def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten() })
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean()) >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low

# Plot a confusion matrix.
# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred,y):
    fpr, tpr, thresholds = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#################################### 
#DATA PREPROCESSING

#Reading in CSV
# metadata = pd.read_csv("./data/metadata.csv", na_values=['NA','?'])
# # #Appropriate Columns

# # #Drop Columns with unnecessary info, duplicates, and empty columns
# metadata = metadata.drop_duplicates(subset=None, keep='first', inplace=False)
# metadata = metadata.dropna()
# drop_columns = ['sex', 'age']

# metadata.drop(drop_columns, axis=1, inplace=True)

# metadata.to_csv("./cleaned.csv", index=False)

num_diseases = 7

condition_name = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec ',
    5: 'vasc',
    6: 'df'
}

lesion_type = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# Cleaned CSV
metadata_cleaned = pd.read_csv("./data/cleaned.csv", na_values=['NA','?'])




# Merge images from both folders into one dictionary and resize them

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob("./data/images/*.jpg")}

metadata_cleaned['path'] = metadata_cleaned['image_id'].map(imageid_path_dict.get)

print("resizing")

metadata_cleaned['image'] = metadata_cleaned['path'].map(lambda i: np.asarray(pil_image.open(i).resize((100,75))))



# Output Features
encode_text_index(metadata_cleaned, "dx")

diagnosis = encode_text_index(metadata_cleaned, "dx")

# #Encoding Categorical Features  using Dummies
categorical_features = ["localization", "dx_type"]
for i in categorical_features:
    encode_text_dummy(metadata_cleaned, i)


metadata_cleaned.drop("lesion_id", axis=1, inplace=True)
metadata_cleaned.drop("image_id", axis=1, inplace=True)
metadata_cleaned.drop("path", axis=1, inplace=True)


metadata_ready = metadata_cleaned

print(metadata_ready.head())


x = metadata_ready
y = metadata_ready.dx


# The data split between train and test sets:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# # Define ModelCheckpoint outside the loop
checkpointer = ModelCheckpoint(filepath="./best_weights.hdf5", verbose=0, save_best_only=True) # save best model


print(x.shape)
print(y.shape)


print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


y_train = tf.keras.utils.to_categorical(y_train, num_diseases)
y_test = tf.keras.utils.to_categorical(y_test, num_diseases)

x_train = np.asarray(x_train['image'].tolist())
x_test = np.asarray(x_test['image'].tolist())

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255





# CNN Design


for i in range(2):

  cnn = Sequential()

  input_shape = (75, 100, 3)

  batch_size = 120
  

  cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                  activation='relu',
                  input_shape=input_shape))

  cnn.add(MaxPooling2D(pool_size=(1,2)))

  cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

  cnn.add(MaxPooling2D(pool_size=(1,2)))

  cnn.add(Flatten())
  cnn.add(Dense(128, activation="relu"))

  cnn.add(Dropout(0.5))
  cnn.add(Dense(num_diseases, activation="softmax"))

  cnn.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])

  monitor = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=2, verbose=1, mode='auto')

  cnn.fit(x_train[:8000], y_train[:8000],     
          batch_size=batch_size,
          callbacks=[monitor,checkpointer],
          epochs=10,
          verbose=2,
          validation_data=(x_test[:800], y_test[:800]))



# print('Training finished...Loading the best model')  

cnn.load_weights('./best_weights.hdf5') # load weights from best model


from sklearn import metrics


# evaluate() computes the loss and accuracy
score = cnn.evaluate(x_test[:800], y_test[:800], verbose=0)

print(score)


y_true = np.argmax(y_test[:800],axis=1)
pred = cnn.predict(x_test[:800])
pred = np.argmax(pred,axis=1)

print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


print(metrics.classification_report(y_true, pred))



#Printing out True Results vs. Predicted

from matplotlib.pyplot import imshow
def draw_img(i):
    im = x_test[i]
    print("method 1")
    print("True: %s (Predicted: %s)" % (condition_name[y_true[i]], condition_name[pred[i]]))
    plt.imshow(im)
    plt.title("True: %s (Predicted: %s)" % (condition_name[y_true[i]], condition_name[pred[i]]))
    plt.axis('on')
    plt.show()


#Testing 5 images
draw_img(1)

draw_img(2)

draw_img(3)



draw_img(100)

draw_img(101)

draw_img(102)



draw_img(150)
draw_img(151)

draw_img(152)


draw_img(198)

draw_img(197)

draw_img(199)



draw_img(647)
draw_img(648)

draw_img(649)












