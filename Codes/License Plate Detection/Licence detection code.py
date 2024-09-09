#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
import os
import glob


# In[2]:


IMG_SIZE = 224


# In[3]:


img_dir = r"D:\ALL SEMESTERS\Semester VIII\BS Project\images"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
files.sort()
X = []
for img in files:
    img = cv2.imread(img)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    X.append(np.array(img))


# In[4]:


X[0].shape


# In[5]:


from lxml import etree
tree = etree.parse("D:\\ALL SEMESTERS\\Semester VIII\\BS Project\\annotations\\Cars0.xml")
tree.getroot()
def resizeannotation(f):
    tree = etree.parse(f)
    for dim in tree.xpath("size"):
        width = int(dim.xpath("width")[0].text)
        height = int(dim.xpath("height")[0].text)
    for dim in tree.xpath("object/bndbox"):
        xmin = int(dim.xpath("xmin")[0].text)/(width/IMG_SIZE)
        ymin = int(dim.xpath("ymin")[0].text)/(height/IMG_SIZE)
        xmax = int(dim.xpath("xmax")[0].text)/(width/IMG_SIZE)
        ymax = int(dim.xpath("ymax")[0].text)/(height/IMG_SIZE)
    return [int(xmax), int(ymax), int(xmin), int(ymin)]


# In[6]:


path = "D:/ALL SEMESTERS/Semester VIII/BS Project/annotations/"
text_files = [path + i for i in sorted(os.listdir(path))]
y = []
for i in text_files:
#     etree.parse(i)
    y.append(resizeannotation(i))


# In[7]:


resizeannotation("D:/ALL SEMESTERS/Semester VIII/BS Project/annotations/Cars0.xml")


# In[8]:


y[:5]


# In[9]:


np.array(X).shape


# In[10]:


np.array(y).shape


# In[11]:


plt.figure(figsize=(10,20))

for i in range(0,18):
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(X[i])


# In[12]:


image = cv2.rectangle(X[0],(y[0][0],y[0][1]),(y[0][2],y[0][3]),(255, 0, 0))
plt.imshow(image)
plt.show()


# In[13]:


image = cv2.rectangle(X[2],(y[2][0],y[2][1]),(y[2][2],y[2][3]),(255, 0, 0))
plt.imshow(image)
plt.show()


# In[14]:


# Transforming the array to numpy array
X = np.array(X)
y = np.array(y)
X = X/255
y = y/255


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)


# In[16]:


X_train.shape


# In[17]:


X_test.shape


# In[21]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications.vgg19 import VGG19


# In[22]:


model = Sequential()
model.add(VGG19(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False

model.summary()


# In[23]:


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)


# In[ ]:


plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title("Accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[ ]:


model.evaluate(X_test,y_test)


# In[ ]:


model.save("number_plate_detection.h5")


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


predictions[:5]


# In[ ]:


plt.figure(figsize=(20,40))
for i in range(20,40) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    ny = predictions[i]*255
    image = cv2.rectangle(X_test[i],(int(ny[0]),int(ny[1])),(int(ny[2]),int(ny[3])),(255, 0, 0))
    plt.imshow(image)

