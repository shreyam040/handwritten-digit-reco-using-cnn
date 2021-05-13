#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[8]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[9]:


len(X_train)


# In[10]:


len(X_test)


# In[11]:


X_train[0].shape


# In[12]:


X_train[0]


# In[18]:


plt.matshow(X_train[0])


# In[19]:


model.evaluate(X_test,y_test)


# In[21]:


X_train = X_train / 255
X_test = X_test / 255


# In[22]:


X_train[0]


# In[23]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[24]:


X_train_flattened.shape


# In[25]:



X_train_flattened[0]


# In[27]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[29]:


model.evaluate(X_test_flattened, y_test)


# In[30]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[31]:


plt.matshow(X_test[0])


# In[32]:


np.argmax(y_predicted[0])


# In[33]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[34]:


y_predicted_labels[:5]


# In[35]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[37]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[38]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[39]:


model.evaluate(X_test_flattened,y_test)


# In[40]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[41]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


# In[42]:


model.evaluate(X_test,y_test)


# In[43]:


plt.matshow(X_train[99])


# In[44]:


y_train(99)


# In[45]:


y_train[99]


# In[46]:


plt.matshow(X_train[7000])


# In[47]:


y_train(7000)


# In[48]:


y_train[7000]


# In[49]:


plt.matshow(X_train[7000])


# In[50]:


y_train[7000]

