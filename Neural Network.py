import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

#Build Model
model = keras.Sequential([keras.layers.Flatten(input_shapes = (2,)) , keras.layers.Dense(100,activation = tf.nn.relu) , keras.layers.Dense(4,activation = tf.nn.softmax)])

#Compile Model
model.compile(optimizer = 'adam' , loss = 'spares_categorical_crossentropy' , metrics = ['accuracy'])

#Preparing Data For Training
x = []
y = []

for i in range(100):
    a , b = [random.randint(0,1) , random.randint(0,1)]
    x.append([a,b])
    if a == 0 and b == 0:
        y.append([0])
    if a == 0 and b == 1:
        y.append([1])
    if a == 1 and b == 0:
        y.append([2])
    if a == 1 and b == 1:
        y.append([3])

x = np.array(x)
y = np.array(y)

#Training Model
model.fit(x , y , epochs = 5)

#Preparing Data For Testing
x1 = [[0,0] , [1,1] , [1,0] , [1,0] , [0,1]]
y1 = [ [0]  ,  [3]  ,  [2]  ,  [2]  ,  [1] ]
x1 = np.array(x1)
y1 = np.array(y1)

#Testing Model(Evaluate Accuracy)
test_loss , test_acc = model.evaluate(x1 , y1)
print('Test Accuracy = ' , test_acc)

#Preparing Data For Prediction
p = [[1,1]]
p = np.array(p)

#Prediction Model
pred = model.predict(p)
print('Prediction = ' , p)

print(np.argmax(pred))

    
