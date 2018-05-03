import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd

Data="./slr05.xls"

#step 1: read in data from the .xls file
book=xlrd.open_workbook(Data,encoding_override="utf-8")
sheet=book.sheet_by_index(0)
data=np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])
n_samples=sheet.nrows-1

#step 2: create placeholder for input x (number of fire) and label y (number of theft)

x=tf.placeholder(tf.float32,name="x")
y=tf.placeholder(tf.float32,name="y")

#step 3: create weight and bias, initialized to 0
w=tf.Variable(0.0,name="w")
b=tf.Variable(0.0,name="b")

#step 4: construct model to predict y (number of theft) from the number of fire 
y_predicted=x*w+b

#step 5: use the square error as the loss function
loss=tf.square(y-y_predicted,name="loss")

#step 6: 
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    #step 7: initialize the necessary variables,in this case,w and b
    sess.run(tf.global_variables_initializer())

    #step 8: train the model
    for i in range(100):
        for x1,y1 in data:
            sess.run(optimizer,feed_dict={x:x1,y:y1})
    #step 9: output the values of w and b
    w_value,b_value=sess.run([w,b])
x,y=data.T[0],data.T[1]
plt.plot(x,y,'bo',label='Real data')
plt.plot(x,x*w_value+b_value,'r-',label="predicted data")
plt.legend()
plt.show()
