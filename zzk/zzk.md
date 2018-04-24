# assemble graph

## 1、Read data

## 2、use placeholders or tf.data for input( and label)

  - placeholder
	
  - tf.data
		
    - tf.data.DataSet
    
    - tf.data.Iterator

## 3、create weights and bias

## 4、define inference

## 5、specify loss function
	
  - least mean square
		
    - tf.square(Y-Y_predict,name='loss')
	
  - cross entropy
		
      - softmax first (用于将目标分成K类。形式类似于贝叶斯，不同点在于需要取指数。原因有两个：①让比重大的类别概率更大②让函数可导)
		
      - cross entropy between softmax and real label
    
## 6、choose optimizer
	
  - GradientDescentOptimizer
	
  - MomentumOptimizer
		
    - vt=γvt−1+η∇θJ(θ)
		
    - θ=θ−vt
	
  - AdagradOptimizer
	
  - [optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#fnref:2)

# train model

## 7、init variables

## 8、train algorithms
	
  - batch gradient descent(BGD)
	
  - stochastic gradient descent(SGD)
	
  - mini-batch gradient descent
