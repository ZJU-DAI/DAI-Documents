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
      交叉熵

	### 信息量

		I（x0）= -log(p(x0)) 0<p(x0)<1

		可以理解为一个事件发生的概率越大，它所携带的信息量就越小。

	### 熵（对于一个随机变量X，所有可能取值的信息量的期望（E[I(x)]））
	
		离散型
		H(X) = -∑p(x)logp(x) x∈X

		连续型
		H(X) = -∫p(x)logp(x)dx x∈X
		
	### 相对熵(KL散度)
		Dkl(p||q)=Ep[log(p(x)/q(x))]=∑p(x)log(p(x)/q(x)) x ∈X

	### 交叉熵
		CEH(p,q) = Ep[-logq]=-∑p(x)logq(x)=H(p)-Dkl(p||q)

	[概念介绍](https://blog.csdn.net/rtygbwwwerr/article/details/50778098)
	[softmax_cross_entropy_with_logits做法](https://blog.csdn.net/mao_xiao_feng/article/details/53382790)

    
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
