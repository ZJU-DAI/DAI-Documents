# Implention of Word2Vec

## Phase 1: Assemble the Graph

### 1. Create dataset and generate samples.

Input:center word,output:context word.Create a dictionary of the most common words and feed the indices of those words.

BATCH_SIZE of the sample inputs have shape [BATCH_SIZE],for the outputs have shape [BATCH_SIZE,1]

code

```python
dataset = tf.data.Dataset.from_generator(gen, 
                            (tf.int32, tf.int32), 
                            (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
iterator = dataset.make_initializable_iterator()
center_words, target_words = iterator.get_next()
```

### 2. Define the Weight(in this case,embedding matrix)

Each row corresponds to the representation vector of one word.If one word is represented with a vector of size EMBED_SIZE, the embedding matrix will have shape[VOCAB_SIZE, EMBED_SIZE]. The embedding matrix is initilized to value from a random distribution. In this case, it's uniform distribution.

```python
embed_matrix = tf.get_variable('embed_matrix', 
                               shape = [VOCAB_SIZE,EMBED_SIZE],                             initializer=tf.random_uniform_initializer())
```

### 3. Inference(compute the forward path of the graph)

The embed_matrix has dimension VOCAB_SIZE * EMBED_SIZE, with each row of  that corresponds to the vector representation of the word at that index.Use the tf.nn.embedding_lookup to get the slice of all coreesponding rows in the embedding matrix.

```python
embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
```

### 4. Define the loss function

NCE loss has been implented by Tensorflow, just use it.

Note that the thrid argument is actually inputs, and the forth is labels. And also we need weights and bias to calculate NCE loss. They will be updated by optimizer. After sampling, the final output will be computed in tf.nn.nce_loss operation.

```python
#create nce_weight and bias
nce_weight = tf.get_variable('nce_weight',
          shape=[VOCABE_SIZE, EMBED_SIZE],
         initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))
nce_bias = tf.get_variable('nce_bias',initializer=tf.zeros([VOCAB_SIZE]))
```

```python
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                    biases=nce_bias,
                                    labels=target_words,
                                    inputs=embed,
                                    num_sampled=NUM_SAMPLED,
                                    num_classes=VOCAB_SIZE))
```

### 5. Define the optimizer

```Python
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
```

##Phase 2: Execute the computation 

```python
with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter('graph/word2vec_simple', sess.graph)
    
    for index in range(NUM_TRAIN_STEPS):
        try:
            loss_batchm,_ = sess.run([loss, optimizer])
        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
    writer.close()
```

Full model in the code word2vec.py, and word2vec_eager with the eager execution.





