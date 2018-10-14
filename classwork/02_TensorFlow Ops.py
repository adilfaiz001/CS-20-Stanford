import tensorflow as tf

def part1(): 
    #tensorboard 
    a = tf.constant(2,name="A")
    b = tf.constant(3,name="B")
    x = tf.add(a, b,name="add")
    writer = tf.summary.FileWriter('./graphs/lecture02', tf.get_default_graph())
    
    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph
        writer.add_graph(sess.graph)
        print(sess.run(x))
        
    writer.close()

def part2():
    #constant,zeros,fill,lin_space,range
    a = tf.constant([2,2],name="vector")
    b = tf.constant([[0,1],[2,3]], name="matrix")
    c = tf.add(a, b, name="vector")
    
    d = tf.zeros(shape=[4,4], dtype=tf.float64, name="zeros")
    f = tf.fill([100,100], 8, name="fill")
    seq = tf.lin_space(10.0, 15.0, 200000, name="seq")
    r = tf.range(9, 100, 0.5, dtype=tf.float32, name='range')
    with tf.Session() as sess :
        print(sess.run(a))
        print(sess.run(b))
        print(sess.run(c))
        print(sess.run(d))
        print(type(sess.run(f)))
        print(sess.run(seq).shape)
        print(sess.run(r))

def part3():
    #operators
    a = tf.constant([10, 20], name='a')
    b = tf.constant([2, 3], name='b')
    with tf.Session() as sess:
        print(sess.run(tf.multiply(a, b)))           # element-wise multiplication
        print(sess.run(tf.tensordot(a, b, 1))) 
        

def part4():
    #get variables,initializers
    a = tf.get_variable(name='a', shape=[2,2], dtype=tf.float32,initializer=tf.truncated_normal_initializer()) 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))
        
def part5():
    #variable assign,assign_add,assign_sub
    a = tf.get_variable('scalar', initializer=tf.constant(2)) 
    a_times_two = a.assign(a * 2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        print(sess.run(a_times_two)) 
        
def part6():
    #placeholders, tensorboard,feed dict
    a = tf.placeholder(tf.float32, shape=[3], name='a')
    b = tf.constant([5, 5, 5], tf.float32,name='b')
    c = a + b
    writer = tf.summary.FileWriter('graphs/lecture02/placeholders', tf.get_default_graph())
    with tf.Session() as sess:
        writer.add_graph(sess.graph)
        print(sess.run(c,{a:[1,2,3]})) 
    writer.close()

def part7():
    #feedable tensors
    #tf.Graph.is_feedable(tensor) to check tensor is feedable or not
    a = tf.add(2, 5)
    b = tf.multiply(a, 3)
    
    with tf.Session() as sess:
        print(sess.run(b))                         # >> 21
        # compute the value of b given the value of a is 15
        print(sess.run(b, feed_dict={a: 15}))  
    
part2()