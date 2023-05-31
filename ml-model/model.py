from __future__ import print_function
import numpy as np
import tensorflow as tf

"""
  05/30/2023
"""

train_mode='train'
test_mode='test'

class model:

  def __init__(self,mode=None,Linear=True,params=None):
    self.in_sz = params['points'] # size of points 
    self.true_sz = params['true_sz']
    stack = False
    self.rnns = {}
    self.params=params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    self.sess=tf.Session(config=config)
    self.mode = mode
    ## 1.a Construct graph
    # inputs and true value placeholders
    self.points = tf.placeholder(tf.float32, [None, self.in_sz], name='points') 
    self.coor_true = tf.placeholder(tf.float32, [None, self.true_sz], name='coor_true') 
        
    if Linear:
      ## 1.b Construct graph
      self.beta = tf.get_variable("beta", None, tf.float32,
                                  tf.random_normal([self.in_sz, self.true_sz],stddev=0.01))
      self.u = tf.get_variable('bias', None, tf.float32, 
                                tf.random_normal([self.true_sz], stddev=0.01))
      self.eps = tf.get_variable("eps", None, 
                                  tf.float32,tf.random_normal([self.true_sz], stddev=0.01)) 
      self.out = tf.add(tf.add(tf.matmul(self.points, self.beta), self.u), self.eps)
      # metrics
      self.residuals  = self.out - self.coor_true 
      self.error = (self.residuals/self.coor_true)
      self.fitted = self.out/tf.log(self.out) # normalized fitted due to scale
      self.cost = tf.square(self.residuals) # use square error for cost function
      self.sse = tf.cumsum(self.cost) # sum squared error metric
      self.loss = tf.reduce_sum(self.cost)
      if mode == 'train':
          ## 3. Loss + optimization
        g_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.02
        self.lr = tf.train.exponential_decay(starter_learning_rate, g_step, 300000, 0.9, staircase=True)
        opti = tf.train.AdamOptimizer(self.lr)
        # compute and apply gradients
        self.LMts_ = opti.minimize(self.loss, global_step=g_step)
      else:
        self.ts_=None

    else:
      self.seqlen = params['seqlen']
      # 1st layer
      self.w = tf.get_variable('w', None, tf.float32, 
                            tf.random_normal([self.seqlen, self.true_sz], stddev=0.01) ) 
      self.b = tf.get_variable('b', None, tf.float32, 
                            tf.random_normal([self.true_sz], stddev=0.01))
      ## 2. create the model
      basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.seqlen,
                                                activation=tf.nn.tanh)   
      self.outs, self.states = tf.nn.static_rnn(basic_cell,
                                            tf.split(X,2,axis=1),
                                            dtype=tf.float32)   
      self.z = tf.add(tf.matmul(self.states.c,self.w), self.b)
      self.residuals = self.z - self.coor_true
      self.error = (self.residuals/self.coor_true) # normalized due to scale points variance 
      self.fitted = self.z/tf.log(self.z) # normalized fitted due to scale
      self.cost = tf.square(self.residuals)
      self.sse = tf.cumsum(self.cost) # tf.cumsum depreciated use tf.math.cumsum instead
      self.loss = tf.reduce_sum(self.cost)   

    if mode == 'train':
        ## 3. Loss + optimization
      g_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 0.02
      self.lr = tf.train.exponential_decay(starter_learning_rate, g_step, 300000, 0.9, staircase=True)
      opti = tf.train.AdamOptimizer(lr)
      if stack:
        s_ts_ = opti.minimize(s_loss, global_step=g_step)

      else:
        # compute and apply gradients
        self.ts_ = opti.minimize(loss, global_step=g_step)

    vars_to_save={v.name: v for v in
                  tf.trainable_variables() + tf.get_collection("bn_pop_stats") + tf.get_collection("bn_counts")}
    #summary_writer = tf.train.SummaryWriter("/home/pogo/QS/SingleTrackRNN.logs", sess.graph)
    self.saver=tf.train.Saver(var_list=vars_to_save,max_to_keep=self.params['max_checkpoints'])
    self.sess.run(tf.global_variables_initializer())
   
  def feed_dict(self,batch,nextp):
      fd={}
      fd[self.points] = batch
      fd[self.coor_true] = nextp
      return fd

  def train(self,points,nextp):
    #Inputs needed by model for a report
    fd = self.set_feed_dict(points, nextp)
    lr = 0.0
    if self.mode==train_mode:
      req_list=[self.LMts_, self.out, self.sse, self.error, self.residuals, self.fitted] 
      ts, y_est, sqerr, err, resi, fitted = self.sess.run(req_list,feed_dict=fd)
    else: # leave out train steps
      req_list=[self.out,self.sse, self.error, self.residuals, self.fittted]
      y_est, sqerr, err, resi, fitted = self.sess.run(req_list,feed_dict=fd)
      lr = 0.0
    return y_est, sqerr, err, resi, fitted
    
  def save(self, path,i):
    save_path = self.saver.save(self.sess, path+str(i)+'.ckpt',write_meta_graph=False)
    
  def restore(self, path):
    self.saver.restore(self.sess, path)
         

