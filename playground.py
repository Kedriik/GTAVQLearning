import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os
import tensorflow as tf
from numpy_ringbuffer import RingBuffer
capture_region = (0,40,1280,720)
reshape_size = (int(720/2),int(1280/2))
def get_state():
    state = grab_screen(region = capture_region)
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state,(reshape_size[1],reshape_size[0]))
    state = state.reshape(reshape_size[0],reshape_size[1],1)
    return state
def show_frames(frames):
    for frame in frames:
        cv2.imshow('',frame)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
def test1():
    visual_history = RingBuffer(capacity=30, dtype=(float,(reshape_size[0],reshape_size[1],1)))
    for i in range(10):
        visual_history.append(np.array(get_state()))
    show_frames(visual_history)
class agentZ():
    def __init__(self, lr, s_size,a_size,h_size):
        self.state_in = tf.compat.v1.placeholder(shape=[None]+list(s_size),dtype=tf.float32)
        conv1         = slim.conv2d(self.state_in,num_outputs = 16, kernel_size = [4,4])
        max_pool1     = slim.max_pool2d(conv1,[2, 2])
        hidden        = slim.fully_connected(max_pool1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.flatten  = slim.flatten(hidden)
        self.logits   = slim.fully_connected(self.flatten,num_outputs=a_size,activation_fn=tf.nn.sigmoid,biases_initializer=None)
        self.action   = tf.random.categorical(logits=tf.math.log(self.logits),num_samples = 1)

class agentY():
    def __init__(self,lr,s_size,a_size,h_size):
        self.state_in = tf.compat.v1.placeholder(shape = [None]+list(s_size),dtype=tf.float32)
        conv1         = tf.compat.v1.layers.conv2d(self.state_in,32,4,strides=(4, 4))
        max_pool1     = tf.compat.v1.layers.max_pooling2d(conv1,32,4)
        flatten       = tf.compat.v1.layers.flatten(max_pool1)
        hidden        = tf.compat.v1.layers.dense(flatten,4096,activation=tf.nn.tanh)
        
        
        hidden_action       = tf.compat.v1.layers.dense(hidden,2048, activation=tf.nn.elu)
        self.action_logits  = tf.compat.v1.layers.dense(hidden_action,9, activation=tf.nn.softmax)
        self.action_out     = tf.one_hot(tf.random.categorical(logits=self.action_logits,num_samples=1), 9,on_value=1.0, off_value=0.0,axis=-1)
        cross_entropy       = tf.nn.softmax_cross_entropy_with_logits(labels=self.action_out,
                                                                  logits=self.action_logits)
        optimizer             = tf.compat.v1.train.AdamOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        
        self.gradients = [grad for grad, variable in grads_and_vars]
        self.gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in grads_and_vars:
            gradient_placeholder = tf.compat.v1.placeholder(tf.float32, shape=grad.get_shape())
            self.gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))
        self.training_op = optimizer.apply_gradients(grads_and_vars_feed)
        

#testAgent = agentY(0.1,(300,400,1),9,11)  
class agentU():
    def __init__(self,lr,s_size,a_size,h_size):
        self.state_in = tf.compat.v1.placeholder(shape = [None]+list(s_size),dtype=tf.float32)
        self.v_in      = tf.compat.v1.placeholder(shape = (None,1), dtype = tf.float32)
        normalised_img = tf.image.per_image_standardization(self.state_in)
        conv1 = tf.compat.v1.layers.conv2d(normalised_img,24,5,strides=(4, 4))
        conv2 = tf.compat.v1.layers.conv2d(conv1, 36, 5, strides = (2,2))
        conv3 = tf.compat.v1.layers.conv2d(conv2, 48, 5, strides = (2,2))
        conv4 = tf.compat.v1.layers.conv2d(conv3, 64, 3, strides = (1,1))
        conv5 = tf.compat.v1.layers.conv2d(conv4, 64, 3, strides = (1,1))
        
        flatten        = tf.compat.v1.layers.flatten(conv5,name="flatten")
        concat         = tf.concat([flatten, self.v_in],axis=1)
        hidden2        = tf.compat.v1.layers.dense(concat,100,activation=tf.nn.tanh,name="hidden2")
        hidden3        = tf.compat.v1.layers.dense(hidden2,100,activation=tf.nn.tanh,name="hidden3")
        hidden4        = tf.compat.v1.layers.dense(hidden3,100,activation=tf.nn.tanh,name="hidden4")
        
        self.action_logits  = tf.compat.v1.layers.dense(hidden4,9, activation=tf.nn.relu,name="action_logits")
        #self.action         = tf.compat.v1.layers.dense(hidden4,9, activation=tf.nn.softmax,name="action_softmax")
        self.result         = tf.compat.v1.placeholder(dtype = tf.float32, shape=(None, 9))
        self.loss = tf.compat.v1.losses.mean_squared_error(labels = self.result, predictions = self.action_logits)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
        
class AgentU(tf.keras.Model):
    def __init__(self,lr,s_size):
        super(AgentU,self).__init__()
        #self.state_in = tf.keras.Input(shape=(s_size[0],s_size[1],s_size[2]), dtype=np.float)
        #self.vs_in     = tf.keras.Input(shape = (1), dtype = np.float)
        self.in_shape = s_size
        #self.inputs = [self.state_in, self.vs_in]
        self.conv1 = tf.keras.layers.Conv2D(24,5,(4,4))#(normalised_img)
        self.conv2 = tf.keras.layers.Conv2D(36,5,(2,2))#(conv1)
        self.conv3 = tf.keras.layers.Conv2D(48,3,(2,2))#(conv2)
        self.conv4 = tf.keras.layers.Conv2D(64,3,(1,1))#(conv3)
        self.flatten = tf.keras.layers.Flatten()#(conv4)
        self.hidden1 = tf.keras.layers.Dense(100,tf.nn.tanh)#(concat)
        self.hidden2 = tf.keras.layers.Dense(100,tf.nn.tanh)#(hidden1)
        self.logits = tf.keras.layers.Dense(9,tf.nn.relu)#(hidden2)
        self.action = tf.keras.layers.Dense(9,tf.nn.softmax)#(hidden2)
        self.loss_fnc = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')
        
    def call(self, x,training=False):
        #visual = self.state_in(x[0])
        #speed  = self.vs_in(x[1])
        visual = x[0]
        speed  = x[1]
        visual = tf.image.per_image_standardization(visual)
        visual = self.conv1(visual)
        visual = self.conv2(visual)
        visual = self.conv3(visual)
        visual = self.conv4(visual)
        visual = self.flatten(visual)
        visual_with_speed = tf.keras.layers.concatenate([visual,speed],axis=1)
        visual_with_speed = self.hidden1(visual_with_speed)
        visual_with_speed = self.hidden2(visual_with_speed)
        logits = self.logits(visual_with_speed)
#        action = self.action(visual_with_speed)
        return logits
    
# =============================================================================
#     def train_step(self, vision, speed, action):
#         with tf.GradientTape() as tape:
#             logits, action = self.call([vision, speed])
#             print(action)
#             reward = 100
#             r_tensor = tf.Variable(reward,dtype=np.float) 
#             rewarded_action = tf.math.multiply(r_tensor, action)    
#     
#             self.loss = self.loss_object(rewarded_action, logits)
#             gradients = tape.gradient(self.loss, self.trainable_variables)
#             self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#             self.train_loss(self.loss)
#             self.train_accuracy(action*reward, logits)
# =============================================================================
        
        
# =============================================================================
# testAgent = AgentU(0.1,(reshape_size[0],reshape_size[1],1))
# testAgent.compile(optimizer=testAgent.optimizer,loss=testAgent.loss_fnc,metrics=['accuracy'])
# state = None
# speed = None
# for i in range(10):
#     state = get_state()
#     state=state.reshape(-1,reshape_size[0],reshape_size[1],1)
#     speed = 100
#     raw_action = testAgent([np.array(state,dtype=np.float), np.array(speed,dtype=np.float).reshape(-1,1)])
#     action = tf.nn.softmax(raw_action).numpy()
#     testAgent.fit([np.array(state,dtype=np.float), np.array(speed,dtype=np.float).reshape(-1,1)],np.array(action,dtype=np.float).reshape(-1,9),batch_size=1)
# 
# path = 'D:\Grand Theft Auto V\scripts\DummyModel'
# testAgent.save(path)
# =============================================================================
testAgent=tf.keras.models.load_model(path)

class AgentX(tf.keras.Model):
    def __init__(self,lr,s_size,history_size):
        super(AgentX,self).__init__()
        self.states_in = tf.keras.Input(shape=(history_size,s_size[0],s_size[1],s_size[2]), dtype=np.float)
        self.vs_in     = tf.keras.Input(shape = (history_size, 1), dtype = np.float)
        concat = tf.keras.layers.concatenate([self.states_in,self.vs_in],axis=1)
        convlstm = tf.keras.layers.ConvLSTM2D(filters=36,kernel_size=5,strides=(4,4))(self.states_in)
        flatten = tf.keras.layers.Flatten()(convlstm)
    
class lesson2agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network.
        #The agent takes a state and produces an action.
        self.state_in= tf.compat.v1.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        #print("Output:",self.output)
        self.chosen_action = tf.argmax(input=self.output,axis=1)

        #The next six lines establish the training proceedure.
        #We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.compat.v1.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.compat.v1.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(input=self.output)[0]) * tf.shape(input=self.output)[1] + self.action_holder
        #print("output[0]:",self.output[0])
        #print("output[1]:",self.output[1])
        #print("action_holder:",self.action_holder)
        #print("indexes:",self.indexes)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        #print("responsible outputs:",self.responsible_outputs)
        #print("reward holder:",self.reward_holder)
        self.loss = -tf.reduce_mean(input_tensor=tf.math.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.compat.v1.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.compat.v1.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(ys=self.loss,xsds=tvars)
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
#myAgent = lesson2agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.

class lesson2agent2():
    def __init__(self, lr, s_size,a_size,h_size):
        self.state_in= tf.compat.v1.placeholder(shape=[None]+list(s_size),dtype=tf.float32)
        conv1 = slim.conv2d(self.state_in,num_outputs = 16, kernel_size = [4,4])
        max_pool1 = slim.max_pool2d(conv1,[2, 2])
        hidden = slim.fully_connected(max_pool1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        flatten = slim.flatten(hidden)
        self.output = slim.fully_connected(flatten,num_outputs=a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        #print("Output:",self.output)
        self.chosen_action = tf.argmax(input=self.output,axis=1)

        #The next six lines establish the training proceedure.
        #We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.compat.v1.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.compat.v1.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(input=self.output)[0]) * tf.shape(input=self.output)[1] + self.action_holder
        #print("output[0]:",self.output[0])
        #print("output[1]:",self.output[1])
        #print("action_holder:",self.action_holder)
        #print("indexes:",self.indexes)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        #print("responsible outputs:",self.responsible_outputs)
        #print("reward holder:",self.reward_holder)
        self.loss = -tf.reduce_mean(input_tensor=tf.math.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.compat.v1.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.compat.v1.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(ys=self.loss,xs=tvars)
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
