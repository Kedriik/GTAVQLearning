import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os
import tensorflow as tf
from tensorflow.contrib import slim
tf.reset_default_graph()
class agentZ():
    def __init__(self, lr, s_size,a_size,h_size):
        self.state_in = tf.placeholder(shape=[None]+list(s_size),dtype=tf.float32)
        conv1         = slim.conv2d(self.state_in,num_outputs = 16, kernel_size = [4,4])
        max_pool1     = slim.max_pool2d(conv1,[2, 2])
        hidden        = slim.fully_connected(max_pool1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.flatten  = slim.flatten(hidden)
        self.logits   = slim.fully_connected(self.flatten,num_outputs=a_size,activation_fn=tf.nn.sigmoid,biases_initializer=None)
        self.action   = tf.multinomial(tf.log(self.logits),num_samples = 1)

class agentY():
    def __init__(self,lr,s_size,a_size,h_size):
        self.state_in = tf.placeholder(shape = [None]+list(s_size),dtype=tf.float32)
        conv1         = tf.layers.conv2d(self.state_in,32,4,strides=(4, 4))
        max_pool1     = tf.layers.max_pooling2d(conv1,32,4)
        flatten       = tf.layers.flatten(max_pool1)
        hidden        = tf.layers.dense(flatten,4096,activation=tf.nn.tanh)
        
        
        hidden_action       = tf.layers.dense(hidden,2048, activation=tf.nn.elu)
        self.action_logits  = tf.layers.dense(hidden_action,9, activation=tf.nn.softmax)
        self.action_out     = tf.one_hot(tf.multinomial(self.action_logits,1), 9,on_value=1.0, off_value=0.0,axis=-1)
        cross_entropy       = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.action_out,
                                                                  logits=self.action_logits)
        optimizer             = tf.train.AdamOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        
        self.gradients = [grad for grad, variable in grads_and_vars]
        self.gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in grads_and_vars:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            self.gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))
        self.training_op = optimizer.apply_gradients(grads_and_vars_feed)
        

#testAgent = agentY(0.1,(300,400,1),9,11)  
class agentU():
    def __init__(self,lr,s_size,a_size,h_size):
        self.state_in = tf.placeholder(shape = [None]+list(s_size),dtype=tf.float32)
        normalised_img = tf.image.per_image_standardization(self.state_in)
        conv1 = tf.layers.conv2d(normalised_img,24,5,strides=(4, 4))
        conv2 = tf.layers.conv2d(conv1, 36, 5, strides = (2,2))
        conv3 = tf.layers.conv2d(conv2, 48, 5, strides = (2,2))
        conv4 = tf.layers.conv2d(conv3, 64, 3, strides = (1,1))
        conv5 = tf.layers.conv2d(conv4, 64, 3, strides = (1,1))
        flatten       = tf.layers.flatten(conv5,name="flatten")
        hidden1        = tf.layers.dense(flatten,1164,activation=tf.nn.tanh,name="hidden1")
        hidden2        = tf.layers.dense(hidden1,100,activation=tf.nn.tanh,name="hidden2")
        hidden3        = tf.layers.dense(hidden2,50,activation=tf.nn.tanh,name="hidden3")
        hidden4        = tf.layers.dense(hidden3,10,activation=tf.nn.tanh,name="hidden4")
        
        self.action_logits  = tf.layers.dense(hidden4,9, activation=tf.nn.softmax)
        self.reward_in      = tf.placeholder(shape = [1], dtype = tf.float32)
        self.optimizer = tf.train.AdamOptimizer(lr)
        #self.loss = tf.losses.mean_squared_error(self.action_in, tf.log(self.action_logits))
        #self.minimize = self.optimizer.minimize(self.loss)
        
                
      

testAgent = agentU(0.1,(300,400,1),9,11)  
     
class lesson2agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network.
        #The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        #print("Output:",self.output)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure.
        #We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        #print("output[0]:",self.output[0])
        #print("output[1]:",self.output[1])
        #print("action_holder:",self.action_holder)
        #print("indexes:",self.indexes)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        #print("responsible outputs:",self.responsible_outputs)
        #print("reward holder:",self.reward_holder)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
myAgent = lesson2agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.

class lesson2agent2():
    def __init__(self, lr, s_size,a_size,h_size):
        self.state_in= tf.placeholder(shape=[None]+list(s_size),dtype=tf.float32)
        conv1 = slim.conv2d(self.state_in,num_outputs = 16, kernel_size = [4,4])
        max_pool1 = slim.max_pool2d(conv1,[2, 2])
        hidden = slim.fully_connected(max_pool1,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        flatten = slim.flatten(hidden)
        self.output = slim.fully_connected(flatten,num_outputs=a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        #print("Output:",self.output)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure.
        #We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        #print("output[0]:",self.output[0])
        #print("output[1]:",self.output[1])
        #print("action_holder:",self.action_holder)
        #print("indexes:",self.indexes)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        #print("responsible outputs:",self.responsible_outputs)
        #print("reward holder:",self.reward_holder)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
# =============================================================================
# def keysToCategories(keys):
#     if keys == [0,0,0,0]: #roll
#         return [1,0,0,0,0,0,0,0,0]
#     if keys == [1,0,0,0]: #roll-right
#         return [0,1,0,0,0,0,0,0,0]
#     if keys == [0,0,1,0]: #roll-left
#         return [0,0,1,0,0,0,0,0,0]
#     if keys == [0,1,0,0]: #acc
#         return [0,0,0,1,0,0,0,0,0]
#     if keys == [1,1,0,0]: #acc-right
#         return [0,0,0,0,1,0,0,0,0]
#     if keys == [0,1,1,0]: #acc-left
#         return [0,0,0,0,0,1,0,0,0]
#     if keys == [1,0,0,1]: #brake-right
#         return [0,0,0,0,0,0,1,0,0]
#     if keys == [0,0,1,1]: #brake-left
#         return [0,0,0,0,0,0,0,1,0]
#     if keys == [0,0,0,1]: #brake
#         return [0,0,0,0,0,0,0,0,1]
#     print("WARNING NON FOUND!")
#     
# resized = [400,300]
# array = None
# while True:
#     screen = grab_screen(region=(0,40,800,640))
#     screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen,(resized[0],resized[1]))
#     if array is None or array.shape[0] < 60:
#         if  array is None:
#             array = screen.reshape(1,resized[1],resized[0])
#         else:
#             array = np.append(array, screen.reshape(1,resized[1],resized[0]),0)
#     else:
#         array = np.delete(array,0,0)
#         array = np.append(array, screen.reshape(1,resized[1],resized[0]),0)
#         
#     cv2.imshow('',array[0])
#     if cv2.waitKey(25) & 0xFF==ord('q'):
#         cv2.destroyAllWindows()
#         break
#    
# =============================================================================
# =============================================================================
# while True:
#     screen = grab_screen(region=(0,40,800,640))
#     screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#     screen = cv2.resize(screen, (400,300))
#     array = screen.reshape(300,400,1)
#         
#     cv2.imshow('',array[0])
#     if cv2.waitKey(25) & 0xFF==ord('q'):
#         cv2.destroyAllWindows()
#         break
# =============================================================================
        
# =============================================================================
# for i in range(60):
#     #screen = grab_screen(region=(0,40,800,640))
#     #screen=cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('',array[i])
#     if cv2.waitKey(25) & 0xFF==ord('q'):
#         cv2.destroyAllWindows()
#         break
# =============================================================================

#cv2.destroyAllWindows()