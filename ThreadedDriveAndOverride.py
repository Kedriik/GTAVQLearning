import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W,A,S,D,T
from random import random
import tensorflow as tf
#import tensorflow.contrib.slim as slim
from playground import AgentU
from threading import Thread
from threading import Lock
from time import sleep
from math import sqrt, exp
import win32file
import struct
from numpy_ringbuffer import RingBuffer
capture_region = (0,40,1280,720)
reshape_size = (int(720/2),int(1280/2))

AGENT_NAME = 'TestAgent'
g_debug_vars = []
def keysToCategories(keys):
    if keys == [0,0,0,0]: #rollx
        return [1,0,0,0,0,0,0,0,0]
    if keys == [1,0,0,0]: #roll-right
        return [0,1,0,0,0,0,0,0,0]
    if keys == [0,0,1,0]: #roll-left
        return [0,0,1,0,0,0,0,0,0]
    if keys == [0,1,0,0]: #acc
        return [0,0,0,1,0,0,0,0,0]
    if keys == [1,1,0,0]: #acc-right
        return [0,0,0,0,1,0,0,0,0]
    if keys == [0,1,1,0]: #acc-left
        return [0,0,0,0,0,1,0,0,0]
    if keys == [1,0,0,1]: #brake-right
        return [0,0,0,0,0,0,1,0,0]
    if keys == [0,0,1,1]: #brake-left
        return [0,0,0,0,0,0,0,1,0]
    if keys == [0,0,0,1]: #brake
        return [0,0,0,0,0,0,0,0,1]
    print("K to C: WARNING NON FOUND!")
    return [1,0,0,0,0,0,0,0,0]
    
def categoriesToKeys(categories):
    
    if categories == [1,0,0,0,0,0,0,0,0]: #rollzz
        return [0,0,0,0]
    if categories == [0,1,0,0,0,0,0,0,0]: #roll-right
        return [1,0,0,0]
    if categories == [0,0,1,0,0,0,0,0,0]: #roll-left
        return [0,0,1,0]
    if categories == [0,0,0,1,0,0,0,0,0]: #acc
        return [0,1,0,0]
    if categories == [0,0,0,0,1,0,0,0,0]: #acc-right
        return [1,1,0,0]
    if categories == [0,0,0,0,0,1,0,0,0]: #acc-left
        return [0,1,1,0]
    if categories == [0,0,0,0,0,0,1,0,0]: #brake-right
        return [1,0,0,1]
    if categories == [0,0,0,0,0,0,0,1,0]: #brake-left
        return [0,0,1,1]
    if categories == [0,0,0,0,0,0,0,0,1]: #brake
        return [0,0,0,1]    
    print("C to K: WARNING NON FOUND!")
    return [0,0,0,0]
        
def keys_to_output(keys):
    #[A,W,D,S]
    output=[0,0,0,0]    
    if 'A' in keys:
        output[0] = 1
    if 'W' in keys:
        output[1] = 1
    if 'D' in keys:
        output[2] = 1
    if 'S' in keys:
        output[3] = 1        
    return output

def update_pressed_keys(keys):
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    if keys[0] == 1:
        PressKey(A)
    if keys[1] == 1:
        PressKey(W)
    if keys[2] == 1:
        PressKey(D)
    if keys[3] == 1:
        PressKey(S)
        
def releaseKeys():
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)

gamma = 0.99
def show_frames(frames):
    for frame in frames:
        cv2.imshow('',frame)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
        
def _discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

def get_state():
    state = grab_screen(region = capture_region)
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state,(reshape_size[1],reshape_size[0]))
    state = state.reshape(reshape_size[0],reshape_size[1],1)
    return state

def teach_agent(agent, all_rewards, all_gradients,sess):
    rewards = np.array(discount_and_normalize_rewards(all_rewards,0.99))
# =============================================================================
#     test = []
#     feed_dict = {}
# 
#     
#     for var_index, gradient_placeholder in enumerate(agent.gradient_placeholders):
#         mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
#                                   for game_index, rewards in enumerate(all_rewards)
#                                       for step, reward in enumerate(rewards)], axis=0)
#         
#         feed_dict[gradient_placeholder] = mean_gradients
#     ret = sess.run(agent.training_op, feed_dict=feed_dict)  
# =============================================================================
    
def teach_agent_rt(agent, reward, gradients, sess):
    feed_dict = {}
    for var_index, gradient_placeholder in enumerate(agent.gradient_placeholders):
        feed_dict[gradient_placeholder] = gradients[var_index]*reward;
    
    sess.run(agent.training_op, feed_dict=feed_dict)
        
def m_multinomial(acc, steer):
    acc_i = np.random.multinomial(1,acc)
    steer_i = np.random.multinomial(1,steer)
    return acc_i, steer_i
def m_multinomial9(action):
    return np.random.multinomial(1,action)

def normalize(v):
    m = 0
    for i in range(len(v)):
        m = m + v[i]*v[i]
    m = sqrt(m)
    for i in range(len(v)):
        v[i] = v[i] / m
    return v

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 

def softmax1(x, axis=-1):
    # save typing...
    kw = dict(axis=axis, keepdims=True)

    # make every value 0 or below, as exp(0) won't overflow
    xrel = x - x.max(**kw)

    # if you wanted better handling of small exponents, you could do something like this
    # to try and make the values as large as possible without overflowing, The 0.9
    # is a fudge factor to try and ignore rounding errors
    #
    #     xrel += np.log(np.finfo(float).max / x.shape[axis]) * 0.9

    exp_xrel = np.exp(xrel)
    return exp_xrel / exp_xrel.sum(**kw)  

def harvest_input():
    global gathered_categories
    global gathered_velocities
    average_categories = np.array([0,0,0,0,0,0,0,0,0]).astype(float)
    mutex.acquire(1)
    for i in range(len(gathered_categories)):
        average_categories = np.array(average_categories) + np.array(gathered_categories[i]).astype(float)
    
    average_categories = softmax(average_categories)    
    average_v = 0
    for i in range(len(gathered_velocities)):
        average_v = average_v + gathered_velocities[i]
    average_v = average_v/len(gathered_velocities)
    gathered_categories.clear()
    gathered_velocities.clear()
    mutex.release()
    return average_categories, average_v

gathered_categories = []
gathered_velocities = []
mutex = Lock()
gather_flag = True
def gather_input():
    global gathered_categories
    global gathered_velocities
    key_check()
    while(gather_flag):
        keys   = key_check()
        mutex.acquire(1)
        gathered_categories.append(keysToCategories(keys_to_output(keys)))
        
        v = struct.unpack('f',data)[0]
        
        gathered_velocities.append(v)
        mutex.release()
def random_action(a, eps=0.1):
  actions = [0,1,2,3,3,3,3,3,3,4,5,6,7,8]
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(actions)
teach_agent_flag = True
save_flag = False
pause_learning = True
reset_time = 0
HISTORY_SIZE = 30

visual_history = RingBuffer(capacity=HISTORY_SIZE, dtype=np.float)
speed_history = RingBuffer(capacity=HISTORY_SIZE, dtype=np.float)
def teach_agent():
    global reset_time
    #tf.compat.v1.reset_default_graph()
    total_loss=0
    counter=0
    path_to_agent = AGENT_NAME
    agent = AgentU(0.1,(reshape_size[0],reshape_size[1],1))
    agent.compile(optimizer=agent.optimizer,loss=agent.loss_fnc,metrics=['accuracy'])
    q_debug_mode = True
    
    try:
        with open('{}.index'.format(path_to_agent),'r') as fh:
            print('Agent {} exists. Loading.'.format(AGENT_NAME))
            testAgent=tf.keras.models.load_model(path_to_agent)
            print('Loading complete')
    except FileNotFoundError:
        print('Agent {} doesnt exist.Initializing.'.format(AGENT_NAME))
        print('Initialization complete')

    fileHandle = win32file.CreateFile(r'\\.\pipe\VStreaming', 
                                      win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, 
                                      win32file.OPEN_EXISTING, 0, None)
    loop_duration = 0;
    while(teach_agent_flag):
        loop_start = time.time()
        #get state
        state = get_state()
        left, data = win32file.ReadFile(fileHandle, 4)
        speed = struct.unpack('f',data)[0]
        raw_action = agent([np.array(state,dtype=np.float), np.array(speed,dtype=np.float).reshape(-1,1)])
        action = tf.nn.softmax(raw_action).numpy()
        selected_index = np.argmax(action)
        selected_index = random_action(selected_index,action[selected_index])
        a= [0,0,0,0,0,0,0,0,0]
        a[selected_index] = 1
        releaseKeys()
        update_pressed_keys(categoriesToKeys(a))
        #receive reward
        left, data = win32file.ReadFile(fileHandle, 4)
        speed = struct.unpack('f',data)[0]
        #print("Reward V:",v_orig)
        v =speed/1
        #rewarded_action = a*v
       
        if speed < 1:
            reset_time = reset_time + loop_duration
            v = -1
            if reset_time > 10:
                PressKey(T)
                sleep(1)
                ReleaseKey(T)
                sleep(1)
                PressKey(T)
                sleep(1)
                ReleaseKey(T)
                reset_time = 0
        else:
            reset_time = 0
            
        reward = [0,0,0,0,0,0,0,0,0]
        for i in range(9):
            if i == selected_index:
                reward[i] = raw_action[i] + 10*v;
            else:
                reward[i] = raw_action[i] - v;
        reward = np.array(reward).reshape((-1,9))
        agent.fit([np.array(state,dtype=np.float), np.array(speed,dtype=np.float).reshape(-1,1)],reward,batch_size=1)
        
        loop_duration = time.time() - loop_start
        
        del state
        #print("Loop duration:",loop_duration)
        #if(loop_duration < 0.5):
         #   sleep(1 - loop_duration)sa
    cv2.destroyAllWindows()
    if(q_debug_mode==False):                
        agent.save(path_to_agent)
        
agent_plays_flag = True
save_flag = False
action = np.array([0,0,0,0,0,0,0,0,0])
def agent_plays():
    print("Dummy")
# =============================================================================
#     tf.reset_default_graph()
#     path_to_agent = AGENT_NAME
#     agent = agentU(0.1,(reshape_size[0],reshape_size[1],1),9,11)
#     saver = tf.train.Saver()
#     global action
#     with tf.Session() as sess:
#         try:
#             with open('{}.index'.format(path_to_agent),'r') as fh:
#                 print('Agent {} exists. Loading.'.format(AGENT_NAME))
#                 saver.restore(sess,path_to_agent)
#                 print('Loading complete')
#         except FileNotFoundError:
#             print('Agent {} doesnt exist.Initializing.'.format(AGENT_NAME))
#             
#             init = tf.global_variables_initializer()
#             sleep(1)
#             sess.run(init)
#             print('Creation complete')
# #x
#         while(agent_plays_flag):
#             loop_start = time.time()
#             state = get_state()
#             feed_dict_steer = {agent.state_in:[state]}
#             action = np.array(sess.run([agent.action_logits], feed_dict = feed_dict_steer)[0][0])
#             loop_duration = time.time() - loop_start
#            # print("ML loop took:", loop_duration)
#             if(loop_duration < 0.5):
#                 sleep(1 - loop_duration)
# =============================================================================
#x                
execute_inputs_flag  = True
def execute_inputs():
    global action
    while(execute_inputs_flag):
        _action = np.around(action,4)
        _action = np.random.multinomial(1,_action)
        if pause_learning == True:
            continue
        #
        print(_action.tolist())
        update_pressed_keys(categoriesToKeys(_action.tolist()))
        sleep(0.01)
        
        
        
    
def control():
    global teach_agent_flag
    global gather_flag
    global agent_plays_flag
    global execute_inputs_flag
    global pause_learning
    collectData = False
    clickFlag = False
    while True:
        time0 = time.time()
        #-----------------
        keys = []
        keys   = key_check()
        
        if 'Z' in keys and clickFlag == False:
            clickFlag = True
            if pause_learning == True:
                pause_learning = False
                print('Training started')
            else:
                pause_learning = True
                releaseKeys()
                print('Training Paused')
        elif 'Z' not in keys and clickFlag == True:
            clickFlag = False
            
                
        if 'X' in keys:
            teach_agent_flag        = False
            gather_flag             = False
            agent_plays_flag        = False
            execute_inputs_flag     = False
            releaseKeys()
            print ('collecting data stopped')
            break
        
# =============================================================================
#         if 'T' in keys and collectData == False:
#             print ('Saving agent, please wait') #x
#             #saver.save(sess, '/tmp/model{}.ckpt'.format(teach_count))x
#             print('Saved.');
# =============================================================================
   
        time1 = time.time()
        delta_time = time1 - time0
play_flag = False
def main():
    control_thread = Thread(target = control)
    control_thread.start()
    if(play_flag == False):
        #gather_input_thread = Thread(target = gather_input)
        #gather_input_thread.start()
        g = tf.Graph()
        with g.as_default():
            teach_agent()    
        #gather_input_thread.join()
    else:
        execute_input_thread = Thread(target = execute_inputs)
        execute_input_thread.start()
        #agent_plays()
        execute_input_thread.join()
    control_thread.join()
        
if __name__ == '__main__':
    main()