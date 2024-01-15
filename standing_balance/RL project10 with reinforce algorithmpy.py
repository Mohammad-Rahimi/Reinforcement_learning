import sys
osim_path = 'C:\OpenSim 4.1\sdk\Python'
sys.path.insert(0,osim_path)
import opensim as osim
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import gym
from gym import spaces

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'balance'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 5
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

class Balance:
    def __init__(self):
        
        a = osim.Model("standingperturbation_Prescribed.osim")
        self.myModel=a

    
    
    
    OBSERVATION_SPACE_VALUES=6
    
    
    
    ACTION_SPACE_SIZE = 9
    def reset(self):
        
        
        self.myModel = osim.Model("standingperturbation_Prescribed.osim")
        self.myModel.setUseVisualizer(True)
        state=[]
        for i in range(6):
            state.append(0)
        self.initial_state = self.myModel.initSystem()
        
        self.manager = osim.Manager(self.myModel)
        self.initial_state.setTime( 0 )
        self.manager.initialize(self.initial_state)
        
        
        state[0]=self.myModel.getStateVariableValue(self.initial_state,'jointset/ankle_r/ankle_angle_r/value')  #ankle
        state[1]=self.myModel.getStateVariableValue(self.initial_state,'jointset/knee_r/knee_angle_r/value') #knee angle
        #initial_state.updQ()[3]=4 #knee_angle_r_beta
        state[2]=self.myModel.getStateVariableValue(self.initial_state,'jointset/hip_r/hip_flexion_r/value') #hip flexion
        #initial_state.updU()[0]=11 #platform_tx
        state[3]=self.myModel.getStateVariableValue(self.initial_state,'jointset/ankle_r/ankle_angle_r/speed') #ankle speed
        state[4]=self.myModel.getStateVariableValue(self.initial_state,'jointset/knee_r/knee_angle_r/speed') #knee speed
        #initial_state.updU()[3]=44 #knee_angle_r_beta
        state[5]=self.myModel.getStateVariableValue(self.initial_state,'jointset/hip_r/hip_flexion_r/speed')
        
        observation=state
        return observation
    def step(self, action):
        state_next=[]
        for i in range(6):
            state_next.append(0)
        self.episode_step += 1
        muscleList=self.myModel.updMuscles()
        action=np.clip(action,0,1)
        for j in range(9):
            muscleList.get(j).setActivation(self.initial_state,action[j])
        self.initial_state = self.manager.integrate( (step) *0.005 )
        self.myModel.realizePosition(self.initial_state)
        
        state_next[0]=self.myModel.getStateVariableValue(self.initial_state,'jointset/ankle_r/ankle_angle_r/value')  #ankle
        state_next[1]=self.myModel.getStateVariableValue(self.initial_state,'jointset/knee_r/knee_angle_r/value') #knee angle
        #initial_state.updQ()[3]=4 #knee_angle_r_beta
        state_next[2]=self.myModel.getStateVariableValue(self.initial_state,'jointset/hip_r/hip_flexion_r/value') #hip flexion
        #initial_state.updU()[0]=11 #platform_tx
        state_next[3]=self.myModel.getStateVariableValue(self.initial_state,'jointset/ankle_r/ankle_angle_r/speed') #ankle speed
        state_next[4]=self.myModel.getStateVariableValue(self.initial_state,'jointset/knee_r/knee_angle_r/speed') #knee speed
        #initial_state.updU()[3]=44 #knee_angle_r_beta
        state_next[5]=self.myModel.getStateVariableValue(self.initial_state,'jointset/hip_r/hip_flexion_r/speed')
        new_observation = state_next
        
        y=self.myModel.calcMassCenterPosition(self.initial_state)
        reward=-1*abs(y[0]-self.initial_state.getQ()[0])
        done= False
        if step== 400 or reward==-0.4:
            done=True
        return new_observation, reward, done
env=Balance()
ep_rewards = [-200]
                  
        
print("action space=",env.ACTION_SPACE_SIZE)  
print("observation space=",env.OBSERVATION_SPACE_VALUES)     
        
    

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
        
class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    
    def create_model(self):
        model = Sequential()
        
        model.add(Dense(24, input_shape=6, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(9, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        action=[]
        muscleList=9
        for muscle in muscleList:
            action.append(0)
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.rand(env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)
        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        
        
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
        # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        
        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
