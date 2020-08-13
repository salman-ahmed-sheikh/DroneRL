import random
import numpy as np
import pickle

def loadAgent(name):
    agent = PolicyAgent()
    agent.load_model(name)
    return agent
class PolicyAgent:
    def __init__(self, learning_rate = 0.2, epsilon = 0.9, epsilon_decay = 0.99, exp_rate = 0.3, batch_size=128, decay_gamma = 0.9, memory_size = 500, buffer_size = 10000, drone = None):
        # set default parameters
        
        self.states = []  # record all positions taken
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.exp_rate = exp_rate
        self.decay_gamma = decay_gamma
        self.states_value = {}  # state -> value
        self.drone = drone
        self.batch_size = batch_size
        self.memory = [] # initialize memeory
        self.buffer_size = buffer_size
        self.min_memory_size = memory_size # maximum memory size
    
    # convert state into location of dictionary index
    def getHash(self, state):
        
        ret = []
        for i in state:
            ret.append((i[0]*25) + i[1])
        return ret
    # apply action to state and return the state location
    def applyAction(self, state, action):
        ret = []
        for s in state:
            a1 = max(s[0] + action[0],0)
            a2 = max(s[1] + action[1],0)                        
            ret.append((a1,a2))
        return self.getHash(ret)

    # get the current state and apply action based on learned policy
    def make_move(self, env, current_state):
        options = []
        # either we need to explore the environment by taking random action
        #  or we will exploit it with previously learned results 
        # 
        if random.random() < self.epsilon:
            # Exploration
            return self.drone.make_move(env.objects, env.actions)
        else:
            
            # Exploitation
            value_max = -999
            state_hash = self.getHash(current_state)  
            # checking all the actions which can produce best result          
            for a in env.actions:
                next_state = self.applyAction(current_state, a)                
                next_state = [str(st) for st in next_state]
                next_state = ",".join(next_state)
                value = 0 if self.states_value.get(next_state) is None else self.states_value.get(next_state)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = a
            # update the location of drone
            self.drone.x += action[0]
            self.drone.y += action[1]
            # handle boundry cases
            if self.drone.x < 0:
                self.drone.x = 0
            elif self.drone.y < 0:
                self.drone.y = 0

            return action[0], action[1]
    
    def save_model(self, name):
        sv = self.__dict__
        with open(name, 'wb') as f:
            pickle.dump(sv, f)

    def load_model(self, name):
        with open(name, 'rb') as f:
            dt = pickle.load(f)
        self.states = dt['states']
        self.lr = dt['lr']
        self.epsilon = dt['epsilon']
        self.epsilon_decay = dt['epsilon_decay']
        self.exp_rate = dt['exp_rate']
        self.decay_gamma = dt['decay_gamma']
        self.states_value = dt['states_value'] # state -> value
        self.drone = dt['drone']
        self.batch_size = dt['batch_size']
        self.memory = dt['memory'] # initialize memeory
        self.buffer_size = dt['buffer_size']
        self.min_memory_size = dt['min_memory_size'] # maximum memory size        

    def store_experience(self, experience):
        if len(self.memory) < self.buffer_size:
            self.memory.append(experience)

    def get_batch(self):
        return random.sample(self.memory, self.batch_size)
    
    #update states
    def update(self, reward):
        mini_batch = self.get_batch()
        for state, action, reward, next_state, done in mini_batch:
            state = self.getHash(state) 
            state = [str(s) for s in state]
            state = ",".join(state)
            if self.states_value.get(state) is None:                
                self.states_value[state] = 0
            self.states_value[state] += self.lr * (self.decay_gamma * reward - self.states_value[state])

    def addState(self, state):
        self.states.append(state)
    # update epsilon
    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay