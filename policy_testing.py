import sys
import glob
from matplotlib import pyplot as plt

from src.Environment import Environment
from src.agents.policyAgent import PolicyAgent, loadAgent

# env constants
NUMBER_OF_DRONES = 1
NUMBER_OF_HUMANS = 1
NUMBER_OF_EPOCHS = 50
TOTAL_EPOCHS = 150
MAX_STEPS = 100

def testing_loop():
    
    agents = []
    for file in glob.glob("saved_agents\\policy_*.h5"):
        agents.append(file)
    # create envoirment
    environment = Environment(NUMBER_OF_DRONES, NUMBER_OF_HUMANS)
    for agent in agents:
        print("for agent: " + agent)
        #start envoirment
        state = environment.reset()
        # load load agent by file name 
        agent = loadAgent(agent)
        agent.drone = environment.drones[0]
        for step in range(MAX_STEPS):
            #make move is updating the drone location
            action = agent.make_move(environment, state)
            # rendering the envoirment in new location
            environment.render()
            state_, reward, done = environment.get_info(agent.drone)
            #self.agent.store_experience([state, action, reward, state_, done])
            state = state_
            if done:
                print("reached to goal state")
                break

if __name__ == '__main__':
    testing_loop()
