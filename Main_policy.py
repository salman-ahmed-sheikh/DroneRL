import sys

from matplotlib import pyplot as plt

from src.Environment import Environment
from src.agents.policyAgent import PolicyAgent, loadAgent


# env constants
NUMBER_OF_DRONES = 1
NUMBER_OF_HUMANS = 1
NUMBER_OF_EPOCHS = 50
TOTAL_EPOCHS = 300
MAX_STEPS = 100


class Main:
    train_results = "Train results 11042020F"

    def __init__(self):
        self.environment = Environment(NUMBER_OF_DRONES, NUMBER_OF_HUMANS)
        self.agent = PolicyAgent() # create agent with default parameters
        #self.agent.load_model("saved_agents\\policy_agent.h5")

    # generating plots
    def generate_plot(self, data: list):
        fig, ax = plt.subplots()
        ax.plot(data, 'r', label=f"{self.train_results}")
        ax.set(xlabel="Episode", ylabel="Distance", title="")
        ax.grid()
        plt.legend(loc='upper left')
        plt.draw()
        fig.savefig(f'results\\{self.train_results}.png', dpi=1200)

    # testing phase->
    def training_loop(self):
        count = 0
        # iterate on epochs
        for episode in range(TOTAL_EPOCHS):
            print("Epoch # " + str(episode+1))
            # reinitialize the envoirment
            state = self.environment.reset()
            # set agent drone to update location
            self.agent.drone = self.environment.drones[0]
            # maximum number of steps to move
            for step in range(MAX_STEPS):
                # agent get the state and decide move, more detail in available in the function  
                action = self.agent.make_move(self.environment, state)
                # make move function has updated the drone location now render the env 
                self.environment.render()
                # check for the reward, get next location and if we reached to goal state 
                state_, reward, done = self.environment.get_info(self.agent.drone)
                self.agent.addState(state)
                self.agent.store_experience([state, action, reward, state_, done])
                state = state_
                # if took specified number of steps it will update policy states
                if len(self.agent.memory) > self.agent.min_memory_size:
                    self.agent.update(self.environment)
                if done:
                    break
            # update learning rate
            self.agent.update_epsilon()
            # after every iteration store the model
            if (episode + 1)%(NUMBER_OF_EPOCHS) == 0:
                print("saving model...")
                self.agent.save_model("saved_agents\\policy_agent_{}.h5".format(count))
                count += 1
        # turn off the env
        self.environment.close()
        # self.testing_loop()

    

if __name__ == '__main__':
    Main().training_loop()
