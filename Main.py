import argparse
import sys
#import pygame
#import gym

from matplotlib import pyplot as plt

from src.Environment import Environment
from src.agents.DDQNAgent import DDQNAgent
from tensorflow import keras

# env constants
NUMBER_OF_DRONES = 1
NUMBER_OF_HUMANS = 1
NUMBER_OF_EPOCHS = 50
MAX_STEPS = 100


class Main:
    train_results = "Train results 11042020F"

    def __init__(self):
        self.environment = Environment(NUMBER_OF_DRONES, NUMBER_OF_HUMANS)
        self.agent = DDQNAgent(learning_rate=0.001,
                               epsilon=0.9,
                               epsilon_decay=0.99,
                               gamma=0.8,
                               batch_size=64,
                               buffer_size=10000,
                               min_memory_size=500,
                               tau=0.1)
        self.agent.model = self.agent.create_model()
        self.agent.target_model = self.agent.create_model()

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
        for episode in range(NUMBER_OF_EPOCHS):
            print(episode)
            state = self.environment.reset()
            self.agent.drone = self.environment.drones[0]
            for step in range(MAX_STEPS):
                action = self.agent.make_move(self.environment, state)
                self.environment.render()
                state_, reward, done = self.environment.get_info(self.agent.drone)
                self.agent.store_experience([state, action, reward, state_, done])
                state = state_

                if len(self.agent.memory) > self.agent.min_memory_size:
                    self.agent.update(self.environment)
                    self.agent.update_target_weights()

                if done:
                    break

            self.agent.update_epsilon()

        self.generate_plot(self.environment.drones[0].distance_history)
        # self.agent.generate_loss()
        self.agent.save_model()
        self.environment.close()
        # self.testing_loop()

    def testing_loop(self):
        model = keras.models.load_model("saved_agents\\agent.h5")


if __name__ == '__main__':
    Main().training_loop()
