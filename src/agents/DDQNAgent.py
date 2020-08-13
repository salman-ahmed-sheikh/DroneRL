import random
import numpy as np

from matplotlib import pyplot as plt
from numpy import array
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam

from src.Environment import Environment


class DDQNAgent:
    memory = list()

    def __init__(self, learning_rate, epsilon, epsilon_decay, gamma, tau, batch_size, buffer_size, min_memory_size,
                 drone=None):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_memory_size = min_memory_size
        self.model = None
        self.target_model = None
        self.drone = drone

    # create DQN model
    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(64, input_dim=4, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(9, activation='linear'))
        # model.compile(loss="mse", optimizer=Adam(self.learning_rate))
        model.compile(loss="mse", optimizer='adam')

        return model

    # generating loss
    def generate_loss(self):
        fig, ax = plt.subplots()

        # Get training and test loss histories
        training_loss = self.create_model().history['loss']
        test_loss = self.create_model().history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history
        plt.plot(epoch_count, training_loss, 'r--')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.draw()
        fig.savefig(f'results\\LossFunction.png', dpi=1200)

    # storing agent experience replay memory
    def store_experience(self, experience: list):
        if len(self.memory) < self.buffer_size:
            self.memory.append(experience)

    # get one batch of given size
    def get_batch(self) -> list:
        return random.sample(self.memory, self.batch_size)

    def save_model(self):
        self.model.save("saved_agents\\agent.h5")

    def load_model(self):
        try:
            self.model.load("saved_agents\\agent.h5")
        except FileNotFoundError:
            print("File with agent not found!")

    # update epsilon
    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay

    # moving with drone
    def make_move(self, environment: Environment, state: list):
        options = []

        if random.random() < self.epsilon:
            # Exploration
            return self.drone.make_move(environment.objects, environment.actions)
        else:
            # Exploitation
            q_values = self.model.predict(self.preprocess_input(state))

            for action in environment.actions:
                value = q_values[0][environment.actions.index(action)]
                options.append((value, action))

            max_value = max(options)[0]
            best_options = [option for option in options if option[0] == max_value]

            x, y = random.choice(best_options)[1]

            self.drone.x += x
            self.drone.y += y

            if self.drone.x < 0:
                self.drone.x = 0
            elif self.drone.y < 0:
                self.drone.y = 0

            return x, y

    # reshaping input to DQN
    def preprocess_input(self, state: list):

        # normalization
        state = [(item[0] / 800, item[1] / 600) for item in state]

        state = array(state).flatten()

        return np.reshape(state, (1, len(state)))

    # update Q-values from batch
    def update(self, environment: Environment):
        mini_batch = self.get_batch()
        new_q_values = []

        for state, action, reward, next_state, done in mini_batch:
            # calculating Q(s,a)
            q_values_state = self.model.predict(self.preprocess_input(state))[0]

            if done:
                target = reward
            else:
                # choosing max action from state_
                max_action = np.argmax(self.model.predict(self.preprocess_input(next_state)))
                # calculating target and Q(s_,a)
                target = reward + self.gamma * self.target_model.predict(self.preprocess_input(next_state))[0][
                    max_action]

            # updated Q values for specific action
            q_values_state[environment.actions.index(action)] = target
            new_q_values.append(q_values_state)

        input_states = np.array([self.preprocess_input(sample[0])[0] for sample in mini_batch])

        self.model.fit(input_states, np.array(new_q_values), verbose=1, epochs=1,
                       batch_size=self.batch_size)

    # copy weights from model to target
    def update_target_weights(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for index, (weight, target_weight) in enumerate(zip(weights, target_weights)):
            target_weight = weight * self.tau + target_weight * (1 - self.tau)
            target_weights[index] = target_weight

        self.target_model.set_weights(target_weights)
