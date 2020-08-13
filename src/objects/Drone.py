import abc
import copy
import math
from random import sample
import os.path as path
import pygame
from pygame.rect import Rect


class Drone(Rect):
    sum_distance = 0
    distance_history = []

    def __init__(self, pos_x: int, pos_y: int, width: int = 1, height: int = 1):
        super().__init__(pos_x, pos_y, width, height)

        self.texture = pygame.image.load('assets\\drone.png')
        self.texture = pygame.transform.scale(self.texture, (width, height))

    # making random moves, default move for drone
    @abc.abstractmethod
    def make_move(self, state, actions) -> tuple:
        action = sample(actions, 1)[0]

        self.x += action[0]
        self.y += action[1]

        objects_copy = copy.deepcopy(state)
        try:
            objects_copy.remove((self.x, self.y, self.width, self.height))
        except ValueError:
            pass

        if self.collidelist(objects_copy) != -1:
            self.make_move(state, actions)

        return action[0], action[1]

    # calculating sum distances to all humans
    def calculate_distance(self, humans: list) -> float:
        self.sum_distance = 0

        for human in humans:
            self.sum_distance += math.hypot(human.x - self.x, human.y - self.y)

        self.sum_distance = round(self.sum_distance, 2)
        self.distance_history.append(self.sum_distance)

        return self.sum_distance
