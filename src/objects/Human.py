import copy
from random import sample

import pygame
from pygame.rect import Rect


class Human(Rect):

    def __init__(self, pos_x: int, pos_y: int, width: int = 1, height: int = 1):
        super().__init__(pos_x, pos_y, width, height)
        self.texture = pygame.image.load('assets\\human.png')
        self.texture = pygame.transform.scale(self.texture, (width, height))

    def make_move(self, state, actions):
        """
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
        """
        return
