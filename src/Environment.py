from random import randrange

import pygame

from src.objects.Drone import Drone
from src.objects.Human import Human

# pygame constants
FPS = 30
WIDTH = 25
HEIGHT = 25

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Environment:
    objects = []
    step = 0
    actions = [(0, -5),
               (-5, -5),
               (-5, 0),
               (-5, 5),
               (0, 5),
               (5, 5),
               (5, 0),
               (5, -5),
               (0, 0)]
    drones = []
    humans = []

    def __init__(self, number_of_drones: int, number_of_humans: int):
        pygame.init()
        pygame.display.set_caption("Experiment")
        self.clock = pygame.time.Clock()
        self.main_surface = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont("comicsansms", 12)
        self.number_of_drones = number_of_drones
        self.number_of_humans = number_of_humans
        self.objects_dict = {Drone: number_of_drones, Human: number_of_humans}
        self.clear()

    def get_info(self, drone: Drone) -> tuple:
        positions = []
        done = False
        reward = 0

        if self.step < 200:
            self.step += 1
        else:
            self.step = 0
            done = True

        for obj in self.objects:
            positions.append((obj.x, obj.y))

        old_dist = drone.sum_distance
        new_dist = drone.calculate_distance(self.humans)

        if old_dist < new_dist:
            reward = -1
        elif old_dist > new_dist:
            reward = 1
        elif old_dist == new_dist:
            reward = -1

        return positions, reward, done

    # put object to the env
    def spawn_objects(self, objects_dict: dict):
        objects_dict_copy = objects_dict.copy()

        if all(value == 0 for value in objects_dict_copy.values()):
            for obj in self.objects:
                self.main_surface.blit(obj.texture, obj)
            return

        for key in objects_dict_copy.keys():
            if objects_dict_copy[key] > 0:
                obj = None

                if key == Drone:
                    obj = Drone(
                        pos_x=randrange(0, self.main_surface.get_width() - 1),
                        pos_y=randrange(0, self.main_surface.get_height() - 1))
                    self.drones.append(obj)
                elif key == Human:
                    obj = Human(randrange(0, self.main_surface.get_width() - 1),
                                randrange(0, self.main_surface.get_height() - 1))
                    self.humans.append(obj)

                if obj.collidelist(self.objects) == -1:
                    self.objects.append(obj)
                    objects_dict_copy[key] -= 1
            else:
                continue

        self.spawn_objects(objects_dict_copy)

    # clear surface
    def clear(self):
        self.main_surface.fill(WHITE)

    def close(self):
        pygame.display.quit()

    # resetting env to init state
    def reset(self) -> list:
        positions = []

        self.clear()
        self.drones.clear()
        self.humans.clear()
        self.objects.clear()
        self.spawn_objects(self.objects_dict)

        for obj in self.objects:
            positions.append((obj.x, obj.y))

        return positions

    # render movements and calculate sum for drones
    def render(self):
        pygame.event.get()
        self.clear()

        for obj in self.objects:
            if type(obj) is Human:
                obj.make_move(self.objects, self.actions)

            obj.clamp_ip(self.main_surface.get_rect())
            self.main_surface.blit(obj.texture, obj)

        # text = self.font.render(f"Sum distance: {self.drones[0].sum_distance}", True, BLACK)
        # self.main_surface.blit(text, (5, 5))

        # pygame.time.delay(500)
        pygame.display.update()
        self.clock.tick(FPS)
