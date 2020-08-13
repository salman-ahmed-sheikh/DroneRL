from src.objects.Drone import Drone


class RandomAgent(Drone):

    def __init__(self, pos_x: int, pos_y: int, width: int = 1, height: int = 1):
        super().__init__(pos_x, pos_y, width, height)
