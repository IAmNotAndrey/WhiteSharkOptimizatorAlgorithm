import numpy as np

class WhiteShark:
    def __init__(self, velocity: float, 
                       position : list[float], 
                       search_space : list[tuple[float, float]]
                ):
        self.velocity = velocity
        self.position = position
        self.search_space = search_space
        
    def adjust_position(self):
        for i, cord in enumerate(self.position):
            if not (self.search_space[i][0] <= cord <= self.search_space[i][1]):
                closest_border_idx = np.argmin([
                    np.abs(cord - self.search_space[i][0]), 
                    np.abs(cord - self.search_space[i][1])
                ])
                self.position[i] = self.search_space[i][closest_border_idx]
