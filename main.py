
from pygame.surfarray import array2d
from game import  Game
import pygame
import numpy as np
size=(320,240)

class ManualAgent():
    def next_move(self,game):
        for event in pygame.event.get():
            return game.process_input(event)
        
class RandomAgent():
    def next_move(self,game):
        
        return np.random.randint(1,5)      


#TODO game needs method give_feedback 
def give_feedback(grow,game_over):
    if game_over:
        return -1
    if grow :
        return 10 
class DQNNetwork():
    def __init__(self) -> None:
        #torch.nn.Conv2D
        pass

class DQNAgent():
    def __init__(self, model) -> None:
        self.model=model
    def next_move(self,game):
       x = np.zeros((320,240))

        
game= Game(size,agent=DQNAgent())
game.run()