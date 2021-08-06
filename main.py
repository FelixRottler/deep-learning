
from pygame.surfarray import array2d
from game import  Game
import pygame
import numpy as np

size=(320,240)

class ManualAgent():
    def next_move(self, game):
        for event in pygame.event.get():
            return self.process_input(event)
    
    def process_input(self,event):
        if event.type == pygame.KEYDOWN:
            #up=1
            #left=2
            #right=3
            #down =4   
            if event.key == pygame.K_w and self.snake.direction != 4:     
                return 1
            elif event.key == pygame.K_a and self.snake.direction != 3:     
                return 2
            elif event.key == pygame.K_d and self.snake.direction != 2:    
                return 3
            elif event.key == pygame.K_s and self.snake.direction != 1:     
                return 4
    def receive_feedback(grow, game_over):
        pass 
        
class RandomAgent():
    def next_move(self, game):
        return np.random.randint(1,5)
    def receive_feedback(grow, game_over):
        pass 

class DQNNetwork():
    def __init__(self) -> None:
        #torch.nn.Conv2D
        pass

    #TODO game needs method give_feedback 
    def receive_feedback(grow, game_over):
        if game_over:
            return -1
        if grow:
            return 10 

class DQNAgent():
    def __init__(self, model) -> None:
        self.model=model
    def next_move(self, game):
       x = np.zeros((320,240))

        
game=Game(size, agent=ManualAgent())
game.run()
