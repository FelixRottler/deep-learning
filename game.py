import torch
from helpers import rand_pos
import pygame

class Snake():
    width=1
    height=1
    length=1
    grow=False
    direction=1
    def __init__(self, screen):
        self.screen=screen
        self.screen_x,self.screen_y = screen.get_size()
        x0,y0 = rand_pos(self.screen_x, self.screen_y)
        self.body = [[x0,y0]]
        
    def draw(self):
        for x,y in self.body:
            pygame.draw.rect(self.screen, (188,45,23), (x,y, 1,1))

    def move(self,direction):
        x,y =self.body[0]
        if   direction==1:
            y-=1
        elif direction==2:
            x-=1
        elif direction==3:
            x+=1
        elif direction==4:
            y+=1

        self.body.insert(0,[x,y]) # insert new body part at head
        
        if not self.grow:
            del self.body[-1] # delete tail if we haven't eaten
        else:
            self.grow=False
        

class Game():
    score=0
    def __init__(self,size,title="Snake"):
        pygame.init()
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("gillsans.ttc", 24)
        self.screen= pygame.display.set_mode(size)
        self.surface = pygame.Surface(size)
        self.clock = pygame.time.Clock()
       
        self.snake=Snake(self.screen)
        self.food=Food(self.screen)

        self.running=False
        self.game_over=False

    def opposite_direction(self,direction):
        if  (self.snake.direction == 4 and direction == 1) or ( self.snake.direction == 1 and direction == 4) or(self.snake.direction == 3 and direction == 2) or (self.snake.direction == 2 and direction == 3):     
            return True
        return False


   
    def execute_action(self,direction):
        self.screen.fill((0,0,0))
        reward = 1.0
        self.handle_game_over()
        
        if direction is not None and not self.opposite_direction(direction):
            self.snake.move(direction)
            self.snake.direction = direction

        self.game_over= self.check_collisions() 
        if self.game_over:
            reward= -1.0
        if (self.check_food()):
            reward = 10.0 
        self.food.draw()
        
        self.snake.draw()
        #self.display_score()

        pygame.display.update()
        pygame.display.flip()
        self.clock.tick(10)
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                pass

        next_state = self.get_state()
        return reward, next_state, self.game_over
        
    def get_state(self):
        return pygame.PixelArray(self.surface)

    def handle_game_over(self):
        if self.game_over:
            self.display_game_over()
            done= False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        self.reset()
                        done = True
    
    def reset(self):
        self.game_over=False
        self.snake= Snake(self.screen)
        self.food = Food(self.screen)
        self.score=0

    def display_game_over(self):
        img = self.font.render("GAME OVER Press any key", True, (180,45,30))
        self.screen.blit(img, (50,50))
        pygame.display.flip()

    def display_score(self):
        img = self.font.render(f'Length {self.score} ', True, (180,180,30))
        self.screen.blit(img, (0,0))
        pygame.display.flip()

    def check_collisions(self):
        x,y =self.snake.body[0]
        x_screen,y_screen= self.screen.get_size()

        if x >= x_screen or x <0 or y > y_screen-1 or y<0:
            return True
        if [x,y] in self.snake.body[1:]:
            return True
        return False

    def check_food(self):
        x,y = self.snake.body[0]
        if x == self.food.x0 and y == self.food.y0:
            self.snake.grow=True
            self.food.respawn()
            self.score+=1
            return True
        return False

class Food():
    def __init__(self, screen):
        self.screen=screen
        self.screen_x,self.screen_y = screen.get_size()
        self.x0,self.y0 = rand_pos(self.screen_x, self.screen_y)

    def draw(self):
        pygame.draw.rect(self.screen, (188,180,23), (self.x0,self.y0, 1,1))
    def respawn(self):
        self.x0,self.y0 = rand_pos(self.screen_x, self.screen_y)