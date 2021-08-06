from helpers import rand_pos
import pygame


class Snake():
    width=10
    height=10
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
            pygame.draw.rect(self.screen, (188,45,23), (x,y, 10,10))
    def move(self,direction):
        x,y =self.body[0]
        if direction == 1:
            y-=10
        elif direction==2:
            x-=10
        elif direction==3:
            x+=10
        elif direction==4:
            y+=10

        self.body.insert(0,[x,y]) # insert new body part at head
        
        if not self.grow:
            del self.body[-1] # delete tail if we haven't eaten
        else:
            self.grow=False
      
        

class Game():
    score=0
    def __init__(self,size,agent,title="Snake"):
        pygame.init()
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("gillsans.ttc", 24)
        self.screen= pygame.display.set_mode(size)
        self.surface = pygame.Surface(size)
        self.clock = pygame.time.Clock()
       
        self.agent= agent
        self.snake=Snake(self.screen)
        self.food=Food(self.screen)

        self.running=False
        self.game_over=False
    def opposite_direction(self,direction):
        if  (self.snake.direction == 4 and direction == 1) or ( self.snake.direction == 1 and direction == 4) or(self.snake.direction == 3 and direction == 2) or (self.snake.direction == 2 and direction == 3):     
            return True
        return False

    def run(self):
        self.running = True
        
        while self.running:
            self.handle_game_over()
            
            direction = self.agent.next_move(self)
            if direction is not None and not self.opposite_direction(direction):
                self.snake.move(direction)
                self.snake.direction = direction

            self.game_over= self.check_collisions()  
            self.check_food() # snake.grow
            self.screen.fill((0,0,0)) 
            self.food.draw()
            
            self.snake.draw()
            self.display_score()

            pygame.display.update()
            self.clock.tick(30)
        pygame.quit()

    def handle_game_over(self):
        if self.game_over:
            self.display_game_over()
            done= False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        done = True
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
        
    def process_input(self,event):
        if event.type == pygame.KEYDOWN:
            #up=1
            # left=2
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

    def check_collisions(self):
        x,y =self.snake.body[0]
        x_screen,y_screen= self.screen.get_size()

        if x >= x_screen or x <0 or y > y_screen-10 or y<0:
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

class Food():
    def __init__(self, screen):
        self.screen=screen
        self.screen_x,self.screen_y = screen.get_size()
        self.x0,self.y0 = rand_pos(self.screen_x, self.screen_y)

    def draw(self):
        pygame.draw.rect(self.screen, (188,180,23), (self.x0,self.y0, 10,10))
    def respawn(self):
        self.x0,self.y0 = rand_pos(self.screen_x, self.screen_y)