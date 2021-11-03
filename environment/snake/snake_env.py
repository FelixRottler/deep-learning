from math import sqrt
from typing import Tuple
import torch
import random
import pygame
import numpy as np
from collections import deque
from environment.snake.renderables.game_objects import Food, Snake
from environment.environments import Environment


COLOR_BLACK = (14, 20, 19)
COLOR_BORDER = (60, 20, 200)


class SnakeEnv(Environment):

    score = 0
    moving = True

    def __init__(self, window_size, frame_stack_count=4) -> None:
        pygame.init()
        pygame.display.set_caption("SnakeEnv")

        self.font = pygame.font.SysFont("gillsans.ttc", 12)
        self.display = pygame.display.set_mode(window_size, 0, 8)
        self.surface = pygame.Surface(window_size)
        self.clock = pygame.time.Clock()

        self.food = [Food(self.display)]
        self.snake = Snake(self.display)
        self.max_frame_stack_count = frame_stack_count
        self.frame_history = deque(maxlen=self.max_frame_stack_count)

    def step(self, direction) -> Tuple[float, np.ndarray, bool]:
        if not self._opposite_direction(direction):
            self.snake.move(direction)
            self.moving = True
        else:
            self.moving = False

        self.done = self._handle_collisions()
        self.is_eating = self._handle_food()
        reward = self.calculate_reward()

        self._render()

        next_observation = self.get_observation()

        return reward, next_observation, self.done

    def get_observation(self) -> np.ndarray:
        pixel_array = pygame.surfarray.array2d(self.display).astype(np.uint8)
        self.frame_history.append(pixel_array)

        while len(self.frame_history) < self.max_frame_stack_count:
            self.frame_history.append(self.frame_history[-1])
        return np.stack(self.frame_history)

    def calculate_reward(self):
        reward = 0.5
        if self.done or not self.moving:
            reward = -10
        elif self.is_eating:
            reward = max(10, self.score)
     
        return reward

    def _opposite_direction(self, direction):
        if (
            (self.snake.direction == 3 and direction == 0)
            or (self.snake.direction == 0 and direction == 3)
            or (self.snake.direction == 2 and direction == 1)
            or (self.snake.direction == 1 and direction == 2)
        ):
            return True
        return False



    def _render(self):
        
        self.display.fill(COLOR_BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for food in self.food:
            food.render()
        self.snake.render()

        self._render_borders()

     
        pygame.display.update()
        self.clock.tick(10)
    def _render_borders(self):
        x,y = self.display.get_size()
        # top line
        pygame.draw.rect(self.display, COLOR_BORDER, [0,0,x,10])
        # bottom line
        pygame.draw.rect(self.display, COLOR_BORDER, [0,y-10,x,y])
        # left line
        pygame.draw.rect(self.display, COLOR_BORDER, [0,0,10, y])
        # right line
        pygame.draw.rect(self.display, COLOR_BORDER, [x-10,0,x, y])

    def _handle_food(self):
        x, y = self.snake.body[0]
        for food in self.food:
            if x == food.position[0] and y == food.position[1]:
                self.snake.grow = True
                food.respawn()
                self.score += 1
                return True
        return False

    def _handle_collisions(self):
        x, y = self.snake.body[0]
        x_screen, y_screen = self.display.get_size()

        if x > x_screen - 20 or x < 10 or y > y_screen - 20 or y < 10:
            return True
        if [x, y] in self.snake.body[1:]:
            return True
        return False

    def reset(self):
        self.done = False
        self.score = 0
        self.moving = True
        self.snake = Snake(self.display)
        self.food = [Food(self.display)]
        self.frame_history = deque(maxlen=self.max_frame_stack_count)
