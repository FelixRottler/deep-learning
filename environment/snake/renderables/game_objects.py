import pygame
import random

COLOR_RED = (255, 0, 0)
COLOR_FOOD = (87, 224, 123)
COLOR_SNAKE = (230, 62, 109)
COLOR_SNAKE_HEAD = (90, 15, 123)

class GameObject:
    def render(self, screen):
        pass


class Food(GameObject):
    def __init__(self, display):
        self.display = display
        x, y = display.get_size()
        self.position = (
            random.randrange(10, x - 20, 10),
            random.randrange(10, y - 20, 10),
        )

    def render(self):
        pygame.draw.rect(
            self.display, COLOR_FOOD, (self.position[0], self.position[1], 10, 10)
        )

    def respawn(self):
        x, y = self.display.get_size()
        self.position = (
            random.randrange(10, x-20 , 10),
            random.randrange(10, y-20 , 10),
        )


class Snake(GameObject):
    grow = False
    direction = 0

    def __init__(self, display):
        self.display = display
        self.screen_x, self.screen_y = display.get_size()
        x0, y0 = random.randrange(0, self.screen_x - 30, 10), random.randrange(
            0, self.screen_y - 10, 10
        )
        self.body = [[x0, y0], [x0 + 10, y0], [x0 + 10 * 2, y0]]

    def render(self):
        x0, y0 = self.body[0]
        pygame.draw.rect(self.display, COLOR_SNAKE_HEAD, (x0, y0, 10, 10))
        for x, y in self.body[1:]:
            pygame.draw.rect(self.display, COLOR_SNAKE, (x, y, 10, 10))

    def move(self, direction):
        x, y = self.body[0]
        if direction == 0:
            y -= 10
        elif direction == 1:
            x -= 10
        elif direction == 2:
            x += 10
        elif direction == 3:
            y += 10
        self.body.insert(0, [x, y])  # insert new body part at head
        if not self.grow:
            self.body.pop()  # delete tail if we haven't eaten
        else:
            self.grow = False
        self.direction = direction
