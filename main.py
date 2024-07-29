import pygame
import sys

from pygame.locals import QUIT, KEYDOWN

from env import GoBangEnv
from utils import *
pygame.init()

is_train = False
player = 1
env = GoBangEnv()
tim = 0
over = False
flag = False
pos = (0, 0)

while True:
    for event in pygame.event.get():
        if event.type in (QUIT, KEYDOWN):
            pygame.quit()
            sys.exit()
    env.render(player, pos)
    if not over:
        x, y = pygame.mouse.get_pos()
        x, y = find_pos(x, y)
        if (x, y) != (-1, -1):
            pos = (x // 44, y // 44)
            # print(pos)
            keys_pressed = pygame.mouse.get_pressed()
            if keys_pressed[0] and tim == 0:
                flag = True
                if not env.is_illegal(pos):
                    _, _, over, winner = env.step(player, pos)
                    player = 2 if player == 1 else 1
        if flag:
            tim += 1
        if tim % 200 == 0:
            tim = 0
            flag = False
        