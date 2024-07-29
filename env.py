import pygame
from pygame.locals import QUIT, KEYDOWN
from utils import *

class GoBangEnv:
    def __init__(self):
        self.board = [[0] * 15 for _ in range(15)]
        pygame.init()
        self.screen = pygame.display.set_mode((870, 670))
        pygame.display.set_caption("GoBang")
        self.screen_color = (238, 154, 73)
        self.line_color = (0, 0, 0)
        self.font = pygame.font.SysFont("FangSong", 20)
        self.font2 = pygame.font.SysFont("FangSong", 50)
        self.white_color = (255, 255, 255)
        self.black_color = (0, 0, 0)
        self.situation_info = ''
    
    def is_win(self, player:int, pos: tuple, direction: str) -> bool:
        x, y = pos
        if direction == 'transverse':
            left = [0, -1]
            right = [0, 1]
        elif direction == 'vertical':
            left = [-1, 0]
            right = [1, 0]
        elif direction == 'diagonal':
            left = [-1, -1]
            right = [1, 1]
        elif direction == 'anti-diagonal':
            left = [-1, 1]
            right = [1, -1]
        else:
            raise ValueError(f'direction {direction} is not supported')
        i_left, j_left = x, y
        i_right, j_right = x + right[0], y + right[1]
        count_left, count_right = 0, 0
        while 0 <= i_left < 15 and 0 <= j_left < 15 and self.board[i_left][j_left] == player:
            i_left, j_left = i_left + left[0], j_left + left[1]
            count_left += 1
            if count_left == 5:
                return True
        while 0 <= i_right < 15 and 0 <= j_right < 15 and self.board[i_right][j_right] == player:
            i_right, j_right = i_right + right[0], j_right + right[1]
            count_right += 1
            if count_right == 5:
                return True
        ans = count_left + count_right
        return ans >= 5
    
    def transverse_win(self, player: int, pos: tuple) -> bool:
        return self.is_win(player, pos, 'transverse')
    
    def vertical_win(self, player: int, pos: tuple) -> bool:
        return self.is_win(player, pos, 'vertical')
    
    def diagonal_win(self, player: int, pos: tuple) -> bool:
        return self.is_win(player, pos, 'diagonal')
    
    def anti_diagonal_win(self, player: int, pos: tuple) -> bool:
        return self.is_win(player, pos, 'anti-diagonal')
    
    def internal_rewards(self, player: int, pos: tuple, number: int) -> int:
        x, y = pos
        left_array = [[0, -1], [-1, 0], [-1, -1], [-1, 1]]
        right_array = [[0, 1], [1, 0], [1, -1], [1, 1]]
        for i in range(4):
            left = left_array[i]
            right = right_array[i]
            i_left, j_left = x, y
            i_right, j_right = x + right[0], y + right[1]
            count_left, count_right = 0, 0
            while 0 <= i_left < 15 and 0 <= j_left < 15 and self.board[i_left][j_left] == player:
                i_left, j_left = i_left + left[0], j_left + left[1]
                count_left += 1
                if count_left == number:
                    return True
            while 0 <= i_right < 15 and 0 <= j_right < 15 and self.board[i_right][j_right] == player:
                i_right, j_right = i_right + right[0], j_right + right[1]
                count_right += 1
                if count_right == number:
                    return True
            ans = count_left + count_right
            if ans >= number:
                return True
        return False
    
    def is_illegal(self, pos: tuple) -> bool:
        x, y = pos
        if self.board[x][y] != 0:
            return True
        return False
        
    def state(self) -> list[list[int]]:
        return self.board
    
    def step(self, player: int, pos: tuple):
        x, y = pos
        state = self.state()
        if self.is_illegal(pos):
            return state, -100, False, 0, True # 非法操作
        self.board[x][y] = player
        if self.transverse_win(player, pos) or self.vertical_win(player, pos) or\
              self.diagonal_win(player, pos) or self.anti_diagonal_win(player, pos):
            return state, 200, True, player, False
        opponent = 1 if player == 2 else 2
        if self.transverse_win(opponent, pos) or self.vertical_win(opponent, pos) or\
              self.diagonal_win(opponent, pos) or self.anti_diagonal_win(opponent, pos):
            return state, -200, True, opponent, False
        if all(0 not in row for row in self.board):
            return state, 0, True, -1, False # 平局
        reward = 0
        if self.internal_rewards(player, pos, 3):
            reward = max(reward, 50)
        if self.internal_rewards(player, pos, 4):
            reward = max(reward, 100)
        return state, reward, False, 0, False
    
    def reset(self):
        self.board = [[0 for _ in range(15)] for _ in range(15)]
        self.situation_info = ''
    
    def render(self, player: int, pos:tuple):
        self.screen.fill(self.screen_color)
        j = 1
        for i in range(27, 670, 44):
            if i == 27 or i == 670 - 27:
                pygame.draw.line(self.screen, self.line_color, (i, 27), (i, 670 - 27), 4)
                pygame.draw.line(self.screen, self.line_color, (27, i), (670 - 27, i), 4)
            else:
                pygame.draw.line(self.screen, self.line_color, (i, 27), (i, 670 - 27), 2)
                pygame.draw.line(self.screen, self.line_color, (27, i), (670 - 27, i), 2)
            self.screen.blit(self.font.render(chr(j + 64), True, (0, 0, 0)), (i - 3, 670 - 25))
            self.screen.blit(self.font.render(str(j), True, (0, 0, 0)), (3, i - 10))
            j += 1
        pygame.draw.circle(self.screen, self.line_color, (27 + 44 * 7, 27 + 44 * 7), 8, 0)
        pygame.draw.circle(self.screen, self.line_color, (27 + 44 * 3, 27 + 44 * 3), 8, 0)
        pygame.draw.circle(self.screen, self.line_color, (27 + 44 * 11, 27 + 44 * 3), 8, 0)
        pygame.draw.circle(self.screen, self.line_color, (27 + 44 * 3, 27 + 44 * 11), 8, 0)
        pygame.draw.circle(self.screen, self.line_color, (27 + 44 * 11, 27 + 44 * 11), 8, 0)
        state = self.state()
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == 1:
                    pygame.draw.circle(self.screen, self.black_color, (27 + i * 44, 27 + j * 44), 20, 0)
                elif state[i][j] == 2:
                    pygame.draw.circle(self.screen, self.white_color, (27 + i * 44, 27 + j * 44), 20, 0)
        self.screen.blit(self.font.render('当前执棋：', True, (0, 0, 0)), (680, 50))
        self.screen.blit(self.font2.render(self.situation_info, True, (0, 0, 0)), (660, 335))
        if player == 1:
            pygame.draw.circle(self.screen, self.black_color, (790, 60), 20, 0)
        if player == 2:
            pygame.draw.circle(self.screen, self.white_color, (790, 60), 20, 0)
        i, j = pos
        x, y = 27 + i * 44, 27 + j * 44
        if not self.is_illegal(pos):
            pygame.draw.rect(self.screen, (0, 229, 238), (x - 22, y - 22, 44, 44), 2, 1)
        if self.transverse_win(1, pos) or self.vertical_win(1, pos) or\
              self.diagonal_win(1, pos) or self.anti_diagonal_win(1, pos):
            self.situation_info = '黑胜'
        elif self.transverse_win(2, pos) or self.vertical_win(2, pos) or\
              self.diagonal_win(2, pos) or self.anti_diagonal_win(2, pos):
            self.situation_info = '白胜'
        elif all(0 not in row for row in self.board):
            self.situation_info = '平局'
        else:
            self.situation_info = '正在进行'
        pygame.display.update()