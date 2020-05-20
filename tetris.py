#!/usr/bin/env python

import argparse
import os, sys, time
from random import random, randint, sample, shuffle, randrange
import numpy as np
import pygame
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from collections import deque

from tetris_dqn import DeepQNetwork
import logging
from logging.config import fileConfig
from logging.handlers import RotatingFileHandler
import yaml

#
# configure directories
#
bindir = os.path.abspath(os.path.dirname(__file__))
etcdir = os.path.join(bindir, "etc")
libdir = os.path.join(bindir, "lib")
logdir = os.path.join(bindir, "log")
basename = os.path.basename(__file__)
exename = os.path.splitext(basename)[0]

logging_cfg_file = os.path.join(etcdir, exename + ".yaml")
with open(logging_cfg_file, 'r') as f:
    logcfg = yaml.safe_load(f.read())
    logging.config.dictConfig(logcfg)

logger = logging.getLogger(exename)


class Tetris:
    # region parameters
    piece_colors = [
        # 0 black background
        (0, 0, 0),
        # 1
        (255, 255, 0),
        # 2
        (153, 0, 204),
        # 3
        (51, 204, 51),
        # 4
        (255, 51, 0),
        # 5
        (0, 255, 255),
        # 6
        (255, 153, 0),
        # 7
        (0, 0, 255),
        # 8 Helper color for background grid
        (25, 25, 25)
    ]
    pieces = [
        # 1
        [[1, 1],
         [1, 1]],
        # 2
        [[0, 2, 0],
         [2, 2, 2]],
        # 3
        [[0, 3, 3],
         [3, 3, 0]],
        # 4
        [[4, 4, 0],
         [0, 4, 4]],
        # 5
        [[5, 5, 5, 5]],
        # 6
        [[0, 0, 6],
         [6, 6, 6]],
        # 7
        [[7, 0, 0],
         [7, 7, 7]]
    ]

    # logical variables
    cols = 10
    rows = 20

    board = None
    piece = None
    piece_id = None
    next_piece_id = None
    current_pos = {"x": 0, "y": 0}
    action = None
    bag = None

    # statistics
    score = 0
    cleared_rows = 0
    tetrominoes = 0

    # states
    gameover = False
    paused = False
    automode = False

    # control
    interval = 1000  # 1 second for pygame.USEREVENT
    key_actions = {}

    # gui variables
    screen = None  # the canvas from pygame
    hint_cols = 8
    cell_size = 36  # pixel
    width = None  # pixel size of play+hint fields = (cols + hint_cols) * cell_size
    height = None  # pixel
    field_width = None  # pixel size of play field = cols * cell_size

    default_font = None
    font_size = 22  # cell_size*3//5
    maxfps = 30
    pygame_clock = None

    # torch
    model = None
    run_mode = "play"  # or train
    run_path = ""

    # debug
    debug = 1

    # endregion

    def __init__(self, automode=False):
        self.rows = opt.rows
        self.cols = opt.cols
        self.cell_size = opt.cell_size
        self.maxfps = opt.maxfps
        self.run_mode = opt.run_mode
        self.run_path = opt.run_path
        self.checkpoint_path = os.path.join(opt.train_path, "checkpoint", "checkpoint.pt")
        self.best_path = os.path.join(opt.train_path, "best", "best_model")

        self.gui = opt.gui
        self.debug = opt.debug

        # display
        self.width = self.cell_size * (self.cols + self.hint_cols)  # play area + hint area
        self.height = self.cell_size * self.rows
        self.field_width = self.cell_size * self.cols
        self.bg_grid = [[8 if x % 2 == y % 2 else 0 for x in range(self.cols)] for y in range(self.rows)]
        self.font_size = self.cell_size * 3 // 5

        # https://www.pygame.org/docs/ref/key.html
        self.key_actions = {
            'ESCAPE': self.quit,
            'LEFT': lambda: self.k_move(-1),
            'RIGHT': lambda: self.k_move(+1),
            'DOWN': lambda: self.k_drop(),
            'PAGEUP': lambda: self.k_speeding(+1),
            'PAGEDOWN': lambda: self.k_speeding(-1),
            'UP': self.k_rotate,
            'p': self.k_toggle_pause,
            'SPACE': self.k_fast_drop,
            'RETURN': self.start_game,
            'F1': self.k_toggle_automode,
            'INSERT': lambda: self.k_debuging(+1),
            'DELETE': lambda: self.k_debuging(-1),
        }

        self.next_piece_id = self.draw_lots()

        if self.gui:
            pygame.init()
            # accelerate key speed, first delay, and following interval
            pygame.key.set_repeat(250, 25)
            self.default_font = pygame.font.Font(pygame.font.get_default_font(), self.font_size)
            # start showing the gui
            self.screen = pygame.display.set_mode((self.width, self.height))
            # We do not need mouse movement events, so we block them.
            pygame.event.set_blocked(pygame.MOUSEMOTION)
            self.pygame_clock = pygame.time.Clock()

    def run(self):
        self.gameover = False
        self.paused = False

        self.init_torch()
        self.start_game()

        if self.gui:
            while True:
                if self.gameover:
                    self.center_msg("""Game Over!\nYour score: %d\nPress enter to continue""" % self.score)
                else:
                    if self.paused:
                        self.center_msg("Paused")
                    else:
                        self.screen.fill((0, 0, 0))
                        # draw info window
                        # draw line to split playing area
                        pygame.draw.line(self.screen, (255, 255, 255), (self.field_width + 1, 0),
                                         (self.field_width + 1, self.height - 1))
                        # display message at top left of split area
                        self.display_msg("Next:", (self.field_width + self.cell_size, self.cell_size))
                        # draw the next stone
                        self.draw_matrix(self.pieces[self.next_piece_id], (self.cols + 1, 2.5))
                        # show the action advised by AI
                        if self.action is not None:
                            self.display_msg("AI: x=%d, r=%d" % (self.action[0], self.action[1]),
                                             (self.field_width + self.cell_size, self.cell_size * 6))
                        # show score and cleaned rows
                        self.display_msg("Score: %d\nRows: %d" % (self.score, self.cleared_rows),
                                         (self.field_width + self.cell_size, self.cell_size * 7.5))
                        bumpiness, height = self.get_bumpiness_and_height(self.board)
                        self.display_msg(
                            "Hole: %d\nbumpiness: %d\nheight: %d" % (self.get_holes(self.board), bumpiness, height),
                            (self.field_width + self.cell_size, self.cell_size * 10))
                        self.display_msg("Automode: %r" % (self.automode),
                                         (self.field_width + self.cell_size, self.cell_size * 14))

                        # draw play area
                        self.draw_matrix(self.bg_grid, (0, 0))  # draw background
                        self.draw_matrix(self.board, (0, 0))
                        self.draw_matrix(self.piece, tuple(self.current_pos.values()))

                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.USEREVENT + 1:
                        self.k_drop()
                    elif event.type == pygame.QUIT:
                        self.quit()
                    elif event.type == pygame.KEYDOWN:
                        for key in self.key_actions:
                            if event.key == eval("pygame.K_" + key):
                                self.key_actions[key]()
                # set maximum frame per second
                self.pygame_clock.tick(self.maxfps)

    def init_torch(self):
        torch.manual_seed(int(time.time()))
        self.model = torch.load(f"{self.run_path}/tetris", map_location=lambda storage, loc: storage)
        self.model.eval()

    def start_game(self):
        self.init_game()
        self.gameover = False
        return self.get_state_properties(self.board)

    def init_game(self):
        self.new_board()
        self.new_piece()
        if self.automode:
            self.ai()

        if self.gui:
            # create an event on USEREVENT queue every 1 second
            pygame.time.set_timer(pygame.USEREVENT + 1, self.interval)

    def new_board(self):
        self.board = [
            [0 for x in range(self.cols)]
            for y in range(self.rows)
        ]
        self.score = 0
        self.cleared_rows = 0
        self.tetrominoes = 0
        self.action = None

    def new_piece(self):
        self.piece_id = self.next_piece_id
        self.next_piece_id = self.draw_lots()
        # copy a piece from pieces
        self.piece = [row[:] for row in self.pieces[self.piece_id]]
        self.current_pos = {"x": int(self.cols / 2 - len(self.piece[0]) / 2), "y": 0}

        if self.check_collided(self.piece, self.current_pos):
            self.gameover = True

    # region gui actions
    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:  # if not 0, a color number
                    pygame.draw.rect(self.screen,
                                     self.piece_colors[val],
                                     pygame.Rect(int((off_x + x) * self.cell_size),
                                                 int((off_y + y) * self.cell_size),
                                                 self.cell_size,
                                                 self.cell_size), 0)

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(line, False, (255, 255, 255), (0, 0, 0))
            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2
            self.screen.blit(msg_image,
                             (self.width // 2 - msgim_center_x,
                              self.height // 2 - msgim_center_y + i * (10 + self.font_size)))

    def display_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(line, False, (255, 255, 255), (0, 0, 0)), (int(x), int(y)))
            y += 10 + self.font_size

    # endregion

    # region keybroad actions
    def quit(self):
        if self.gui:
            self.center_msg("Exiting...")
            pygame.display.update()
        sys.exit()

    def k_move(self, delta_x):
        if not self.gameover and not self.paused:
            copy_pos = self.current_pos.copy()
            new_x = copy_pos["x"] + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > self.cols - len(self.piece[0]):
                new_x = self.cols - len(self.piece[0])
            copy_pos["x"] = new_x
            if not self.check_collided(self.piece, copy_pos):
                self.current_pos["x"] = new_x

    def k_rotate(self):
        if not self.gameover and not self.paused:
            rotated_piece = self.rotate(self.piece)
            if not self.check_collided(rotated_piece, self.current_pos):
                self.piece = rotated_piece

    def k_drop(self):
        if not self.gameover and not self.paused:
            if self.check_colliding(self.piece, self.current_pos):
                self.tetrominoes += 1
                overflow = self.truncate(self.piece, self.current_pos)
                if overflow:
                    self.gameover = True
                self.merge(self.board, self.piece, self.current_pos)
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board):
                        if 0 not in row:
                            self.remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    # if no row found to be removed
                    else:
                        if cleared_rows > 0:
                            self.score += self.score_formula(cleared_rows)
                            self.cleared_rows += cleared_rows

                            if not self.gui:
                                logger.debug(f"score {self.score} cleaned rows {self.cleared_rows}")
                        break

                self.new_piece()
                self.ai()
                if opt.debug > 1:
                    logger.debug(self.action)

                # auto
                if self.automode:
                    self.current_pos["x"] = self.action[0]
                    for _ in range(self.action[1]):
                        self.piece = self.rotate(self.piece)
                    if self.check_collided(self.piece, self.current_pos) or self.check_colliding(self.piece,
                                                                                                 self.current_pos):
                        self.gameover = True
                return True
            else:
                self.current_pos["y"] += 1
        return False

    def k_fast_drop(self):
        if not self.gameover and not self.paused:
            while (not self.k_drop()):
                pass

    def k_speeding(self, updown):
        if self.debug > 0:
            if updown > 0:
                print(f"speeding up, interval is {self.interval} ms")
            else:
                print(f"speeding down, interval is {self.interval} ms")

        while updown > 0 and self.interval > 1:
            self.interval = self.interval // 10
            updown -= 1
        while updown < 0:
            self.interval = self.interval * 10
            updown += 1
        if self.gui:
            # display. create an event on USEREVENT queue every 1 second
            pygame.time.set_timer(pygame.USEREVENT + 1, 0)
            pygame.time.set_timer(pygame.USEREVENT + 1, self.interval)

    def k_debuging(self, updown):
        if updown == 1:
            self.debug += 1
            if self.debug > 0:
                print(f"debug level up {self.debug}")
        elif updown == -1:
            if self.debug > 0:
                self.debug -= 1
                print(f"debug level down {self.debug}")

    def k_toggle_pause(self):
        self.paused = not self.paused
        if self.debug > 0:
            print(f"paused? {self.paused}")

    def k_toggle_automode(self):
        self.automode = not self.automode

    # endregion

    # region static methods
    @staticmethod
    def rotate(piece):
        # this is counterclockwise!
        return [[piece[y][x] for y in range(len(piece))] for x in range(len(piece[0]) - 1, -1, -1)]
        # this is clockwise
        # return [[piece[y][x] for y in range(len(piece) - 1, -1, -1)] for x in range(len(piece[0]))]

    @staticmethod
    def merge(board, piece, pos, inplace=True):
        if not inplace:
            # make a copy of recent board
            board = [x[:] for x in board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    # endregion

    # region general functions
    def score_formula(self, cleared_rows):
        return 1 + (cleared_rows ** 2) * self.cols
        # return 1 + cleared_rows * self.cols - delta_holes * self.cols // 2

    # check whether one more drop will collide or not
    def check_colliding(self, piece, pos):
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.rows - 1 \
                        or pos["x"] + x > self.cols - 1 \
                        or pos["x"] + x < 0 \
                        or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    # check whether the piece has already collided or not
    def check_collided(self, piece, pos):
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if pos["y"] + y > self.rows - 1 \
                        or pos["x"] + x > self.cols - 1 \
                        or pos["x"] + x < 0 \
                        or self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def remove_row(self, board, row):
        del board[row]
        board.insert(0, [0 for _ in range(self.cols)])

    def remove_rows(self, board, indices):
        for i in indices[::-1]:
            self.remove_row(board, i)

    def draw_lots(self):
        if opt.cheating:
            # cheating version
            if self.bag is None or not len(self.bag):
                self.bag = list(range(len(self.pieces)))
                shuffle(self.bag)
            return self.bag.pop()
        else:
            # self.bag = list(range(len(self.pieces)))
            # shuffle(self.bag)
            return randrange(len(self.pieces))

    # endregion

    # region torch
    def ai(self):
        # dict of (move, rotation) => tensor([cleaned_rows, holes, bumpiness, height])
        next_steps = self.get_next_states()
        # next_actions: tuple of tuples of (move, rotation)
        # next_states: list of tensors of states
        next_actions, next_states = zip(*next_steps.items())
        # turn a list of tensors into one tensor
        next_states = torch.stack(next_states)
        # use model to calculate gradients of next_states
        predictions = self.model(next_states)[:, 0]
        # find the maximum gradient
        index = torch.argmax(predictions).item()
        # take the action that gradient is corresponding to
        self.action = next_actions[index]
        if self.debug > 1:
            print(f"advised action: {self.action}\n")
        if self.debug > 2:
            print("piece:")
            [print(i, row) for i, row in enumerate(self.piece)]
            print("board:")
            [print(i, row) for i, row in enumerate(self.board)]

    def get_next_states(self):
        states = {}
        piece_id = self.piece_id
        # make a copy of the piece on board
        copy_piece = [row[:] for row in self.piece]

        # different piece has different rotation possibilities
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.cols - len(copy_piece[0])
            for x in range(valid_xs + 1):
                # make a copy of copy_piece
                piece = [row[:] for row in copy_piece]
                pos = {"x": x, "y": 0}
                while not self.check_colliding(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                # create a copy of self.board and join the this piece
                board = self.merge(self.board, piece, pos, inplace=False)
                # save a state for each of possible x position and rotated position of the stone after dropped
                # calculate removed rows, holes, bumpiness and height, get torch tensor value
                states[(x, i)] = self.get_state_properties(board)
                if self.debug > 3:
                    print(x, i, states[(x, i)])
            # rotating copy_piece once
            copy_piece = self.rotate(copy_piece)
        if self.debug > 2:
            print("x, rotate, rows_cleared, holes, bumpiness, height")
        return states

    def get_state_properties(self, board):
        rows_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        return torch.FloatTensor([rows_cleared, holes, bumpiness, height])

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            self.remove_rows(board, to_delete)
        return len(to_delete), board

    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.rows)
        heights = self.rows - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.rows and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def truncate(self, piece, pos):
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    # endregion

    # region no gui
    def test(self):
        self.automode = True
        self.init_torch()
        while True:
            self.start_game()
            while not self.gameover:
                self.ai()
                self.step(self.action)
            logger.info(f"score {self.score}, cleaned rows {self.cleared_rows}")

    def step(self, action):
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_colliding(self.piece, self.current_pos):
            self.current_pos["y"] += 1

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.merge(self.board, self.piece, self.current_pos)

        cleared_rows, self.board = self.check_cleared_rows(self.board)
        score = self.score_formula(cleared_rows)
        self.score += score
        self.cleared_rows += cleared_rows
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 2

        if self.gui:
            if not self.paused:
                self.display()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    for key in self.key_actions:
                        if event.key == eval("pygame.K_" + key):
                            self.key_actions[key]()
            # set maximum frame per second
            # self.pygame_clock.tick(self.maxfps)

        return score, self.gameover

    def training_report(self):
        torch.manual_seed(int(time.time()))
        model = DeepQNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

        # self.model = torch.load(f"{self.run_path}/tetris", map_location=lambda storage, loc: storage)
        # self.model.eval()

        # '''
        if os.path.isfile(self.checkpoint_path):
            model, optimizer, epoch, max_score, best_cleaned_rows, best_epoch = self.load_checkpoint(model, optimizer)
            logger.info(
                f"loaded checkpoint before epoch {epoch} with historical record of max score {max_score} cleaned rows {best_cleaned_rows} on epoch {best_epoch}")
        # '''

        from torchvision import models
        from torchsummary import summary
        vgg = models.vgg16()
        summary(vgg, (3, 224, 224))

    def training(self):
        torch.manual_seed(int(time.time()))
        writer = SummaryWriter(os.path.join(opt.log_path, "tensorboard"))

        epoch = 0
        max_score = 0
        best_cleaned_rows = 0
        best_epoch = 0

        model = DeepQNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

        if opt.cleanup:
            self.cleanup_checkpoint()
        if os.path.isfile(self.checkpoint_path):
            model, optimizer, epoch, max_score, best_cleaned_rows, best_epoch = self.load_checkpoint(model, optimizer)
            logger.info(
                f"loaded checkpoint before epoch {epoch} with historical record of max score {max_score} cleaned rows {best_cleaned_rows} on epoch {best_epoch}")

        criterion = nn.MSELoss()
        state = self.start_game()
        replay_memory = deque(maxlen=opt.replay_memory_size)
        while epoch < opt.num_epochs:
            next_steps = self.get_next_states()
            # Exploration or exploitation
            epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
            u = random()
            random_action = u <= epsilon
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            # notify all your layers that you are in eval mode, that way,
            # batchnorm or dropout layers will work in eval mode instead of training mode
            model.eval()
            # no_grad() impacts the autograd engine and deactivate it
            # here it helps saving some memory for performace consideration
            with torch.no_grad():
                predictions = model(next_states)[:, 0]
            # change back to train mode
            model.train()
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                index = torch.argmax(predictions).item()

            next_state = next_states[index, :]
            action = next_actions[index]

            reward, done = self.step(action)

            # recording rewards from recent state (rows_cleared, holes, bumpiness, height) to next state
            replay_memory.append([state, reward, next_state, done])
            if done:
                final_score = self.score
                final_tetrominoes = self.tetrominoes
                final_cleared_rows = self.cleared_rows
                final_bumpiness, final_height = self.get_bumpiness_and_height(self.board)
                state = self.start_game()
            else:
                state = next_state
                continue
            if len(replay_memory) < opt.replay_memory_size / 10:
                continue
            epoch += 1

            # randomly taking opt.batch_size of samples from the recorded history
            batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
            state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = torch.stack(tuple(state for state in state_batch))
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = torch.stack(tuple(state for state in next_state_batch))

            # q values (512x1) from states (512x4)
            q_values = model(state_batch)
            model.eval()
            with torch.no_grad():
                next_prediction_batch = model(next_state_batch)
            model.train()
            # calculate the predicated q values of next states from the model and reward
            q_predicates = torch.cat(
                tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                      zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
            # loss function (least squares),
            optimizer.zero_grad()
            loss = criterion(q_values, q_predicates) # single float
            loss.backward() # dLoss/dWeight
            optimizer.step() # adjust weight

            is_best = False
            if final_score > max_score:
                max_score = final_score
                best_cleaned_rows = final_cleared_rows
                best_epoch = epoch
                is_best = True

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'max_score': max_score,
                'best_cleaned_rows': best_cleaned_rows,
                'best_epoch': best_epoch
            }

            self.save_checkpoint(checkpoint, model, is_best)

            logger.info(
                "Epoch: {}/{}, Loss: {}, Score: {}, Tetrominoes {}, Cleared Rows: {}, record high: {} {} {}".format(
                    epoch,
                    opt.num_epochs,
                    loss.item(),
                    final_score,
                    final_tetrominoes,
                    final_cleared_rows,
                    max_score,
                    best_cleaned_rows,
                    best_epoch))
            writer.add_scalar('Train/Score', final_score, epoch - 1)
            writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
            writer.add_scalar('Train/Cleared_Rows', final_cleared_rows, epoch - 1)
            writer.add_scalar('Train/Bumpiness', final_bumpiness, epoch - 1)
            writer.add_scalar('Train/Height', final_height, epoch - 1)
            writer.add_scalar('Train/Loss', loss.item(), epoch - 1)

            # if epoch > 0 and epoch % opt.save_interval == 0:
            # torch.save(model, "{}/tetris_{}".format(opt.train_path, epoch))
        torch.save(model, "{}/tetris".format(opt.train_path))

    def save_checkpoint(self, checkpoint, model, is_best):
        torch.save(checkpoint, self.checkpoint_path)
        if is_best:
            torch.save(model, self.best_path)

    def load_checkpoint(self, model, optimizer):
        checkpoint = torch.load(self.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch'], checkpoint['max_score'], checkpoint['best_cleaned_rows'], \
               checkpoint['best_epoch']

    def cleanup_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        if os.path.isfile(self.best_path):
            os.remove(self.best_path)

    def display(self):
        if self.gameover:
            self.center_msg("""Game Over!\nYour score: %d\nPress enter to continue""" % self.score)
        else:
            if False:  # self.paused:
                self.center_msg("Paused")
            else:
                self.screen.fill((0, 0, 0))
                # draw info window
                # draw line to split playing area
                pygame.draw.line(self.screen, (255, 255, 255), (self.field_width + 1, 0),
                                 (self.field_width + 1, self.height - 1))
                # display message at top left of split area
                self.display_msg("Next:", (self.field_width + self.cell_size, 2))
                # draw the next stone
                self.draw_matrix(self.pieces[self.next_piece_id], (self.cols + 1, 1))
                # show the action advised by AI
                if self.action is not None:
                    self.display_msg("Action: x=%d, rotation=%d" % (self.action[0], self.action[1]),
                                     (self.field_width + self.cell_size, self.cell_size * 4))
                # show score and cleaned rows
                self.display_msg("Score: %d\nRows: %d" % (self.score, self.cleared_rows),
                                 (self.field_width + self.cell_size, self.cell_size * 5))
                bumpiness, height = self.get_bumpiness_and_height(self.board)
                self.display_msg(
                    "Hole: %d\nbumpiness: %d\nheight: %d" % (self.get_holes(self.board), bumpiness, height),
                    (self.field_width + self.cell_size, self.cell_size * 8))
                self.display_msg("Automode: %r" % (self.automode),
                                 (self.field_width + self.cell_size, self.cell_size * 11))

                # draw play area
                self.draw_matrix(self.bg_grid, (0, 0))  # draw background
                self.draw_matrix(self.board, (0, 0))
                self.draw_matrix(self.piece, tuple(self.current_pos.values()))
        pygame.display.update()
    # endregion


def get_args():
    parser = argparse.ArgumentParser("""AI Powered Tetris""")
    parser.add_argument("--cols", type=int, default=10, help="columns. default:10")
    parser.add_argument("--rows", type=int, default=20, help="rows. default:20")
    # display
    parser.add_argument("--cell_size", type=int, default=30, help="size of a cell. default:30")
    parser.add_argument("--maxfps", type=int, default=30, help="cap the fps maximum at. default:30")

    parser.add_argument("--cleanup", type=bool, default=False,
                        help="True|False. clean up training history. Default:False")
    parser.add_argument("--log_path", type=str, default="log", help="Default:log")
    parser.add_argument("--train_path", type=str, default="models/training", help="Default: models/training")
    parser.add_argument("--run_path", type=str, default="models", help="Default: models")
    parser.add_argument("--cheating", type=bool, default=True,
                        help="use shuffle bag instead of full random piece. default:False")
    # run modes
    parser.add_argument("--run_mode", type=str, default="play", help="play|train|test|report. default:play")
    parser.add_argument("--gui", type=bool, default=True, help="default:True")
    parser.add_argument("--debug", type=int, default=0, help="default:0")

    # torch arguments
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1, help="Default:1")
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Default: 30000. Number of epoches between testing phases")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    if opt.run_mode == "play":
        opt.gui = True
        App = Tetris()
        App.run()
    elif opt.run_mode == "test":
        App = Tetris(automode=True)
        if opt.gui:
            App.k_speeding(3)
            App.automode = True
            App.run()
        else:
            App.test()
        [print(row) for row in App.board]
    elif opt.run_mode == "train":
        App = Tetris(automode=True)
        App.training()
    elif opt.run_mode == "report":
        opt.gui = False
        App = Tetris(automode=True)
        App.training_report()
