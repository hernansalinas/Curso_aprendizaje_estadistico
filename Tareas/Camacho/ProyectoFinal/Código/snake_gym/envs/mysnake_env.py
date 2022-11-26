from collections import deque
import time

import gym
import numpy as np

from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

class SnakeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": "35"
    }

    def __init__(self, height=20, width=20, scaling_factor=6,
                 starting_position=(7, 5), snake_size=3, direction=(0, 1),
                 time_penalty=0, food_reward=10, loss_penalty=-10, win_reward=100):
        self.action_space = spaces.Discrete(3)
        self.ACTIONS = ["STRAIGHT", "LEFT", "RIGHT"]
        self.observation_space = spaces.Box(0, 2, (11,), dtype="uint8")
        self.viewer = None
        self.seed()

        # rewards and penalties
        self.reward = 0
        self.time_penalty = time_penalty
        self.food_reward = food_reward
        self.loss_penalty = loss_penalty
        self.win_reward = win_reward
        if loss_penalty > 0 or time_penalty > 0:
            logger.warn("Values of penalties should not be positive.")

        # initialize size and position properties
        self.height = height
        self.width = width
        if height + 1 > starting_position[0] > 0 and width + 1 > starting_position[1] > snake_size:
            self.starting_position = starting_position
        else:
            raise ValueError("starting_position of snake should be in range (0 - height + 1, snake_size - width + 1)")
        self.scaling_factor = scaling_factor
        self.initial_size = snake_size
        self.snake_size = snake_size
        self.max_size = height * width
        self.state = np.zeros((height + 2, width + 2), dtype="uint8")
        self.game_over = False

        # set bounds of the environment
        self.state[:, 0] = self.state[:, -1] = 1
        self.state[0, :] = self.state[-1, :] = 1

        # initialize snake properties
        self.initial_direction = direction
        self.direction = direction
        self.snake = deque()

        # initialize position of the snake
        self._init_field(starting_position, snake_size)

        # place food on the field
        self.food = self._generate_food()

        # Create a new state of the snake to train the model
        self.my_state = self.get_state()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _init_field(self, starting_position, snake_size):
        y, x = starting_position
        for i in range(snake_size):
            # Bool 1 if there is snake-body
            self.state[y][x] = 1
            self.snake.appendleft((y, x))
            x -= 1                             # BLOCK_SIZE=1

    def _generate_food(self):
        # Random position generator for food
        y, x = self.np_random.randint(self.height), self.np_random.randint(self.width)  # BLOCK_SIZE=1
        while self.state[y][x]:
            y, x = self.np_random.randint(self.height), self.np_random.randint(self.width)
        self.state[y][x] = 2

        return y, x

    # Function to predict the collision of the snake with your environment
    def check_for_collision_agent(self, y, x):
        bool_collision = False
        if self.state[y][x]:
            bool_collision = True
            # If the head-snake collides with the food so returns false
            if self.state[y][x] == 2:
                bool_collision = False

        return bool_collision

    def _check_for_collision(self, y, x):
        done = False
        pop = True
        reward = self.time_penalty

        if self.state[y][x]:
            if self.state[y][x] == 2:
                pop = False
                reward += self.food_reward
                self.snake_size += 1
                if self.snake_size == self.max_size:
                    reward += self.win_reward
                    self.game_over = done = True
                self.food = self._generate_food()
            else:
                reward += self.loss_penalty
                self.game_over = done = True
                pop = False

        self.state[y][x] = 1

        return reward, done, pop

    def step(self, action):
        y, x = self.snake[-1]
        if action == 0:
            y += self.direction[0]
            x += self.direction[1]
        elif action == 1:
            if self.direction[0] == 0:
                self.direction = (-self.direction[1], 0)
                y += self.direction[0]
            else:
                self.direction = (0, self.direction[0])
                x += self.direction[1]
        elif action == 2:
            if self.direction[0] == 0:
                self.direction = (self.direction[1], 0)
                y += self.direction[0]
            else:
                self.direction = (0, -self.direction[0])
                x += self.direction[1]

        reward, done, pop = self._check_for_collision(y, x)

        if not done:
            self.snake.append((y, x))

        if pop:
            y, x = self.snake.popleft()
            self.state[y][x] = 0

        observation = self.state

        my_observation = self.get_state()

        info = {
            "snake": self.snake,
            "snake_size": self.snake_size,
            "direction": self.direction,
            "food": self.food
        }

        return my_observation, reward, done, info

    def reset(self):
        self.game_over = False
        self.direction = self.initial_direction

        while self.snake:
            y, x = self.snake.pop()
            self.state[y][x] = 0

        self.state[self.food[0]][self.food[1]] = 0

        self._init_field(self.starting_position, self.initial_size)
        self.food = self._generate_food()
        self.snake_size = self.initial_size

        self.my_state = self.get_state()

        return self.my_state

    def _to_rgb(self, scaling_factor):
        scaled_grid = np.zeros(((self.height + 2) * scaling_factor, (self.width + 2) * scaling_factor), dtype="uint8")
        scaled_grid[:, :scaling_factor] = scaled_grid[:, -scaling_factor:] = 255
        scaled_grid[:scaling_factor, :] = scaled_grid[-scaling_factor:, :] = 255

        y, x = self.food
        scaled_y, scaled_x = y * scaling_factor, x * scaling_factor
        scaled_grid[scaled_y : scaled_y + scaling_factor, scaled_x : scaled_x + scaling_factor] = 255

        for (y, x) in self.snake:
            scaled_y, scaled_x = y * scaling_factor, x * scaling_factor
            scaled_grid[scaled_y : scaled_y + scaling_factor, scaled_x : scaled_x + scaling_factor] = 255

        img = np.empty(((self.height + 2) * scaling_factor, (self.width + 2) * scaling_factor, 3), dtype="uint8")
        img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = scaled_grid

        return img

    def render(self, mode="human", close=False):
        img = self._to_rgb(self.scaling_factor)
        if mode == "rgb_array":
            return img
        elif mode == "human":
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            time.sleep(0.027)

            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # Function to return the actual state of the snake
    def get_state(self):
        # Position of the head of the snake
        y, x = self.snake[-1]

        # Positions of the left, right, up and down pixels of the head-snake
        (yl, xl) = (y, x - 1)
        (yr, xr) = (y, x + 1)
        (yu, xu) = (y - 1, x)
        (yd, xd) = (y + 1, x)

        # Get the danger-state of each pixel
        b_l = self.check_for_collision_agent(yl, xl)
        b_r = self.check_for_collision_agent(yr, xr)
        b_u = self.check_for_collision_agent(yu, xu)
        b_d = self.check_for_collision_agent(yd, xd)
        
        # Actual direction of the snake
        dir_l = self.direction == (0, -1)
        dir_r = self.direction == (0, 1)
        dir_u = self.direction == (-1, 0)
        dir_d = self.direction == (1, 0)
        
        # Position of the food
        y_food, x_food = self.food

        state = [
            # Danger straight
            (dir_r and b_r) or 
            (dir_l and b_l) or 
            (dir_u and b_u) or 
            (dir_d and b_d),

            # Danger right
            (dir_u and b_r) or 
            (dir_d and b_l) or 
            (dir_l and b_u) or 
            (dir_r and b_d),

            # Danger left
            (dir_d and b_r) or 
            (dir_u and b_l) or 
            (dir_r and b_u) or 
            (dir_l and b_d),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            x_food < x,
            x_food > x,
            y_food < y,
            y_food > y
            ]

        return np.array(state, dtype=int)
