import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding

DIR_MAP = {
    0: (0, 2),
    1: (1, 1),
    2: (1, -1),
    3: (0, -2),
    4: (-1, -1),
    5: (-1, 1)
}

class Map():
    def __init__(self, n_markers, map_dim, marker_pos, initial_pos, terminal_pos, block_pos):
        self.n_markers = n_markers
        self.map_dim = map_dim
        self.marker_pos = marker_pos
        self.initial_pos = initial_pos
        self.terminal_pos = terminal_pos
        self.block_pos = block_pos

    def get_map(self) -> dict:
        """
        To get the map object. This is a capsule object that contains the map data.

        Returns:
            dict: The map object.
        """

        return {
            "n_markers": self.n_markers,
            "map_dim": self.map_dim,
            "marker_pos": self.marker_pos,
            "initial_pos": self.initial_pos,
            "terminal_pos": self.terminal_pos,
            "block_pos": self.block_pos
        }

class RookEnv(gym.Env):
    def __init__(self, map_file: str, reward_dict: dict, max_steps: int = 100, seed: int = None):
        super(RookEnv, self).__init__()

        self._map_file = map_file
        self.reward_dict = reward_dict
        self.map_steps = max_steps
        # self.seed = seed

        # Initialize map
        self.reset()

        # Environment settings
        self.action_space = spaces.Discrete(self.n_markers * 6)
        # TODO: change observation space shape according to _construct_state()
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_markers, self.map_dim[0], self.map_dim[1]), dtype=np.int32)
        

    def _load_map(self, map_file: str) -> Map:
        """
        Load the map from a file into a Map object. This map is NOT the same as the environment state. Also, the number of checkers on the board is returned.

        Args:
            map_file (str): The path to the grid file.

        Returns:
            Map: The map object.
        """
        with open(map_file, 'r') as f:
            lines = f.readlines()

        n_markers = int(lines[0].strip())
        self.n_markers = n_markers
        grid = [[int(i.strip()) for i in line.strip().split()] for line in lines[1:]]
        grid = np.array(grid)
        h, w = grid.shape
        self.map_dim = (h, w)

        marker_pos = np.zeros((n_markers, h, w), dtype=np.int32)
        initial_pos = np.full((h, w), -1, dtype=np.int32)
        terminal_pos = np.full((h, w), -1, dtype=np.int32)
        block_pos = np.zeros((h, w), dtype=np.int32)

        # initial index offset
        initial_offset = 2 ** n_markers
        terminal_offset = initial_offset + n_markers
        block_offset = terminal_offset + n_markers

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j] == 0:
                    # empty cell
                    marker_pos[:, i, j] = 0
                elif grid[i][j] >= 1 and grid[i][j] < 2 ** n_markers:
                    # markers
                    for k in range(n_markers):
                        if grid[i][j] & (1 << k):
                            marker_pos[k, i, j] = 1
                elif grid[i][j] >= initial_offset and grid[i][j] < terminal_offset:
                    # initial position
                    initial_pos[i, j] = grid[i][j] - initial_offset
                elif grid[i][j] >= terminal_offset and grid[i][j] < block_offset:
                    # terminal position
                    terminal_pos[i, j] = grid[i][j] - terminal_offset
                elif grid[i][j] >= block_offset:
                    # block position
                    block_pos[i, j] = 1

        for k in range(n_markers):
            # check if k exists in initial_pos
            if not np.any(initial_pos == k):
                # initial position of marker k is its current position
                pos = np.argwhere(marker_pos[k] == 1)
                pos = pos[0]
                initial_pos[pos[0], pos[1]] = k

            # check if k exists in terminal_pos
            if not np.any(terminal_pos == k):
                # terminal position of marker k is its current position
                pos = np.argwhere(marker_pos[k] == 1)
                pos = pos[0]
                terminal_pos[pos[0], pos[1]] = k

        return Map(
            n_markers=self.n_markers,
            map_dim=self.map_dim,
            marker_pos=marker_pos,
            initial_pos=initial_pos,
            terminal_pos=terminal_pos,
            block_pos=block_pos
        )
    
    def _construct_state(self, map: Map) -> np.ndarray:
        """
        Construct the environment state from the map.

        Args:
            map (Map): The map object.

        Returns:
            np.ndarray: The environment state.
        """
        # TODO: construct the state from the map
        return map.marker_pos
    
    def _check_valid_position(self, map: Map, marker: int, pos: tuple) -> bool:
        """
        Check if the given position is valid for the marker on the given map.

        Args:
            map (Map): The map object.
            marker (int): The marker index.
            pos (tuple): The position to check.

        Returns:
            bool: Whether the position is valid.
        """
        # check if pos exceeds the map boundary
        bound = pos[0] < 0 or pos[0] >= map.map_dim[0] or pos[1] < 0 or pos[1] >= map.map_dim[1]
        if bound:
            return False
        # check if block
        block = map.block_pos[pos[0], pos[1]] == 1
        # check if the marker is at other marker's initial position
        other_initial = map.initial_pos[pos[0], pos[1]] != -1 and map.initial_pos[pos[0], pos[1]] != marker
        # check if the marker is at other marker's terminal position
        other_terminal = map.terminal_pos[pos[0], pos[1]] != -1 and map.terminal_pos[pos[0], pos[1]] != marker

        return not (bound or block or other_initial or other_terminal)
    
    def _check_valid_move(self, map: Map, marker: int, direction: int) -> bool:
        """
        Check if the move is valid.

        Args:
            map (Map): The current map.
            marker (int): The marker index.
            direction (int): The direction to move the marker.

        Returns:
            bool: Whether the move is invalid.
        """
        cur_pos = np.argwhere(map.marker_pos[marker] == 1)[0]
        next_pos = (cur_pos[0] + DIR_MAP[direction][0], cur_pos[1] + DIR_MAP[direction][1])

        return self._check_valid_position(map, marker, next_pos)

    def _move_marker(self, map:Map, marker: int, direction: int) -> Map:
        """
        Move the marker in the given direction. We only move one marker at a time.

        Args:
            map (Map): The current map object.
            marker (int): The marker index.
            direction (int): The direction to move the marker.

        Returns:
            Map: The new map object after moving the marker.
        """
        valid = self._check_valid_move(map, marker, direction)

        if valid:
            marker_pos = map.marker_pos.copy()
            cur_pos = np.argwhere(marker_pos[marker] == 1)[0]
            next_pos = (cur_pos[0] + DIR_MAP[direction][0], cur_pos[1] + DIR_MAP[direction][1])
            marker_pos[marker, cur_pos[0], cur_pos[1]] = 0
            marker_pos[marker, next_pos[0], next_pos[1]] = 1
        else:
            marker_pos = map.marker_pos.copy()
        
        return Map(
            n_markers=map.n_markers,
            map_dim=map.map_dim,
            marker_pos=marker_pos,
            initial_pos=map.initial_pos,
            terminal_pos=map.terminal_pos,
            block_pos=map.block_pos
        )

    def _calculate_reward(self, map: Map) -> float:
        """
        Calculate the reward for the current state.

        Args:
            map (Map): The current map.

        Returns:
            float: The reward.
        """
        # TODO: use self.reward_dict to define the reward
        return self.reward_dict["step"]
    
    def _check_finish(self, map: Map) -> bool:
        """
        Check if the game is finished.

        Args:
            map (np.ndarray): The current map.

        Returns:
            bool: Whether the game is finished.
        """
        # check if all markers are at their terminal positions
        done = np.full(map.n_markers, False)

        for k in range(map.n_markers):
            pos = np.argwhere(map.marker_pos[k] == 1)[0]
            done[k] = map.terminal_pos[pos[0], pos[1]] == k

        return np.all(done)
    
    def _decode_action(self, action: int) -> tuple:
        """
        Decode the action into the marker index and direction.

        Args:
            action (int): The action to decode.

        Returns:
            tuple: The marker index and direction.
        """
        marker = action // 6
        direction = action % 6

        return marker, direction
    
    def _construct_info(self, map: Map) -> dict:
        """
        Construct the info dictionary for the current state. This is used in step().

        Args:
            map (Map): The current map.

        Returns:
            dict: The info dictionary.
        """
        # TODO: write some metrics to show here
        return {"map": map.get_map()}
    
    def reset(self, seed: int = None):
        self.step_count = 0
        self.map = self._load_map(self._map_file)

        info = self._construct_info(self.map)

        return self._construct_state(self.map), info

    def step(self, action: tuple):
        self.step_count += 1

        marker, direction = self._decode_action(action)
        self.map = self._move_marker(self.map, marker, direction)
        next_state = self._construct_state(self.map)
        done = self._check_finish(self.map)
        reward = self._calculate_reward(self.map)

        if self.step_count >= self.map_steps:
            done = True

        if done:
            self.reset()

        truncated = False
        if not done and self.step_count >= self.map_steps:
            truncated = True

        info = self._construct_info(self.map)

        return next_state, reward, done, truncated, info
    
    def render(self, mode='human'):
        if mode == "human":
            print(self.map.marker_pos)
            print(self.map.block_pos)
