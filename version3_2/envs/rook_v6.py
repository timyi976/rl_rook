import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding
import os

from utils import T1Norm, pt2lineDist_points, pt2lineDist_traj, combine_trajectory

DIR_MAP = {
    0: (0, 2),
    1: (1, 1),
    2: (1, -1),
    3: (0, -2),
    4: (-1, -1),
    5: (-1, 1)
}

class Map():
    def __init__(self, n_markers, map_dim, marker_pos, initial_pos, terminal_pos, block_pos, trajectory):
        self.n_markers = n_markers
        self.map_dim = map_dim
        self.marker_pos = marker_pos
        self.initial_pos = initial_pos
        self.terminal_pos = terminal_pos
        self.block_pos = block_pos
        self.trajectory = trajectory

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
            "block_pos": self.block_pos,
            "trajectory": self.trajectory
        }
    
    def __str__(self):
        s = f"""n_markers: {self.n_markers}
                map_dim: {self.map_dim}
                marker_pos: {self.marker_pos}
                initial_pos: {self.initial_pos}
                terminal_pos: {self.terminal_pos}
                block_pos: {self.block_pos}
                trajectory: {self.trajectory}"""
        
        # remove leading whitespaces of each line
        s = "\n".join([line.strip() for line in s.split("\n")])
        
        return s
        
class RookEnv(gym.Env):
    def __init__(self, mode: str,map_file_dir: str, reward_dict: dict, max_steps: int = 100, seed: int = None):
        super(RookEnv, self).__init__()

        self.mode = mode
        self.map_idx = 0

        self._map_file_dir = map_file_dir
        self.reward_dict = reward_dict
        self.map_steps = max_steps
        # self.seed = seed

        # read files under the directory
        self._map_files = []
        for file in os.listdir(map_file_dir):
            if file.endswith(".txt"):
                self._map_files.append(os.path.join(map_file_dir, file))
        # print("maze number",len(self._map_files))
        # Initialize map
        self.reset()

        # Environment settings
        self.action_space = spaces.Discrete(self.n_markers * 6)
        # TODO: change observation space shape according to _construct_state()
        # print("map dim", self.map_dim) # 7 * 11
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_markers * 4, self.map_dim[0], self.map_dim[1]), dtype=np.int32)

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

        trajectory = np.zeros((n_markers, h, w), dtype=np.int32)

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
                trajectory[k,pos[0],pos[1]] = 1

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
            block_pos=block_pos,
            trajectory=trajectory
        )
    
    def _construct_state(self, map: Map) -> np.ndarray:
        """
        Construct the environment state from the map, including:
            - Marker positions (n_markers layers)
            - Block positions (1 layer)
            - Initial positions (n_markers layer)
            - Terminal positions (n_markers layer)
            - Trajectory (n_markers layers)
                    
        Args:
            map (Map): The map object.

        Returns:
            np.ndarray: The environment state with several components:
                - Marker positions (n_markers layers)
                - Block positions (1 layer)
                - Initial positions (n_markers layer)
                - Terminal positions (n_markers layer)
                - Trajectory (n_markers layers)
        """
        state = np.zeros((self.n_markers * 4, self.map_dim[0], self.map_dim[1]), dtype=np.int32)

        for k in range(self.n_markers):
            block_pos = map.block_pos.copy()
            # put the other markers' initial and terminal position on the block
            for m in range(self.n_markers):
                if k != m:
                    block_pos[map.initial_pos == m] = 1
                    block_pos[map.terminal_pos == m] = 1
            marker_pos = map.marker_pos[k].copy()
            terminal_pos = np.zeros((self.map_dim[0], self.map_dim[1]), dtype=np.int32)
            terminal_pos[map.terminal_pos == k] = 1
            trajectory = map.trajectory[k].copy()

            state[k * 4] = marker_pos
            state[k * 4 + 1] = block_pos
            state[k * 4 + 2] = terminal_pos
            state[k * 4 + 3] = trajectory

        return state
    
    def _check_valid_position(self, map: Map, marker: int, cur_pos: tuple,pos: tuple) -> bool:
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

        self_terminal = map.terminal_pos[cur_pos[0], cur_pos[1]] != -1 and map.terminal_pos[cur_pos[0], cur_pos[1]] == marker

        return not (bound or block or other_initial or other_terminal or self_terminal)
    
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

        return self._check_valid_position(map, marker,cur_pos, next_pos)

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

            trajectory = map.trajectory.copy()
            # mark the trajectory
            # x1 = cur_pos[0] * 2
            # y1 = cur_pos[1]
            # x2 = next_pos[0] * 2
            # y2 = next_pos[1]

            # x = (x1 + x2) // 2
            # y = int((y1 + y2) // 2)

            # trajectory[marker, cur_pos[0], cur_pos[1]] = 1
            trajectory[marker, next_pos[0], next_pos[1]] = 1

        else:
            marker_pos = map.marker_pos.copy()
            trajectory = map.trajectory.copy()
        
        return Map(
            n_markers=map.n_markers,
            map_dim=map.map_dim,
            marker_pos=marker_pos,
            initial_pos=map.initial_pos,
            terminal_pos=map.terminal_pos,
            block_pos=map.block_pos,
            trajectory=trajectory
        )

    def _calculate_reward(self, old_map: Map, new_map: Map, marker: int, direction: int) -> float:
        """
        Calculate the reward for the current step based on the map's current state.

        Args:
            map (Map): The current map object containing all necessary state information.

        Returns:
            float: Reward for the action.
        """
        reward = 0
        prev_marker_coor = np.argwhere(old_map.marker_pos[marker] == 1)[0] 
        cur_marker_coor = np.argwhere(new_map.marker_pos[marker] == 1)[0]
        direction = cur_marker_coor - prev_marker_coor
        terminal_coor = np.argwhere(new_map.terminal_pos == marker)[0]
        #prev_distance_to_terminal = np.sum(np.abs(terminal_coor - prev_marker_coor))
        #cur_distance_to_terminal = np.sum(np.abs(terminal_coor - cur_marker_coor[0]))
        prev_distance_to_terminal = T1Norm(terminal_coor, prev_marker_coor)
        cur_distance_to_terminal = T1Norm(terminal_coor, cur_marker_coor)

        # Add step penalty
        reward += self.reward_dict["step_penalty"]

        # Add stay penalty
        if np.all(old_map.marker_pos == new_map.marker_pos):
            reward += self.reward_dict["stay_penalty"]
            return reward

        # Add reach target reward
        if np.any(prev_marker_coor != terminal_coor) and np.all(cur_marker_coor == terminal_coor):
            reward += self.reward_dict["reached_target"]
            self.wire_length+=1
            return reward

        # Add leave target penalty
        if np.all(prev_marker_coor == terminal_coor) and np.any(cur_marker_coor != terminal_coor):
            reward += self.reward_dict["leave_target_penalty"]
            return reward

        # Add distance bonus for current position to terminal
        #distance_reduce = cur_distance_to_terminal - prev_distance_to_terminal
        distance_reduce = (prev_distance_to_terminal - cur_distance_to_terminal)/2
        # reward += self.reward_dict["target_distance_bonus"] * distance_reduce
        #CANNOT EXCLUDE
        if distance_reduce > 0:
            # print("close to target, +0.2")
            reward += self.reward_dict["target_distance_bonus"]
        elif distance_reduce == 0:
            # print("same dis to target, -0.2")
            reward -= self.reward_dict["target_distance_bonus"]
        if distance_reduce < 0:
            # print("far to target, -0.6")
            reward -= self.reward_dict["target_distance_bonus"] * 3

        # ----------------------------------------------
        # # Add distance bonus for current position to cluster
        # marker_center_coor = np.zeros(2)
        # terminal_center_coor = np.zeros(2)
        # for i in range(map.n_markers):
        #     if i != marker:
        #         marker_center_coor += np.argwhere(map.prev_marker_pos[i] == 1)[0]
        #     # terminal_center_coor += np.argwhere(map.terminal_pos == i)[0]
        # # marker_center_coor += terminal_coor
        # marker_center_coor /= (map.n_markers - 1)
        # #prev_distance_to_center = np.sum(np.abs(marker_center_coor - prev_marker_coor))
        # #cur_distance_to_center = np.sum(np.abs(marker_center_coor - cur_marker_coor))
        # #distance_reduce = cur_distance_to_center - prev_distance_to_center
        # prev_distance_to_center = T1Norm(marker_center_coor, prev_marker_coor)
        # cur_distance_to_center = T1Norm(marker_center_coor, cur_marker_coor)
        # ----------------------------------------------
        # TODO: This might be wrong, need to check. The agent learn to go back to trajectory and oscillate between two cells
        # gray_traj = combine_trajectory(old_map.trajectory, marker)
        # gray_dist = pt2lineDist_traj(prev_marker_coor, gray_traj)
        # # distance_reduce = (prev_distance_to_center - cur_distance_to_center)/2
        # # terminal_center_coor /= map.n_markers
        # # cur_distace_to_cluster = abs(marker_center_coor[0] - cur_marker_coor[0]) + abs(marker_center_coor[1] - cur_marker_coor[1])
        # # cluster_distance_to_terminal = abs(terminal_coor[0] - terminal_center_coor[0]) + abs(terminal_coor[1] - terminal_center_coor[1])
        # # distance_reduce_cluster = max(0, prev_distance_to_terminal - (cur_distace_to_cluster + cluster_distance_to_terminal))
        # # if np.dot(direction, marker_center_coor - prev_marker_coor) > 0:
        # reward += self.reward_dict["distance_bonus"] * gray_dist

        #---------------------------------------------------------------------------
        #Penalty for too many tragectories
        old_trajectory = old_map.trajectory
        new_trajectory = new_map.trajectory

        old_trajectory_pos = np.any(old_trajectory, axis=0)
        new_trajectory_pos = np.any(new_trajectory, axis=0)

        old_trajectory_num = np.sum(old_trajectory_pos)
        new_trajectory_num = np.sum(new_trajectory_pos)

        # if new_trajectory_num - 8 > 0:
        #     reward += max(-0.2 * (new_trajectory_num - 8),-1)

        old_trajectory_flatten = np.zeros_like(old_trajectory[0])
        for m in range(self.n_markers):
            if m==marker:
                old_trajectory_flatten += old_trajectory[m]*10
            else:
                old_trajectory_flatten += old_trajectory[m]

        # print(old_trajectory_flatten)
        # print(cur_marker_coor)
        
        #CANNOT EXCLUDE
        # if old_trajectory_flatten[cur_marker_coor[0]][cur_marker_coor[1]] >= 10: #It goes backward
        #     # print("move to self traj, -1.5")
        #     reward -= 1.5
        # elif  old_trajectory_flatten[cur_marker_coor[0]][cur_marker_coor[1]] > 0: #It goes other's tragectory
        #     # print("move to other's traj, +0.6")
        #     reward += 0.8

        if not (old_trajectory_flatten[prev_marker_coor[0]][prev_marker_coor[1]] > 10 and old_trajectory_flatten[cur_marker_coor[0]][cur_marker_coor[1]] > 0):
            self.wire_length+=1


        #---------------------------------------------------------------------------
        
        # Add reward for merging (if multiple markers are in the same cell)
        # prev_merge_num = np.max(np.sum(old_map.marker_pos, axis=0))  # Shape: (height, width)
        # cur_merge_num = np.max(np.sum(new_map.marker_pos, axis=0))  # Shape: (height, width)
        # merge_num_delta = cur_merge_num - prev_merge_num
        # if merge_num_delta > 0 and np.dot(direction, terminal_coor - prev_marker_coor) > 0:
        #     reward += self.reward_dict["distance_bonus"] * (new_map.n_markers) * prev_merge_num
            # reward += self.reward_dict["merge_reward"] * prev_merge_num

        #----------------------------------------------------------------------------
        #encourage one marker achieve the termonal first,after that(the trajectory is formed), the remaining markers followup
        old_trajectory = old_map.trajectory
        new_trajectory = new_map.trajectory
        # print("old traj:\n",old_trajectory)
        # print("new traj:\n",new_trajectory)

        old_ones_num = np.sum(old_trajectory, axis=(1, 2))
        new_ones_num = np.sum(new_trajectory, axis=(1, 2))

        for m in range(self.n_markers):
            prev_m_coor = np.argwhere(old_map.marker_pos[m] == 1)[0] 
            cur_m_coor = np.argwhere(new_map.marker_pos[m] == 1)[0]
            terminal_m_coor = np.argwhere(new_map.terminal_pos == m)[0]
            if np.all(prev_m_coor == terminal_m_coor): #Then it is excluded
                old_ones_num[m] = -1
                new_ones_num[m] = -1

        # print("old_ones_num",old_ones_num)
        old_most_ones = np.max(old_ones_num)
        new_most_ones = np.max(new_ones_num)

        old_indices_with_most_ones = np.where(old_ones_num == old_most_ones)[0]
        new_indices_with_most_ones = np.where(new_ones_num == new_most_ones)[0]
        # print("old_indices_with_most_ones",old_indices_with_most_ones)

        if not marker in old_indices_with_most_ones and old_most_ones > 0: #you move the marker wrong, you should move the marker who has been started
            # print("not moving most traj marker, -2")
            reward -= 1

        return reward
    
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
        return {"map": map.get_map(),"wire_length":self.wire_length}
    
    def reset(self, seed: int = None):
        self.step_count = 0
        self.wire_length = 0

        if seed is not None:
            np.random.seed(seed)
        if self.mode == "eval":
            self._map_file = self._map_files[self.map_idx]
            self.map_idx += 1
            if self.map_idx == len(self._map_files):
                self.map_idx = 0 
        else:
            self._map_file = np.random.choice(self._map_files)

        # print(f"Loading map from {self._map_file}")

        self.map = self._load_map(self._map_file)

        info = self._construct_info(self.map)

        return self._construct_state(self.map), info

    def step(self, action: tuple):
        self.step_count += 1

        marker, direction = self._decode_action(action)
        old_map = self.map
        self.map = self._move_marker(self.map, marker, direction)
        next_state = self._construct_state(self.map)
        done = self._check_finish(self.map)
        reward = self._calculate_reward(old_map, self.map, marker, direction)

        if self.step_count >= self.map_steps:
            done = True

        info = self._construct_info(self.map)

        # if done:
        #     self.reset()

        truncated = False
        if not done and self.step_count >= self.map_steps:
            truncated = True

        

        return next_state, reward, done, truncated, info
    
    def render(self, mode='human'):
        if mode == "human":
            print(self.map.marker_pos)
            print(self.map.block_pos)
