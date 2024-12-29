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
    
    def copy(self):
        return Map(
            n_markers=self.n_markers,
            map_dim=self.map_dim,
            marker_pos=self.marker_pos.copy(),
            initial_pos=self.initial_pos.copy(),
            terminal_pos=self.terminal_pos.copy(),
            block_pos=self.block_pos.copy(),
            trajectory=self.trajectory.copy()
        )

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
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_markers * 5, self.map_dim[0], self.map_dim[1]), dtype=np.int32)
        

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

        # trajectory = np.zeros((n_markers, h * 2 - 1, w), dtype=np.int32)
        trajectory = np.zeros((n_markers, h , w), dtype=np.int32)

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
            - Marker positions
            - Blockages
            - Targets
            - Initial positions.

        Args:
            map (Map): The map object.

        Returns:
            np.ndarray: The environment state with four components:
                - Marker positions (n_markers, height, width)
                - Block positions (1, height, width)
                - Target positions (1, height, width)
                - Initial positions (1, height, width)
        """
        def insert_rows(array):
            h, w = array.shape
            ret = np.zeros((h * 2 - 1, w), dtype=np.int32)

            ret[::2, :] = array

            return ret
        
        #With initial position
        state = np.zeros((self.n_markers * 5, self.map_dim[0], self.map_dim[1]), dtype=np.int32) 

        block_pos = map.block_pos
        # state[0] = block_pos

        for k in range(self.n_markers-1):
            marker = k+1
            location_map = map.marker_pos[marker]
            initial_pos = map.initial_pos == marker
            initial_pos = initial_pos.astype(np.int32) #exclude it if unnecesarry
            terminal_pos = map.terminal_pos == marker
            terminal_pos = terminal_pos.astype(np.int32)
            trajectory = map.trajectory[marker]

            for m in range(self.n_markers)[1:]:
                if m == marker:
                    continue
                block_pos = block_pos + map.initial_pos == m +  map.terminal_pos == m
            
            state[k] = block_pos
            state[k+1] = location_map
            state[k+2] = initial_pos
            state[k+3] = terminal_pos
            state[k+4] = trajectory

        return state

    
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

    def _move_marker(self, map: Map, marker: int, direction: int) -> Map:
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

    def _calculate_reward(self, old_map: Map, new_map: Map, marker:int) -> float:
        def T1Norm(terminal_coor, marker_coor):
            diff_x = abs(terminal_coor[0] - marker_coor[0])
            diff_y = abs(terminal_coor[1] - marker_coor[1])
            #if (diff_x+diff_y)%2 == 1:
                #raise AssertionError
            return max(diff_x+diff_y, 2*diff_y)

        def pt2lineDist(pt, line):
            def inLine(inst, line):
                return ((line[0][0]-inst[0])*(inst[0]-line[1][0]) >= 0) and ((line[0][1]-inst[1])*(inst[1]-line[1][1]) >= 0)

            x, y = pt[0], pt[1]
            left, right = line[0], line[1]
            if left[0] > right[0]:
                left, right = right, left

            if left[1] == right[1]:
                common = left[1]
                inst_1 = [x+(common-y), common]
                inst_2 = [x-(common-y), common]
                if inLine(inst_1, line) or inLine(inst_2, line):
                    return T1Norm(pt, inst_1)
                return min(T1Norm(pt,left), T1Norm(pt,right))

            if left[0]+left[1] == right[0]+right[1]:
                common = left[0]+left[1]
                inst_1 = [x+(common-x-y)/2, y+(common-x-y)/2]
                inst_2 = [common-y, y]
                if inLine(inst_1, line) or inLine(inst_2, line):
                    return T1Norm(pt, inst_1)
                return min(T1Norm(pt,left), T1Norm(pt,right))

            if left[0]-left[1] == right[0]-right[1]:
                common = left[0] - left[1]
                inst_1 = [x+common/2, y-common/2]
                inst_2 = [common+y, y]
                if inLine(inst_1, line) or inLine(inst_2, line):
                    return T1Norm(pt, inst_1)
                return min(T1Norm(pt,left), T1Norm(pt,right))
            raise AssertionError
        """
        Calculate the reward for the current step based on the map's current state.

        Args:
            old_map (Map): The map object containing all necessary state information before moving the marker.
            new_map (Map): The map object containing all necessary state information after moving the marker.
            marker (int): The marker index.

        Returns:
            float: Reward for the action.
        """
        reward = 0
        prev_marker_coor = np.argwhere(old_map.marker_pos[marker] == 1)[0] 
        cur_marker_coor = np.argwhere(new_map.marker_pos[marker] == 1)[0]
        direction = cur_marker_coor - prev_marker_coor
        terminal_coor = np.argwhere(new_map.terminal_pos == marker)[0]
        # prev_distance_to_terminal = np.sum(np.abs(terminal_coor - prev_marker_coor))
        # cur_distance_to_terminal = np.sum(np.abs(terminal_coor - cur_marker_coor[0]))
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
            return reward

        # Add leave target penalty
        if np.all(prev_marker_coor == terminal_coor) and np.any(cur_marker_coor != terminal_coor):
            reward += self.reward_dict["leave_target_penalty"]
            return reward

        # Add distance bonus for current position to terminal
        distance_reduce = (prev_distance_to_terminal - cur_distance_to_terminal)/2
        reward += self.reward_dict["distance_bonus"] * distance_reduce

        #----------------------------------------------------------------------------
        # Add distance bonus for current position to cluster
        # marker_center_coor = np.zeros(2)
        # terminal_center_coor = np.zeros(2)
        # for i in range(new_map.n_markers):
        #     if i != marker:
        #         marker_center_coor += np.argwhere(old_map.marker_pos[i] == 1)[0]
        #     # terminal_center_coor += np.argwhere(map.terminal_pos == i)[0]
        # # marker_center_coor += terminal_coor
        # marker_center_coor /= (new_map.n_markers - 1)
        # # prev_distance_to_center = np.sum(np.abs(marker_center_coor - prev_marker_coor))
        # # cur_distance_to_center = np.sum(np.abs(marker_center_coor - cur_marker_coor))
        # # distance_reduce = cur_distance_to_center - prev_distance_to_center
        # prev_distance_to_center = T1Norm(marker_center_coor, prev_marker_coor)
        # cur_distance_to_center = T1Norm(marker_center_coor, cur_marker_coor)
        # distance_reduce = (prev_distance_to_center - cur_distance_to_center)/2
        # # terminal_center_coor /= map.n_markers
        # # cur_distace_to_cluster = abs(marker_center_coor[0] - cur_marker_coor[0]) + abs(marker_center_coor[1] - cur_marker_coor[1])
        # # cluster_distance_to_terminal = abs(terminal_coor[0] - terminal_center_coor[0]) + abs(terminal_coor[1] - terminal_center_coor[1])
        # # distance_reduce_cluster = max(0, prev_distance_to_terminal - (cur_distace_to_cluster + cluster_distance_to_terminal))
        # # if np.dot(direction, marker_center_coor - prev_marker_coor) > 0:
        # reward += self.reward_dict["distance_bonus"] * distance_reduce

        #---------------------------------------------------------------------------
        # Add distance bonus for Trajectory distance bonus
        #current distance
        distance_record = []
        for m in range(self.n_markers)[1:]:
            if m == marker:
                continue
            trajectory = old_map.trajectory[m]
            trajectory_pt_pos = np.argwhere(trajectory == 1)
            
            for pt_posi in trajectory_pt_pos:
                distance_record.append(T1Norm(prev_marker_coor, pt_posi))
        # print(distance_record)
        if len(distance_record) != 0:
            prev_dis = min(distance_record)
        else:
            prev_dis = None

        #next distance
        distance_record = []
        for m in range(self.n_markers)[1:]:
            if m == marker:
                continue
            trajectory = new_map.trajectory[m]
            trajectory_pt_pos = np.argwhere(trajectory == 1)

            for pt_posi in trajectory_pt_pos:
                distance_record.append(T1Norm(cur_marker_coor, pt_posi))
        if len(distance_record) != 0:
            cur_dis = min(distance_record)
        else:
            cur_dis = None
        
        if prev_dis and cur_dis:
            distance_reduce = (prev_dis- cur_dis)/2
            reward += self.reward_dict["distance_bonus"] * distance_reduce

        #---------------------------------------------------------------------------
        
        # Add reward for merging (if multiple markers are in the same cell)
        prev_merge_num = np.max(np.sum(old_map.marker_pos, axis=0))  # Shape: (height, width)
        cur_merge_num = np.max(np.sum(new_map.marker_pos, axis=0))  # Shape: (height, width)
        merge_num_delta = cur_merge_num - prev_merge_num
        if merge_num_delta > 0 and np.dot(direction, terminal_coor - prev_marker_coor) > 0:
            reward += self.reward_dict["distance_bonus"] * (new_map.n_markers) * prev_merge_num
            # reward += self.reward_dict["merge_reward"] * prev_merge_num

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
        return {"map": map.get_map(), "steps_taken": self.step_count}
    
    def reset(self, seed: int = None):
        self.step_count = 0
        self.map = self._load_map(self._map_file)

        info = self._construct_info(self.map)

        return self._construct_state(self.map), info

    def step(self, action: tuple):
        self.step_count += 1

        marker, direction = self._decode_action(action)
        self.map.marker_pos = self.map.marker_pos
        old_map = self.map.copy()
        self.map = self._move_marker(self.map, marker, direction)
        next_state = self._construct_state(self.map)
        done = self._check_finish(self.map)
        reward = self._calculate_reward(old_map, self.map, marker)

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
