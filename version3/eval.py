import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC

###by Nicky
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import imageio
import os
from collections import defaultdict
from matplotlib.patches import Wedge
import argparse

#by Harry
from matplotlib.patches import Wedge, Patch
import io

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="RookEnv_v6")
    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--map_file", type=str, default="./tasks/base_v1.txt")
    parser.add_argument("--map_file_dir", type=str, default="./tasks1")

    return parser.parse_args()

def infer_one_epsidoe(env, model):
    done = False
    obs, info = env.reset()

    traces = []
    actions = []
    rewards = []
    while not done:
        # action, _state = model.predict(obs, deterministic=True)
        action, _state = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        matrix = info['map']['trajectory'][0]+info['map']['trajectory'][1]+info['map']['trajectory'][2]
        wire_length = info["wire_length"]

        traces.append(obs)
        actions.append(action)
        rewards.append(reward)

    return traces, actions, rewards,wire_length

def eval_many_episodes(env, model, num_episodes=100):
    avg_score = 0
    wire_lengths = 0

    for _ in range(num_episodes):
        traces, actions, rewards,wire_length = infer_one_epsidoe(env, model)
        avg_score += sum(rewards)
        wire_lengths += wire_length

    avg_score /= num_episodes
    wire_lengths /= num_episodes

    return avg_score,wire_lengths

### by Nicky
def process_pos(pos, Height, Width):
    processed_pos = []
    for layer in pos:
        new_layer = []
        for y in range(Height):
            row = layer[y]
            new_row = []
            for x in range(Width):
                if (x + y)%2 != 1:
                    new_row.append(row[x])
            # 如果新行长度不足 Width // 2，填充 0
            while len(new_row) < (Width+1) // 2:
                new_row.append(0)
            new_layer.append(new_row)
        processed_pos.append(new_layer)
    return processed_pos, Height, (Width+1) // 2

# def draw_board(pos, Height, Width, Num, colors, save_path):
#     a = 1
#     x_spacing = a
#     y_spacing = a * np.sqrt(3) / 2

#     x_coords = np.arange(Width)
#     y_coords = np.arange(Height)
#     X, Y = np.meshgrid(x_coords, y_coords)

#     X = X.astype(float)
#     X += (Y % 2) * (x_spacing / 2)

#     X_positions = X * x_spacing
#     Y_positions = Y * y_spacing

#     position_dict = defaultdict(list)

#     for idx, layer in enumerate(pos):
#         layer = np.array(layer)
#         positions = np.argwhere(layer > 0)
#         for pos_idx in positions:
#             row, col = pos_idx
#             x_pos = col + (row % 2) * 0.5
#             position_key = (x_pos, row)
#             position_dict[position_key].append(idx + 1)

#     plt.figure(figsize=(Width, Height*0.9))

#     plt.scatter(X_positions.flatten(), Y_positions.flatten(), color='lightgray', marker='o', s=100)

#     ax = plt.gca()

#     for position_key, idx_list in position_dict.items():
#         x_pos, row = position_key
#         x_plot = x_pos * x_spacing
#         y_plot = row * y_spacing

#         if len(idx_list) == 1:
#             idx = idx_list[0] - 1
#             ax.add_patch(plt.Circle((x_plot, y_plot), 0.12 * a, color=colors[idx+1], zorder=2))
#         else:
#             num_pieces = len(idx_list)
#             angle_step = 360 / num_pieces
#             start_angle = 0
#             for idx in idx_list:
#                 wedge = Wedge(center=(x_plot, y_plot), r=0.12 * a, theta1=start_angle, theta2=start_angle + angle_step,
#                               facecolor=colors[idx], edgecolor='black', lw=0.5, zorder=2)
#                 ax.add_patch(wedge)
#                 start_angle += angle_step

#     plt.xlim(-a, Width * x_spacing + a)
#     plt.ylim(-a, Height * y_spacing + a)
#     plt.gca().invert_yaxis()
#     plt.axis('equal')
#     plt.xticks([])
#     plt.yticks([])

#     from matplotlib.patches import Patch
#     # print(len(colors),colors)
#     legend_elements = [Patch(facecolor=colors[i+1], edgecolor='black', label=f'Chess {i+1}') for i in range(Num)]
#     plt.legend(handles=legend_elements, fontsize='small', bbox_to_anchor=(0.92,1), loc='upper left')

#     plt.title('chessLayout')

#     plt.savefig(save_path)
#     plt.close()

# def single_visualize(traces, gif_path="chess_animation.gif"):
#     Num     = len(traces[0])
#     # print("single visualize iter loops number",len(traces))
#     # print("Num",Num,"traces",traces)
#     Height  = len(traces[0][0])
#     Width   = len(traces[0][0][0])
#     # print(Num, Width, Height)
#     # colors = ['white', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray']
#     colors = ['white', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'teal', 'gold','lime','turquoise']
#     image_files = []
#     #output_dir = 'C:/Users/harry/Big_data_storage/RL_final_project/model/temp_images'
#     output_dir = 'temp_images'
#     os.makedirs(output_dir, exist_ok=True)

#     for idx, pos in enumerate(traces):
#         # print("single visualize",idx)
#         image_path = os.path.join(output_dir, f'board_{idx:03d}.png')
#         image_files.append(image_path)
        
#         # print("process_pos")
#         processed_pos, new_Height, new_Width = process_pos(pos, Height, Width)
#         # print("draw board")
#         draw_board(processed_pos, new_Height, new_Width, Num, colors, image_path)

#     images = []
#     for filename in image_files:
#         images.append(imageio.imread(filename))

#     # gif_path = 'chess_animation.gif'

#     imageio.mimsave(gif_path, images, duration=2, loop=0)

#     for filename in image_files:
#         os.remove(filename)
#     os.rmdir(output_dir)

#     print(f"GIF is saved as {gif_path}")

def draw_board_to_buffer(pos, Height, Width, Num, colors):
    """
    Draw the board and return it as an in-memory buffer image.
    """
    a = 1
    x_spacing = a
    y_spacing = a * np.sqrt(3) / 2

    x_coords = np.arange(Width)
    y_coords = np.arange(Height)
    X, Y = np.meshgrid(x_coords, y_coords)

    X = X.astype(float)
    X += (Y % 2) * (x_spacing / 2)

    X_positions = X * x_spacing
    Y_positions = Y * y_spacing

    position_dict = defaultdict(list)

    for idx, layer in enumerate(pos):
        layer = np.array(layer)
        positions = np.argwhere(layer > 0)
        for pos_idx in positions:
            row, col = pos_idx
            x_pos = col + (row % 2) * 0.5
            position_key = (x_pos, row)
            position_dict[position_key].append(idx + 1)

    plt.figure(figsize=(Width, Height * 0.9))

    plt.scatter(X_positions.flatten(), Y_positions.flatten(), color='lightgray', marker='o', s=100)

    ax = plt.gca()

    for position_key, idx_list in position_dict.items():
        x_pos, row = position_key
        x_plot = x_pos * x_spacing
        y_plot = row * y_spacing

        if len(idx_list) == 1:
            idx = idx_list[0] - 1
            ax.add_patch(plt.Circle((x_plot, y_plot), 0.12 * a, color=colors[idx + 1], zorder=2))
        else:
            num_pieces = len(idx_list)
            angle_step = 360 / num_pieces
            start_angle = 0
            for idx in idx_list:
                wedge = Wedge(center=(x_plot, y_plot), r=0.12 * a, theta1=start_angle, theta2=start_angle + angle_step,
                              facecolor=colors[idx], edgecolor='black', lw=0.5, zorder=2)
                ax.add_patch(wedge)
                start_angle += angle_step

    plt.xlim(-a, Width * x_spacing + a)
    plt.ylim(-a, Height * y_spacing + a)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])

    legend_elements = [Patch(facecolor=colors[i + 1], edgecolor='black', label=f'Chess {i + 1}') for i in range(Num)]
    plt.legend(handles=legend_elements, fontsize='small', bbox_to_anchor=(0.92, 1), loc='upper left')

    plt.title('chessLayout')

    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def single_visualize(traces, gif_path="chess_animation.gif"):
    """
    Create a GIF directly from traces without saving intermediate images to disk.
    """
    Num = len(traces[0])
    Height = len(traces[0][0])
    Width = len(traces[0][0][0])
    colors = ['white', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'teal', 'gold', 'lime', 'turquoise']

    with imageio.get_writer(gif_path, mode='I', duration=4,loop=0) as writer:
        for pos in traces:
            # print("hhh")
            processed_pos, new_Height, new_Width = process_pos(pos, Height, Width)
            buf = draw_board_to_buffer(processed_pos, new_Height, new_Width, Num, colors)
            image = imageio.imread(buf)
            writer.append_data(image)

    print(f"GIF is saved as {gif_path}")


def infer_and_visualize(env, model, gif_path="chess_animation.gif"):
    done = False
    obs, info = env.reset()

    traces = []
    #actions = []
    rewards = 0

    while not done:
        # action, _state = model.predict(obs, deterministic=True)
        action, _state = model.predict(obs)
        obs, reward, done, _, info = env.step(action)

        marker_pos = info['map']['marker_pos']  # (3, 7, 11)
        initial_pos = info['map']['initial_pos']  # (7, 11)
        terminal_pos = info['map']['terminal_pos']  # (7, 11)
        block_pos = info['map']['block_pos']  # (7, 11)
        trajectory = info['map']['trajectory'][0]+info['map']['trajectory'][1]+info['map']['trajectory'][2] #(3,7,11)

        initial_pos_expanded = np.expand_dims(initial_pos, axis=0)  # (1, 7, 11)
        terminal_pos_expanded = np.expand_dims(terminal_pos, axis=0)  # (1, 7, 11)
        block_pos_expanded = np.expand_dims(block_pos, axis=0)  # (1, 7, 11)
        trajectory_expanded = np.expand_dims(trajectory, axis=0)
        terminal_pos_new = terminal_pos_expanded + np.ones(terminal_pos_expanded.shape, dtype='int32')

        multi = np.concatenate(
            (marker_pos, initial_pos_expanded, terminal_pos_new, block_pos_expanded, trajectory_expanded), axis=0)

        # if not done:
        rewards = np.sum(trajectory>0)
        traces.append(multi)
        #actions.append(action)

    ###
    print("Visualized game's score:  ", rewards)
    single_visualize(traces, gif_path)
    ###
    return traces, rewards
    #return traces, actions, rewards


def visualize_trace(env, model, save_path=[]):
    trace, actions, rewards = infer_one_epsidoe(env, model)
    visualize_trace(trace, save_path)

    return trace, actions, rewards

if __name__ == "__main__":
    args = parse_arg()
    register(
        id='Rook-v0',
        entry_point=f'envs:{args.env}'
    )

    model_path = "./260"
    # env = gym.make('Rook-v0', map_file_dir=args.map_file_dir, reward_dict={
    #     "step_penalty": -1,               # -1,Penalty for every step to encourage efficiency.
    #     "blockage_penalty": 0,           # -3,Penalty for moving into a blocked cell.
    #     "stay_penalty": -2,               # -3,Penalty for failing to move (hitting a wall or staying in place).
    #     "source_collision_penalty": -1,   # -3,Penalty for moving into another marker's initial position.
    #     "target_collision_penalty": -1,   # -3,Penalty for moving into another marker's target position.
    #     "merge_reward": 0.6,               # 20,Reward for merging markers or transitioning into special states.
    #     "reached_target": 10,             # 10,Reward for reaching the correct target.
    #     "leave_target_penalty": -20,      # -20,Penalty for leaving a marker's target after reaching it.
    #     "target_distance_bonus": 0.2,              # 1,Scaled reward for reducing the Manhattan distance to the target.
    #     "merge_bonus": 0.2,
    # })
    env = gym.make('Rook-v0', mode="eval",map_file_dir=args.map_file_dir, reward_dict = {
            "step_penalty": -1,               # -1,Penalty for every step to encourage efficiency.
            "blockage_penalty": 0,           # -3,Penalty for moving into a blocked cell.
            "stay_penalty": -2,               # -3,Penalty for failing to move (hitting a wall or staying in place).
            "source_collision_penalty": -1,   # -3,Penalty for moving into another marker's initial position.
            "target_collision_penalty": -1,   # -3,Penalty for moving into another marker's target position.
            "merge_reward": 0.6,               # 20,Reward for merging markers or transitioning into special states.
            "reached_target": 10,             # 10,Reward for reaching the correct target.
            "leave_target_penalty": -20,      # -20,Penalty for leaving a marker's target after reaching it.
            "target_distance_bonus": 0.2,              # 1,Scaled reward for reducing the Manhattan distance to the target.
            "merge_bonus": 0.2,
        }
        )

    algorithm = eval(args.algorithm)

    model = algorithm.load(model_path)

    # a,b,c,d = infer_one_epsidoe(env,model)

    # print(a,b,c,d)

    infer_and_visualize(env, model)

    # eval_num = 3
    # avg_sc = eval_many_episodes(env, model)
    # print("Avg_score:  ", avg_sc)
