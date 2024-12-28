import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import argparse

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3 import A2C, DQN, PPO, SAC

from tqdm import tqdm
import os

from collections import deque

from eval import eval_many_episodes, infer_and_visualize

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.callbacks import BaseCallback
from torch.optim.lr_scheduler import LambdaLR

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_id", type=str, default="mixed_cnn_tasks6_v2")
    parser.add_argument("--epoch_num", type=int, default=1500)
    parser.add_argument("--timesteps_per_epoch", type=int, default=1000)
    parser.add_argument("--eval_episode_num", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1.06e-4) #1.05e-4
    parser.add_argument("--env", type=str, default="RookEnv_v6")
    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--max_save", type=int, default=3)
    parser.add_argument("--map_file_dir", type=str, default="./tasks6")
    parser.add_argument("--val_map_file_dir", type=str, default="./tasks_same_t_val")
    parser.add_argument("--map_file", type=str, default="./tasks/custom_v2.txt")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--policy_net", type=str, default="CnnPolicy")
    parser.add_argument("--gif_strategy", type=str, default="best", choices=["every", "best", "none"])

    return parser.parse_args()

# Custom Callback for Scheduler
class SchedulerCallback(BaseCallback):
    def __init__(self, optimizer, total_timesteps, warmup_fraction=0.1):
        super(SchedulerCallback, self).__init__()
        self.optimizer = optimizer
        self.total_timesteps = total_timesteps
        self.warmup_fraction = warmup_fraction
        self.scheduler = None

    def _on_training_start(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_fraction * self.total_timesteps:
                return current_step / (self.warmup_fraction * self.total_timesteps)
            return 1.0
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _on_step(self) -> bool:
        # Update the scheduler after each step
        self.scheduler.step()
        return True

def train(model, epoch_num, timesteps_per_epoch, eval_episode_num, seen_eval_env, unseen_eval_env, save_path, gif_strategy="every"):
    current_best_reward = float("-inf")
    with open(os.path.join(save_path,"record.txt"), "w") as f:
        for epoch in tqdm(range(epoch_num)):

            model.learn(
                total_timesteps=timesteps_per_epoch,
                reset_num_timesteps=False,
            )
            seen_avg_score = eval_many_episodes(seen_eval_env, model, eval_episode_num)
            unseen_avg_score = eval_many_episodes(unseen_eval_env, model, eval_episode_num)

            if gif_strategy == "every":
                infer_and_visualize(eval_env, model, f"chess_animation.gif")

            if seen_avg_score > current_best_reward:
                current_best_reward = seen_avg_score
                best_ckpts.append(epoch)
                if gif_strategy == "best" and seen_avg_score>7:
                    infer_and_visualize(seen_eval_env, model, save_path+"chess_animation1.gif")
                    infer_and_visualize(seen_eval_env, model, save_path+"chess_animation2.gif")
                    infer_and_visualize(seen_eval_env, model, save_path+"chess_animation3.gif")
                    infer_and_visualize(seen_eval_env, model, save_path+"chess_animation4.gif")
                    infer_and_visualize(seen_eval_env, model, save_path+"chess_animation5.gif")
                    infer_and_visualize(unseen_eval_env, model, save_path+"chess_animation6.gif")
                    infer_and_visualize(unseen_eval_env, model, save_path+"chess_animation7.gif")
                    infer_and_visualize(unseen_eval_env, model, save_path+"chess_animation8.gif")
                    infer_and_visualize(unseen_eval_env, model, save_path+"chess_animation9.gif")
                    infer_and_visualize(unseen_eval_env, model, save_path+"chess_animation10.gif")
                print(f"Epoch {epoch}: New best seen reward: {current_best_reward}; Current unseen reward: {unseen_avg_score}")
                f.write(f"Epoch {epoch}: New best seen reward: {current_best_reward}; Current unseen reward: {unseen_avg_score}\n")
                save_model_with_limit(model, save_path, epoch)

def save_model_with_limit( model, save_path, epoch, max_ckpts=3):
    """Save model with limit on number of checkpoints"""
    # create save_path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    ckpts = sorted([int(ckpt.split(".")[0]) for ckpt in os.listdir(save_path) if ckpt.endswith('.zip')])
    n_to_remove = len(ckpts) - max_ckpts + 1
    if n_to_remove > 0:
        for ckpt in ckpts:
            if n_to_remove <= 0:
                break
            if ckpt not in best_ckpts:
                os.remove(f"{save_path}/{ckpt}.zip")
                n_to_remove -= 1

    model.save(f"{save_path}/{epoch}")

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, kernel_size: int = 3):
        super().__init__(observation_space, features_dim)

        # Number of input channels (e.g., n_markers + 3 layers)
        n_input_channels = observation_space.shape[0]

        # Define the CNN layers
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=kernel_size // 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample spatial dimensions
            
        #     nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=kernel_size // 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample spatial dimensions

        #     nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=kernel_size // 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample spatial dimensions

        #     nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=kernel_size // 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample spatial dimensions

        #     nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=kernel_size // 2),
        #     nn.ReLU()
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Further downsample spatial dimensions

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=kernel_size // 2),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=kernel_size // 2),
            nn.ReLU()
        )

        # Calculate the size of the flattened output
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            # n_flatten = self.cnn(sample_input).shape[1]
            n_flatten = self.cnn(sample_input).view(-1).shape[0]

        # Define a fully connected layer to reduce to the desired feature dimension
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Pass through CNN and then the fully connected layer
        x = self.cnn(observations)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear(x)
        return x
        # return self.linear(self.cnn(observations))


# class CustomCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, kernel_size: int = 3):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
        
#         # Number of input channels
#         n_input_channels = observation_space.shape[0]

#         # Convolutional layers
#         self.cnn = nn.Sequential(
#             # First convolution (preserves resolution)
#             nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),

#             # Second convolution with stride=2 to downsample moderately
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 4 x 6
#             nn.ReLU(),

#             # Third convolution with dilation to expand the receptive field
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2),  # Output: 4 x 6
#             nn.ReLU(),

#             # Fourth convolution (preserves resolution)
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Output: 4 x 6
#             nn.ReLU()
#         )

#         # Calculate the flattened size dynamically
#         with torch.no_grad():
#             sample_input = torch.as_tensor(observation_space.sample()[None]).float()
#             n_flatten = self.cnn(sample_input).view(-1).shape[0]

#         # Fully connected layer to reduce to the desired feature dimension
#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU()
#         )

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         # Pass through CNN layers
#         x = self.cnn(observations)
#         # Flatten and pass through the fully connected layer
#         x = x.view(x.size(0), -1)
#         return self.linear(x)


def main():
    args = arg_parse()

    print(f"I am running {args.run_id}")

    # register gym env
    warnings.filterwarnings("ignore")
    register(
        id='Rook-v0',
        entry_point=f'envs:{args.env}'
    )

    global best_ckpts
    best_ckpts = deque(maxlen=args.max_save)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    algorithm = eval(args.algorithm)

    def make_env():
        # TODO: define reward_dict
        env = gym.make('Rook-v0', mode="train",map_file_dir=args.map_file_dir, reward_dict = {
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
        return env
    
    def make_seen_eval_env():
        # TODO: define reward_dict
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
        return env

    def make_unseen_eval_env():
        # TODO: define reward_dict
        env = gym.make('Rook-v0', mode="eval",map_file_dir=args.val_map_file_dir, reward_dict = {
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
        return env
    
    train_env = DummyVecEnv([make_env for _ in range(args.n_envs)]) 
    # eval_env = DummyVecEnv([make_env])
    seen_eval_env = make_seen_eval_env()
    unseen_eval_env = make_unseen_eval_env()

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256, kernel_size=3),  # Customize kernel size
        normalize_images=False  # Disable internal normalization
    )

    model = algorithm(
        args.policy_net,
        train_env,
        verbose=0,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
        # policy_kwargs={"normalize_images": False}
        device=device
    )

    # # Add scheduler callback
    # optimizer = model.policy.optimizer
    # total_timesteps = args.epoch_num * args.timesteps_per_epoch
    # scheduler_callback = SchedulerCallback(optimizer, total_timesteps, warmup_fraction=0.1)


    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(f"models/{args.run_id}"):
        os.makedirs(f"models/{args.run_id}")
    train(model, args.epoch_num, args.timesteps_per_epoch, args.eval_episode_num, seen_eval_env, unseen_eval_env, f"models/{args.run_id}", args.gif_strategy)

if __name__ == "__main__":
    main()
