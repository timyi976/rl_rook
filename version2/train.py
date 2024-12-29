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

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_id", type=str, default="cnn_add_merge")
    parser.add_argument("--epoch_num", type=int, default=500)
    parser.add_argument("--timesteps_per_epoch", type=int, default=100)
    parser.add_argument("--eval_episode_num", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--env", type=str, default="RookEnv_v5")
    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--max_save", type=int, default=40)
    parser.add_argument("--map_file", type=str, default="./tasks/base_v1.txt")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--policy_net", type=str, default="CnnPolicy")
    parser.add_argument("--gif_strategy", type=str, default="best", choices=["every", "best"])

    return parser.parse_args()

def train(model, epoch_num, timesteps_per_epoch, eval_episode_num, eval_env, save_path, gif_strategy="every"):
    current_best_reward = float("-inf")

    for epoch in tqdm(range(epoch_num)):

        model.learn(
            total_timesteps=timesteps_per_epoch,
            reset_num_timesteps=False,
            callback=WandbCallback(
                # gradient_save_freq=100,
                verbose=2,
            ),
        )

        avg_score = eval_many_episodes(eval_env, model, eval_episode_num)

        

        # if gif_strategy == "every":
        _, step = infer_and_visualize(eval_env, model, f"chess_animation.gif")

        wandb.log(
            {"avg_score": avg_score,
             "step": step}
        )

        if avg_score > current_best_reward:
            current_best_reward = avg_score
            best_ckpts.append(epoch)
            # if gif_strategy == "best":
            #     infer_and_visualize(eval_env, model, save_path+"chess_animation.gif")
            print(f"Epoch {epoch}: New best reward: {current_best_reward}")

        save_model_with_limit(model, save_path, epoch)

def save_model_with_limit(model, save_path, epoch, max_ckpts=40):
    """Save model with limit on number of checkpoints"""
    # create save_path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    ckpts = sorted([int(ckpt.split(".")[0]) for ckpt in os.listdir(save_path)])
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
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of the flattened output
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # Define a fully connected layer to reduce to the desired feature dimension
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Pass through CNN and then the fully connected layer
        return self.linear(self.cnn(observations))

def main():
    args = arg_parse()

    # register gym env
    warnings.filterwarnings("ignore")
    # Create wandb session (Uncomment to enable wandb logging)
    run = wandb.init(
        project="Rook",
        # config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id="COG"
    )
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
        env = gym.make('Rook-v0', map_file=args.map_file, max_steps=args.timesteps_per_epoch, reward_dict = {
            "step_penalty": -1,               # -1,Penalty for every step to encourage efficiency.
            "blockage_penalty": 0,           # -3,Penalty for moving into a blocked cell.
            "stay_penalty": -1,               # -3,Penalty for failing to move (hitting a wall or staying in place).
            "source_collision_penalty": -1,   # -3,Penalty for moving into another marker's initial position.
            "target_collision_penalty": -1,   # -3,Penalty for moving into another marker's target position.
            "merge_reward": 0.5,               # 20,Reward for merging markers or transitioning into special states.
            "reached_target": 5,             # 10,Reward for reaching the correct target.
            "leave_target_penalty": -5,      # -20,Penalty for leaving a marker's target after reaching it.
            "distance_bonus": 0.2,              # 1,Scaled reward for reducing the Manhattan distance to the target.
        }
        )
        return env
    
    train_env = DummyVecEnv([make_env for _ in range(args.n_envs)]) 
    # eval_env = DummyVecEnv([make_env])
    eval_env = make_env()

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


    if not os.path.exists("model"):
        os.makedirs("model")
    train(model, args.epoch_num, args.timesteps_per_epoch, args.eval_episode_num, eval_env, f"model/{args.run_id}_{args.map_file[8:-4]}_", args.gif_strategy)

if __name__ == "__main__":
    main()
