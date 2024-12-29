from rook3 import RookEnv
import argparse


env = RookEnv("../tasks/base_v1.txt", {
            "step_penalty": -1,               # -1,Penalty for every step to encourage efficiency.
            "blockage_penalty": 0,           # -3,Penalty for moving into a blocked cell.
            "stay_penalty": -0.1,               # -3,Penalty for failing to move (hitting a wall or staying in place).
            "source_collision_penalty": -1,   # -3,Penalty for moving into another marker's initial position.
            "target_collision_penalty": -1,   # -3,Penalty for moving into another marker's target position.
            "merge_reward": 0.6,               # 20,Reward for merging markers or transitioning into special states.
            "reached_target": 5,             # 10,Reward for reaching the correct target.
            "leave_target_penalty": -5,      # -20,Penalty for leaving a marker's target after reaching it.
            "distance_bonus": 0.2,              # 1,Scaled reward for reducing the Manhattan distance to the target.
        })

ret = env.reset()
print(ret)

# env.render()
# print("-------------------")
obs, reward, done, truncated, info = env.step(13)
print(obs)
print(obs.shape)
# env.render()
# print(info["map"]["block_pos"])
# print("-------------------")
# obs, reward, done, truncated, info = env.step(13)
# # env.render()
# print(info)
# print("-------------------")
# obs, reward, done, truncated, info = env.step(13)
# # env.render()
# print(info)
# print("-------------------")
# obs, reward, done, truncated, info = env.step(13)
# # env.render()
# print(info)
# print("-------------------")
# obs, reward, done, truncated, info = env.step(13)
# # env.render()
# print(info)
# print("-------------------")
# print(obs, reward, done, truncated, info)