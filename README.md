# rl_rook
RL Course Final Project, 2024 Fall 

Version 3:
To reproduce the experimental results for version 3, please follow the steps below:
* 1. Navigate to the directory: Go to the "version3" folder.
* 2. Create the Conda environment: Run "conda env create -f environment.yml" to build the Conda environment.
* 3. Training (Retraining Models): If you plan to retrain the models from scratch, follow the sub-steps below:<br>
  * 3-1. Exp. A (version3 part): Simply run "python train_v6_wandb.py".<br>
  * 3-2. Exp. B:
   * 3-2-1: Open "rook_v6.py".<br>
   * 3-2-2: For the "full" case: Do nothing.<br>
   * 3-2-3: For the "old_dist_reward" case: Uncomment line 340 and comment lines 342 to 350.<br>
   * 3-2-4: For the "no_traj" case: Comment lines 404 to 409.<br>
   * 3-2-5: For the "no_keep_choosing_one_marker" case: Comment lines 451 to 453.<br>
   * 3-2-6: In "train_v6_wandb.py", set the correct run-id on line 32.<br>
   * 3-2-7: Run "python train_v6_wandb.py".<br>
  * 3-3. Exp. C: In "train_v6_wandb.py", set the algorithm on line 38 to "PPO", "TRPO", or "A2C". Then run "python train_v6_wandb.py".<br>
  * 3-4. Exp. D:<br>
     * 3-4-1: In "train_v6_wandb.py", set "map_file_dir" on line 40 to "tasks123456".<br>
     * 3-4-2: To use a CNN feature extractor: Set the "policy_net" on line 44 to "CnnPolicy" and uncomment line 359.<br>
     * 3-4-3: Run "python train_v6_wandb.py".<br>
  * 3-5. Exp. E:<br>
     * 3-5-1: In "train_v6_wandb.py", set "map_file_dir" on line 40 to "tasks_same_t_train".<br>
     * 3-5-2: To use a CNN feature extractor: Set the "policy_net" on line 44 to "CnnPolicy" and uncomment line 359.<br>
     * 3-5-3: Run "python train_v6_wandb.py".<br>
  * Note on Plotting GIF Inferences: 
    * To plot GIF inferences from the best model at each epoch during training, refer to lines 99–109 in "train_v6_wandb.py".
    * For Exp. A to D, you only need to plot the GIF using "seen_eval_env".
    * For Exp. E, you only need to use "unseen_eval_env".

* 4. Inference (Using Pre-trained Models): If you wish to reproduce the results directly using pre-trained models:
   * 4-1: Download the trained models from the following link: [https://drive.google.com/file/d/1AqJmkxnk13xylLxdtDEMvuPO8UmOEKuH/view?usp=drive_link](https://drive.google.com/drive/folders/1kh4oMNmzemlLqXdaESgoDLzg39LbHPNa?usp=drive_link)
   * 4-2: Open "eval.py", and set the model path and test_folder correctly (refer to Section 3 for specific parameter settings).
   * 4-3: Run "python eval.py". Then the gif is saved as "chess_animation.gif"

