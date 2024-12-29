# rl_3chess
RL Course Final Project, 2024 Fall

## Map file entry type

### Definition

- $n$: the number of markers
- $0$: empty cell
- $1$ ~ $2^n-1$: markers location in binary representation
- $2^n$ ~ $2^n+n-1$: markers' initial location
- $2^n+n$ ~ $2^n+2n-1$: markers' terminal location
- $2^n+2n$: block
- Some extra rules
    - If a marker locates at its initial location
        - no initial location is given for this marker
    - If a marker locates at its terminal location
        - no termination location is given for this marker

### Example

For 3-markers case:
- $n=3$
- $0$ empty cell
- $1$ ~ $7$: markers location in binary representation
    - For example, if a position has the the entry $3$, then it means marker 0 and marker 1 are both at this location, because $3 = 011_2$.
- $8$ ~ $10$: initial location of markers 0, 1, 2 respectively
- $11$ ~ $13$: terminal location of markers 0, 1, 2 respectively
- $14$: block

Following is the `base_v1.txt` map for reference:
```
3
0 0 2 0 4 0 14 0 0 0 0
0 1 0 0 0 14 0 0 0 0 0
0 0 0 0 0 0 14 0 14 0 14
0 0 0 0 0 0 0 0 0 0 0
0 0 14 0 0 0 0 0 14 0 0
0 0 0 0 0 0 0 0 0 12 0
0 0 0 0 0 0 11 0 13 0 0
```
Note that $8$ ~ $10$ does not show on this map, because all the markers are at their initial location respectively.

### Why this design
- Any possible states can be represented uniquely, even when some of the markers still at their initial location or already at the terminal location
    - so we can actually start with any possible states
- Any number of markers is possible

## Actions

### Move only one marker at a time

- Action space: a discrete space in the size of $n\times 6$
- Moving directions
    ```
    (4)  \       / (5)
    (3) --       -- (0)
    (2)  /       \ (1)
    ```
- Action `i` means
    - Moving marker `i // 6` toward `i % 6` direction
    - For example, action `1` means moving marker 0 toward direction 1

## How to try different reward?
- Use your editor and search for "TODO" in `train.py` and `envs/rook.py`
- Edit those reward related parts, including
    - TODO in `train.py`
    - TODO in `RookEnv._calculate_reward()` of `envs/rook.py`

## How to try different format of state?
- Use your editor and search for "TODO" in `envs/rook.py`
- Edit those reward related parts, including
    - TODO in `RookEnv.__init__()` of `envs/rook.py`
    - TODO in `RookEnv._construct_state()` of `envs/rook.py`

## How to log anything after each step?
- Use your editor and search for "TODO" in `envs/rook.py`
- Edit following related parts, including
    - TODO in `RookEnv._construct_info()` in `envs/rook.py`

## How to create another environment?
- Copy `envs/rook.py`, and edit that copied file
- Edit `envs/__init__.py`, and import the new environment class with a different name using `import [class] from [file] as [name]` syntax
- Use `--env` argument in `train.py` to designate the environment to use using the name

## How to run the training script?

- Following will use the default arguments
    ```bash
    python3 train.py
    ```
- For customized arguments, check following usage example:
    ```
    usage: python3 train.py [-h] [--run_id RUN_ID] [--epoch_num EPOCH_NUM] [--timesteps_per_epoch TIMESTEPS_PER_EPOCH] [--eval_episode_num EVAL_EPISODE_NUM] [--learning_rate LEARNING_RATE][--env ENV] [--algorithm ALGORITHM] [--max_save MAX_SAVE] [--map_file MAP_FILE] [--n_envs N_ENVS] [--policy_net POLICY_NET]

    options:
    -h, --help            show this help message and exit
    --run_id RUN_ID
    --epoch_num EPOCH_NUM
    --timesteps_per_epoch TIMESTEPS_PER_EPOCH
    --eval_episode_num EVAL_EPISODE_NUM
    --learning_rate LEARNING_RATE
    --env ENV
    --algorithm ALGORITHM
    --max_save MAX_SAVE
    --map_file MAP_FILE
    --n_envs N_ENVS
    --policy_net POLICY_NET
    --gif_strategy {every,best}
    ```


## Memo

### **T1Norm Function**
This function calculates the T1 norm of a given point. (By Nicky)

```cpp
double T1Norm(const Dpt& a) {
    double p = abs(a.x);
    double q = abs(a.y) * 3.0 / 5.0;
    return max(p + q, 2 * q);
}

Dpt findInterOpt(Dpt& s, Dpt& p, Dpt& q, Dpt& v, Dpt& w) {
    vector<Dpt> insect_s;
    if (q.x > p.x) {
        insect_s = {
            p, q, v, w, 
            Dpt(p.x + (3.0 / 5.0) * (s.y - p.y), s.y), 
            Dpt((s.x + p.x + (3.0 / 5.0) * (s.y - p.y)) / 2.0, 
                (s.y + p.y + (5.0 / 3.0) * (s.x - p.x)) / 2.0), 
            Dpt(s.x + (v.y - s.y) * 3.0 / 5.0, v.y), 
            Dpt(s.x + (v.y - s.y) * -3.0 / 5.0, v.y)
        };
    } else {
        insect_s = {
            p, q, v, w, 
            Dpt(p.x + (-3.0 / 5.0) * (s.y - p.y), s.y), 
            Dpt((s.x + p.x + (-3.0 / 5.0) * (s.y - p.y)) / 2.0, 
                (s.y + p.y + (-5.0 / 3.0) * (s.x - p.x)) / 2.0), 
            Dpt(s.x + (v.y - s.y) * 3.0 / 5.0, v.y), 
            Dpt(s.x + (v.y - s.y) * -3.0 / 5.0, v.y)
        };
    }

    // Return the intersection point with the minimum length, adhering to routing constraints.
    return *min_element(
        insect_s.begin(), 
        insect_s.end(), 
        [&s, &p, &q, &v, &w](const Dpt& a, const Dpt& b) {
            const double len_a = 
                ((a.y != v.y && p.y <= a.y && a.y <= q.y) || 
                (a.y == v.y && v.x <= a.x && a.x <= w.x)) 
                ? T1Norm(s - a) : inf;

            const double len_b = 
                ((b.y != v.y && p.y <= b.y && b.y <= q.y) || 
                (b.y == v.y && v.x <= b.x && b.x <= w.x)) 
                ? T1Norm(s - b) : inf;

            return len_a < len_b;
        }
    );
}

