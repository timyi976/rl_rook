import numpy as np
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_markers", type=int, default=3)
    parser.add_argument("--shape_x", type=int, default=7)
    parser.add_argument("--shape_y", type=int, default=11)
    parser.add_argument("--output_file", type=str, default="map.txt") # Output file name, only valid when --n_maps is 1
    parser.add_argument("--output_prefix", type=str, default="./tasks_same_t_full/base_v") # Output file prefix, only valid when --n_maps is greater than 1
    parser.add_argument("--n_maps", type=int, default=515)

    return parser.parse_args()

def check_valid_cell(x, y):
    if x % 2 == 0 and y % 2 == 0:
        return True
    if x % 2 == 1 and y % 2 == 1:
        return True
    return False

def generate_map(n_markers, shape):
    map_array = np.zeros(shape, dtype=int)

    # randomly assign targets 11, 12, 13, 14, ... to available cells
    targets = [11 + i for i in range(n_markers)]

    map_array[6,4] = 11
    map_array[4,8] = 12
    map_array[6,8] = 13
    map_array[0,6] = 14
    map_array[1,5] = 14
    map_array[2,8] = 14
    map_array[4,2] = 14

    # randomly assign markers 1, 2, 4, ... to available cells
    markers = [2 ** i for i in range(n_markers)]

    for marker in markers:
        x, y = np.random.randint(shape[0]), np.random.randint(shape[1])
        while not check_valid_cell(x, y) or map_array[x, y] != 0:
            x, y = np.random.randint(shape[0]), np.random.randint(shape[1])
        map_array[x, y] = marker

    # # randomly assign targets 11, 12, 13, 14, ... to available cells
    # targets = [11 + i for i in range(n_markers)]

    # for target in targets:
    #     x, y = np.random.randint(shape[0]), np.random.randint(shape[1])
    #     while not check_valid_cell(x, y) or map_array[x, y] != 0:
    #         x, y = np.random.randint(shape[0]), np.random.randint(shape[1])
    #     map_array[x, y] = target

    # # randomly assign blockages 14 to available cells, number of blockages is between 2 to 6
    # n_blockages = np.random.randint(2, 7)

    # for _ in range(n_blockages):
    #     x, y = np.random.randint(shape[0]), np.random.randint(shape[1])
    #     while not check_valid_cell(x, y) or map_array[x, y] != 0:
    #         x, y = np.random.randint(shape[0]), np.random.randint(shape[1])
    #     map_array[x, y] = 14

    return map_array

def save_map(n_markers, map_array, filename):
    with open(filename, "w") as f:
        f.write(f"{n_markers}\n")
        for row in map_array:
            f.write(" ".join(map(str, row)) + "\n")

def main():
    args = parse_arg()
    record = []

    if args.n_maps == 1:
        map = generate_map(args.n_markers, (args.shape_x, args.shape_y))
        save_map(args.n_markers, map, args.output_file)

        print(f"Map generated and saved to {args.output_file}")

    else:
        for i in range(args.n_maps):
            while True:
                map = generate_map(args.n_markers, (args.shape_x, args.shape_y))
                for m in record:
                    if np.all(m==map):
                        continue
                break

            record.append(map)
            save_map(args.n_markers, map, f"{args.output_prefix}_{i}.txt")

            print(f"Map {i} generated and saved to {args.output_prefix}{i}.txt")

if __name__ == "__main__":
    main()