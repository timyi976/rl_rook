import numpy as np

def T1Norm(pt1, pt2):
    diff_x = abs(pt1[0] - pt2[0])
    diff_y = abs(pt1[1] - pt2[1])
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

def pt2lineDist_points(pt: tuple, target_pts: list):
    if len(target_pts) == 0:
        return 0
    
    min_dist = float('inf')

    for target_pt in target_pts:
        min_dist = min(min_dist, T1Norm(pt, target_pt))

    return min_dist

def pt2lineDist_traj(pt: tuple, trajectory: np.ndarray):
    # trajectory should be a 2D numpy array
    assert len(trajectory.shape) == 2, "Trajectory should be a 2D numpy array"

    min_dist = float('inf')

    # get coordinates of cells that are not 0
    target_pts = trajectory.nonzero()

    # if there are no cells that are not 0, return 0
    if len(target_pts[0]) == 0:
        return 0

    for i in range(len(target_pts[0])):
        target_pt = (target_pts[0][i], target_pts[1][i])
        min_dist = min(min_dist, T1Norm(pt, target_pt))

    return min_dist

def combine_trajectory(trajectory: np.ndarray, skip: int | None = None):
    if skip is not None:
        if skip == 0:
            slices = trajectory[1:]
        elif skip == trajectory.shape[0] - 1:
            slices = trajectory[:-1]
        else:
            slices = np.concatenate((trajectory[:skip], trajectory[skip+1:]))
    else:
        slices = trajectory

    result = np.logical_or.reduce(slices, axis=0)

    return result