from cube_class import Cube, face, axis
import pickle
import numpy as np
from time import time
import keras
from heapq import *
'''
def search_distance(arr):
    pre_l = 0
    pre_r = len(few_move) - 1
    for i in range(324):
        l = pre_l
        r = pre_r
        if few_move[l][i] == 0 and few_move[r][i] == 1:
            while r - l > 1:
                c = (r + l) // 2
                if few_move[c][i] == 0:
                    l = c
                else:
                    r = c
            if arr[i] == 0:
                pre_r = l
            else:
                pre_l = r
        elif few_move[l][i] == 1:
            if arr[i] == 0:
                return -1
        elif few_move[r][i] == 0:
            if arr[i] == 1:
                return -1
        if pre_r == pre_l:
            break
    for i in range(pre_l, pre_r + 1):
        if few_move[i][:324] == arr:
            return few_move[i][324]
    return -1
'''
def distance(puzzle):
    solved = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    arr = puzzle.idx()
    if arr == solved:
        return 0
    else:
        input_shape = (36, 3, 3)
        data = np.array([arr]).reshape(-1, input_shape[0], input_shape[1], input_shape[2])
        arr = model.predict(data)[0]
        mx = 0
        res = -1
        for i, j in enumerate(arr):
            if j > mx:
                res = i
                mx = j
        return res
    '''
    tmp = search_distance(arr)
    if tmp == -1:
        input_shape = (36, 3, 3)
        data = np.array([arr]).reshape(-1, input_shape[0], input_shape[1], input_shape[2])
        arr = model.predict(data)[0]
        mx = 0
        res = -1
        for i, j in enumerate(arr):
            if j > mx:
                res = i
                mx = j
        return res
    else:
        return tmp
    '''

def search(puzzle, depth, dis):
    global path
    if depth == 0:
        if dis == 0:
            return True
    else:
        if dis == 0:
            return True
        if dis <= depth:
            l_twist = path[-1] if len(path) >= 1 else -10
            ll_twist = path[-2] if len(path) >= 2 else -10
            for twist in range(18):
                if face(twist) == face(l_twist) or axis(twist) == axis(l_twist) == axis(ll_twist):
                    continue
                n_puzzle = puzzle.move(twist)
                n_dis = distance(n_puzzle)
                if n_dis > dis:
                    continue
                path.append(twist)
                if search(n_puzzle, depth - 1, n_dis):
                    return True
                path.pop()
        return False

def solver(puzzle):
    global path
    res = []
    que = []
    dis = distance(puzzle)
    heapify(que)
    heappush(que, [dis, dis, [], puzzle])
    weight = 0.5
    cnt = 0
    while que:
        cnt += 1
        if cnt % 10 == 0:
            print(cnt)
        _, dis, path, puz = heappop(que)
        if dis == 0:
            return path
        l0_twist = path[-1] if len(path) >= 1 else -10
        l1_twist = path[-2] if len(path) >= 2 else -10
        l = (len(path) + 1) * weight
        for twist in range(18):
            if face(twist) == face(l0_twist) or axis(twist) == axis(l0_twist) == axis(l1_twist):
                continue
            n_puz = puz.move(twist)
            n_dis = distance(n_puz)
            '''
            if n_dis > dis:
                continue
            '''
            n_path = [i for i in path]
            n_path.append(twist)
            heappush(que, [n_dis + l, n_dis, n_path, n_puz])
    return -1
'''
few_move = []
with open('few_move.csv', mode='r') as f:
    for line in map(str.strip, f):
        few_move.append([int(i) for i in line.replace('\n', '').split(',')])
'''

#                  0     1     2    3     4    5     6     7    8     9    10    11    12   13    14    15   16    17
move_candidate = ["R", "R2", "R'", "L", "L2", "L'", "U", "U2", "U'", "D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]

model = keras.models.load_model('models/model-2445-2607.h5', compile=False)

scramble = [move_candidate.index(i) for i in input().split()]
print('distance', len(scramble))
#print(scramble)

puzzle = Cube()
for i in scramble:
    puzzle = puzzle.move(i)
print('predicted', distance(puzzle))

strt = time()
solution = solver(puzzle)
print(time() - strt, 'sec')
if solution == -1:
    print('error')
print(' '.join([move_candidate[i] for i in solution]))