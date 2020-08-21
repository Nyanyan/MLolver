from cube_class import Cube, face, axis
import pickle
import numpy as np
from time import time
import keras

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

def distance(puzzle):
    arr = puzzle.idx()
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
    print('depth', end=' ',flush=True)
    strt = time()
    dis = distance(puzzle)
    for depth in range(30):
        print(depth, end=' ', flush=True)
        path = []
        if search(puzzle, depth, dis):
            for twist in path:
                puzzle = puzzle.move(twist)
            print('')
            for i in path:
                print(move_candidate[i], end=' ')
            print('')
            print(time() - strt, 'sec')
            break

few_move = []
with open('few_move.csv', mode='r') as f:
    for line in map(str.strip, f):
        few_move.append([int(i) for i in line.replace('\n', '').split(',')])
few_move.sort()

#                  0     1     2    3     4    5     6     7    8     9    10    11    12   13    14    15   16    17
move_candidate = ["R", "R2", "R'", "L", "L2", "L'", "U", "U2", "U'", "D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]
'''
filename = 'model.sav'
knn = pickle.load(open(filename, 'rb'))
'''
model = keras.models.load_model('model.h5', compile=False)

scramble = [move_candidate.index(i) for i in input().split()]
print('distance', len(scramble))
#print(scramble)

puzzle = Cube()
for i in scramble:
    puzzle = puzzle.move(i)
print('predicted', distance(puzzle))

path = []
solver(puzzle)