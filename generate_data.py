from cube_class import Cube
import csv
from random import randint, shuffle

move_candidate = ["R", "R2", "R'", "L", "L2", "L'", "U", "U2", "U'", "D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]

t = [0 for _ in range(21)]

max_depth = 20
'''
all_num = 4
new_branch_min = 1
new_branch_max = 2
now_depth = 0
def generate(cube, depth, l_twist):
    global res, t
    if depth == max_depth:
        return
    depth += 1
    if depth > all_num:
        num = randint(new_branch_min, new_branch_max)
        tmp = []
        twist = randint(0, 17)
        while twist // 3 == l_twist // 3 or twist in set(tmp):
            twist = randint(0, 17)
        tmp.append(twist)
    else:
        tmp = list(range(18))
        shuffle(tmp)
    if t[depth] < 100:
        for twist in tmp:
            n_cube = cube.move(twist)
            arr = n_cube.idx()
            arr.append(depth)
            if True or depth > all_num:
                res.append(arr)
                t[depth] += 1
            else:
                few_move.append(arr)
            generate(n_cube, depth, twist)
'''

def generate(depth, l_twist, cube):
    if depth == 0:
        return cube
    twist = randint(0, 17)
    while twist // 3 == l_twist // 3:
        twist = randint(0, 17)
    cube = cube.move(twist)
    return cube

res = []
few_move = []
cube = Cube()
tmp = cube.idx()
tmp.append(0)
few_move.append(tmp)
print(cube.idx())
#label = list(range(324))
#label.append('num')
#res.append(label)
#generate(cube, 0, -10)
for depth in range(21):
    for _ in range(100):
        tmp = generate(depth, -10, Cube()).idx()
        tmp.append(depth)
        res.append(tmp)

with open('data_test.csv', mode='w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for arr in res:
        writer.writerow(arr)
'''
few_move.sort()
with open('few_move.csv', mode='w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for arr in few_move:
        writer.writerow(arr)
'''