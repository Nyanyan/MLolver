from cube_class import Cube
import csv
from random import randint, shuffle

move_candidate = ["R", "R2", "R'", "L", "L2", "L'", "U", "U2", "U'", "D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]

t = [0 for _ in range(21)]

max_depth = 20
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
        tmp = range(randint(new_branch_min, new_branch_max))
    else:
        tmp = list(range(18))
        shuffle(tmp)
    if t[depth] < 100:
        l_twists = set([])
        for twist in tmp:
            if depth > all_num:
                twist = randint(0, 17)
                while twist // 3 == l_twist // 3 or twist in l_twists:
                    twist = randint(0, 17)
                l_twists.add(twist)
            n_cube = cube.move(twist)
            arr = n_cube.idx()
            arr.append(depth)
            if depth <= all_num:
                few_move.append(arr)
            else:
                res.append(arr)
                t[depth] += 1
            generate(n_cube, depth, twist)

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
generate(cube, 0, -10)

with open('data.csv', mode='w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for arr in res:
        writer.writerow(arr)

few_move.sort()
with open('few_move.csv', mode='w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for arr in few_move:
        writer.writerow(arr)
