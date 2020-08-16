from cube_class import Cube
import csv
from random import randint

move_candidate = ["R", "R2", "R'", "L", "L2", "L'", "U", "U2", "U'", "D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]

num = 100000
t = 0

max_depth = 20
new_branch_min = 2
new_branch_max = 2
now_depth = 0
def generate(cube, depth, l_twist):
    global res, t
    if depth == max_depth:
        return
    if t >= num:
        return
    if depth > 3:
        tmp = range(randint(new_branch_min, new_branch_max))
    else:
        tmp = range(18)
    for _ in tmp:
        twist = l_twist
        while twist // 3 == l_twist // 3:
            twist = randint(0, 17)
        n_cube = cube.move(twist)
        arr = n_cube.idx()
        arr.append(depth + 1)
        res.append(arr)
        t += 1
        generate(n_cube, depth + 1, twist)

res = []
cube = Cube()
label = list(range(324))
label.append('num')
res.append(label)
generate(cube, 0, -10)

with open('data.csv', mode='w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for arr in res:
        writer.writerow(arr)