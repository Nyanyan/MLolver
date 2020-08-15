from cube_class import Cube
import csv
from random import randint

move_candidate = ["R", "R2", "R'", "L", "L2", "L'", "U", "U2", "U'", "D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]

max_depth = 20
new_branch_min = 2
new_branch_max = 2
now_depth = 0
def generate(cube, depth):
    global res, now_depth
    if depth == max_depth + 1:
        return
    for _ in range(randint(new_branch_min, new_branch_max)):
        n_cube = cube.move(randint(0, 17))
        arr = n_cube.idx()
        arr.append(depth + 1)
        res.append(arr)
        generate(n_cube, depth + 1)

res = []
cube = Cube()
arr = cube.idx()
arr.append(0)
label = list(range(40))
label.append('num')
res.append(label)
res.append(arr)
generate(cube, 0)

with open('data.csv', mode='a') as f:
    writer = csv.writer(f, lineterminator='\n')
    for arr in res:
        writer.writerow(arr)