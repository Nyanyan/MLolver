from cube_class import Cube
import csv
from random import randint

move_candidate = ["R", "R2", "R'", "L", "L2", "L'", "U", "U2", "U'", "D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]

max_depth = 21
new_branch_min = 1
new_branch_max = 2
now_depth = 0
def generate(cube, depth):
    global res, now_depth
    if depth == max_depth:
        return
    for _ in range(randint(new_branch_min, new_branch_max + 1)):
        n_cube = cube.move(randint(0, 17))
        cp, co, ep, eo = n_cube.idx()
        res.append([cp, cp, ep, eo, depth + 1])
        generate(n_cube, depth + 1)

res = []
cube = Cube()
cp, co, ep, eo = cube.idx()
res.append(['cp', 'co', 'ep', 'eo', 'num'])
res.append([cp, co, ep, eo, 0])
generate(cube, 0)

with open('data.csv', mode='w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for arr in res:
        writer.writerow(arr)