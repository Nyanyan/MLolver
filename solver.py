from cube_class import Cube
import pickle
import numpy as np

move_candidate = ["R", "R2", "R'", "L", "L2", "L'", "U", "U2", "U'", "D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]

knn = pickle.load(open(filename, 'rb'))

scramble = [move_candidate.index(i) for i in input().split()]
print(len(scramble))
print(scramble)

cube = Cube()
for i in scramble:
    cube = cube.move(i)

data = np.array(cube.idx())
prediction = knn.predict(newdata)
print(prediction)
print("Predicted target name: {}".format(prediction))
