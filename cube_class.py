'''
Corner
   B
  0 1 
L 2 3 R
   F

   F
  4 5
L 6 7 R
   B


Edge
top layer
    B
    0
L 3   1 R
    2
    F

middle layer
4 F 5 R 6 B 7

bottom layer
    F
    8
L 11  9 R
    10
    B
'''


fac = [1 for _ in range(15)]
for i in range(1, 15):
    fac[i] = fac[i - 1] * i

def cmb(n, r):
    return fac[n] // fac[r] // fac[n - r]

class Cube:
    def __init__(self, cp=list(range(8)), co=[0 for _ in range(8)], ep=list(range(12)), eo=[0 for i in range(12)]):
        self.Cp = cp
        self.Co = co
        self.Ep = ep
        self.Eo = eo
    
    def move_cp(self, mov):
        surface = [[3, 1, 7, 5], [0, 2, 4, 6], [0, 1, 3, 2], [4, 5, 7, 6], [2, 3, 5, 4], [1, 0, 6, 7]]
        res = [i for i in self.Cp]
        mov_type = mov // 3
        mov_amount = mov % 3
        for i in range(4):
            res[surface[mov_type][(i + mov_amount + 1) % 4]] = self.Cp[surface[mov_type][i]]
        return res
    
    def move_co(self, mov):
        surface = [[3, 1, 7, 5], [0, 2, 4, 6], [0, 1, 3, 2], [4, 5, 7, 6], [2, 3, 5, 4], [1, 0, 6, 7]]
        pls = [2, 1, 2, 1]
        res = [i for i in self.Co]
        mov_type = face(mov)
        mov_amount = mov % 3
        for i in range(4):
            res[surface[mov_type][(i + mov_amount + 1) % 4]] = self.Co[surface[mov_type][i]]
            if axis(mov) != 1 and mov_amount != 1:
                res[surface[mov_type][(i + mov_amount + 1) % 4]] += pls[(i + mov_amount + 1) % 4]
                res[surface[mov_type][(i + mov_amount + 1) % 4]] %= 3
        return res
    
    def move_ep(self, mov):
        surface = [[1, 6, 9, 5], [3, 4, 11, 7], [0, 1, 2, 3], [8, 9, 10, 11], [2, 5, 8, 4], [0, 7, 10, 6]]
        res = [i for i in self.Ep]
        mov_type = face(mov)
        mov_amount = mov % 3
        for i in range(4):
            res[surface[mov_type][(i + mov_amount + 1) % 4]] = self.Ep[surface[mov_type][i]]
        return res
    
    def move_eo(self, mov):
        surface = [[1, 6, 9, 5], [3, 4, 11, 7], [0, 1, 2, 3], [8, 9, 10, 11], [2, 5, 8, 4], [0, 7, 10, 6]]
        res = [i for i in self.Eo]
        mov_type = face(mov)
        mov_amount = mov % 3
        for i in range(4):
            res[surface[mov_type][(i + mov_amount + 1) % 4]] = self.Eo[surface[mov_type][i]]
        if axis(mov) == 2 and mov_amount != 1:
            for i in surface[mov_type]:
                res[i] += 1
                res[i] %= 2
        return res
    
    def move(self, mov):
        return Cube(cp=self.move_cp(mov), co=self.move_co(mov), ep=self.move_ep(mov), eo=self.move_eo(mov))
    
    def idx(self):
        '''
        idx_cp = 0 # max 40320
        for i in range(8):
            cnt = self.Cp[i]
            for j in self.Cp[:i]:
                if j < self.Cp[i]:
                    cnt -= 1
            idx_cp += fac[7 - i] * cnt
        idx_co = 0 # max 2187
        for i in range(7):
            idx_co *= 3
            idx_co += self.Co[i]
        idx_ep = 0 # max 479001600
        for i in range(12):
            cnt = self.Ep[i]
            for j in self.Ep[:i]:
                if j < self.Ep[i]:
                    cnt -= 1
            idx_ep += fac[12 - i] * cnt
        idx_eo = 0 # max 2048
        for i in range(11):
            idx_eo *= 2
            idx_eo += self.Eo[i]
        return [idx_cp, idx_co, idx_ep, idx_eo]
        
        res = []
        res.extend(self.Cp)
        res.extend(self.Co)
        res.extend(self.Ep)
        res.extend(self.Eo)
        return res
        '''
        '''
        res = [0 for _ in range(324)]
        for i in range(6):
            res[i * 9 + 4 + i * 54] = 1
        corner_colors = [[0, 4, 3], [0, 3, 2], [0, 1, 4], [0, 2, 1], [5, 4, 1], [5, 1, 2], [5, 3, 4], [5, 2, 3]]
        edge_colors = [[0, 3], [0, 2], [0, 1], [0, 4], [1, 4], [1, 2], [3, 2], [3, 4], [5, 1], [5, 2], [5, 3], [5, 4]]
        corner_stickers = [[0, 36, 29], [2, 27, 20], [6, 9, 38], [8, 18, 11], [45, 44, 15], [47, 17, 24], [51, 35, 42], [53, 26, 33]]
        edge_stickers = [[1, 28], [5, 19], [7, 10], [3, 37], [12, 41], [14, 21], [30, 23], [32, 39], [46, 16], [50, 25], [52, 34], [48, 43]]
        for corner_idx in range(8):
            corner = self.Cp[corner_idx]
            co = self.Co[corner_idx]
            for i, j in enumerate(corner_stickers[corner_idx]):
                color = corner_colors[corner][(i - co) % 3]
                res[j + color * 54] = 1
        for edge_idx in range(12):
            edge = self.Ep[edge_idx]
            eo = self.Eo[edge_idx]
            for i, j in enumerate(edge_stickers[edge_idx]):
                color = edge_colors[edge][(i - eo) % 2]
                res[j + color * 54] = 1
        '''
        '''
        res_2 = [0 for _ in range(36)]
        for i in range(36):
            cnt = 0
            for j in range(9):
                cnt += res[i * 9 + j]
            res_2[i] = cnt
        #return res_2

        res_reshape = [[[res[54 * color + 9 * face + i:54 * color + 9 * face + i + 3] for i in range(0, 7, 3)] for face in range(6)] for color in range(6)]
        res_3 = [0 for _ in range(6)] # 乱雑さ=面にあるステッカーの種類数 x 四辺で接している塊数
        for face in range(6):
            species = 0
            for color in range(6):
                flag = False
                for y in range(3):
                    for x in range(3):
                        if res_reshape[color][face][y][x] == 1:
                            species += 1
                            flag = True
                        if flag:
                            break
                    if flag:
                        break
            cnt = 0
            for color in range(6):
                marked = [[False for _ in range(3)] for _ in range(3)]
                for y in range(3):
                    for x in range(3):
                        if marked[y][x]:
                            continue
                        marked[y][x] = True
                        if not res_reshape[color][face][y][x]:
                            continue
                        cnt += 1
                        stack = [[y, x]]
                        dy = [-1, 1, 0, 0]
                        dx = [0, 0, -1, 1]
                        while stack:
                            yy, xx = stack.pop()
                            for i in range(4):
                                ny = yy + dy[i]
                                nx = xx + dx[i]
                                if 0 <= ny < 3 and 0 <= nx < 3 and res_reshape[ny][nx] and not marked[ny][nx]:
                                    marked[ny][nx] = True
                                    stack.append([ny, nx])
            res_3[face] = (species * cnt) ** 0.5
        
        res_all = [i for i in res_2]
        res_all.extend([i for i in res_3])
        return res_all
        '''
        '''
        res = []
        res.extend(self.Co)
        res.extend(self.Eo)
        for corner in range(8):
            tmp = [0 for _ in range(8)]
            tmp[self.Cp[corner]] = 1
            res.extend(tmp)
        for edge in range(12):
            tmp = [0 for _ in range(12)]
            tmp[self.Ep[edge]] = 1
            res.extend(tmp)
        '''
        res = []
        for ep in self.Ep:
            tmp = [0 for _ in range(12)]
            tmp[ep] = 1
            res.extend(tmp)
        for cp in self.Cp:
            tmp = [0 for _ in range(12)]
            tmp[cp] = 1
            res.extend(tmp)
        tmp = [0 for _ in range(12)]
        for i in range(6):
            tmp[i * 2 + self.Eo[i]] = 1
        res.extend(tmp)
        tmp = [0 for _ in range(12)]
        for i in range(6, 12):
            tmp[(i - 6) * 2 + self.Eo[i]] = 1
        res.extend(tmp)
        tmp = [0 for _ in range(12)]
        for i in range(4):
            tmp[i * 3 + self.Co[i]] = 1
        res.extend(tmp)
        tmp = [0 for _ in range(12)]
        for i in range(4, 8):
            tmp[(i - 4) * 3 + self.Co[i]] = 1
        res.extend(tmp)
        return res

def face(twist):
    return twist // 3

def axis(twist):
    return twist // 6
