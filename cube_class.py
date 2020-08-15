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
    
    def idx_cp(self):
        res = 0
        for i in range(8):
            cnt = self.Cp[i]
            for j in self.Cp[:i]:
                if j < self.Cp[i]:
                    cnt -= 1
            res += fac[7 - i] * cnt
        return res
    
    def idx_co(self):
        res = 0
        for i in range(7):
            res *= 3
            res += self.Co[i]
        return res

    def idx_ep(self):
        mse_parts = [[2, 0, 10, 8], [3, 1, 9, 11], [4, 5, 6, 7]]
        arr = [[-1 for _ in range(4)] for _ in range(3)]
        for i in range(12):
            for mse in range(3):
                if self.Ep[i] in mse_parts[mse]:
                    arr[mse][mse_parts[mse].index(self.Ep[i])] = i
        #print(arr)
        res = [-1 for _ in range(3)]
        for mse in range(3):
            tmp = 0
            for i in range(4):
                cnt = arr[mse][i]
                for j in arr[mse][:i]:
                    if j < arr[mse][i]:
                        cnt -= 1
                tmp += cnt * cmb(11 - i, 3 - i) * fac[3 - i]
            res[mse] = tmp
        return res
    
    def idx_eo(self):
        res_eo = 0
        for i in range(11):
            res_eo *= 2
            res_eo += self.Eo[i]
        return res_eo
    
    def solved_ep_phase0(self):
        for i in range(4, 8):
            if not self.Ep[i] in {4, 5, 6, 7}:
                return False
        return True

    '''
    def idx_phase0_ep(self):
        res_ep = 0
        cnt = 0
        for i in range(12):
            if cnt == 4:
                break
            if self.Ep[i] // 4 == 1:
                res_ep += cmb(11 - i, 4 - cnt)
                cnt += 1
        return res_ep
    
    def idx_phase0_eo(self):
        res_eo = 0
        for i in range(11):
            res_eo *= 2
            res_eo += self.Eo[i]
        return res_eo
    
    def idx_phase1_ep_ud(self):
        res_ep_ud = 0
        arr_ep_ud = [self.Ep[i] for i in [0, 1, 2, 3, 8, 9, 10, 11]]
        for i in range(8):
            if arr_ep_ud[i] > 3:
                arr_ep_ud[i] -= 4
        for i in range(8):
            cnt = arr_ep_ud[i]
            for j in arr_ep_ud[:i]:
                if j < arr_ep_ud[i]:
                    cnt -= 1
            res_ep_ud += fac[7 - i] * cnt
        return res_ep_ud

    def idx_phase1_ep_fbrl(self):
        res_ep_fbrl = 0
        arr_ep_fbrl = [self.Ep[i] - 4 for i in [4, 5, 6, 7]]
        for i in range(4):
            cnt = arr_ep_fbrl[i]
            for j in arr_ep_fbrl[:i]:
                if j < arr_ep_fbrl[i]:
                    cnt -= 1
            res_ep_fbrl += fac[3 - i] * cnt
        return res_ep_fbrl
    '''
    '''
    def idx_phase0(self):
        res_co = 0
        for i in range(7):
            res_co *= 3
            res_co += self.Co[i]
        res_ep = 0
        cnt = 0
        for i in range(12):
            if cnt == 4:
                break
            if self.Ep[i] // 4 == 1:
                res_ep += cmb(11 - i, 4 - cnt)
                cnt += 1
        res_eo = 0
        for i in range(11):
            res_eo *= 2
            res_eo += self.Eo[i]
        return res_co, res_ep + res_eo * 495
    
    def idx_phase1(self):
        res_cp = 0
        for i in range(8):
            cnt = self.Cp[i]
            for j in self.Cp[:i]:
                if j < self.Cp[i]:
                    cnt -= 1
            res_cp += fac[7 - i] * cnt
        res_ep_ud = 0
        arr_ep_ud = [self.Ep[i] for i in [0, 1, 2, 3, 8, 9, 10, 11]]
        for i in range(8):
            if arr_ep_ud[i] > 3:
                arr_ep_ud[i] -= 4
        for i in range(8):
            cnt = arr_ep_ud[i]
            for j in arr_ep_ud[:i]:
                if j < arr_ep_ud[i]:
                    cnt -= 1
            res_ep_ud += fac[7 - i] * cnt
        res_ep_fbrl = 0
        arr_ep_fbrl = [self.Ep[i] - 4 for i in [4, 5, 6, 7]]
        for i in range(4):
            cnt = arr_ep_fbrl[i]
            for j in arr_ep_fbrl[:i]:
                if j < arr_ep_fbrl[i]:
                    cnt -= 1
            res_ep_fbrl += fac[3 - i] * cnt
        return res_cp, res_ep_ud * 24 + res_ep_fbrl
    '''
def face(twist):
    return twist // 3

def axis(twist):
    return twist // 6
