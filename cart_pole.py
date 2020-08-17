import gym
import numpy as np
import matplotlib.pyplot as plt

def do_episode(w, env):
    done = False
    observation = env.reset()
    num_steps = 0

    while not done and num_steps <= max_number_of_steps:
        action = take_action(observation, w)
        observation, _, done, _ = env.step(action)
        num_steps += 1
    # ここで報酬を与える。基本的に(連続したステップ数)-(最大ステップ数)で与えられる。
    step_val = -1 if num_steps >= max_number_of_steps else num_steps - max_number_of_steps
    return step_val, num_steps

def take_action(X, w): # 値が0を超えたら1を返すようにする
    action = 1 if calculate(X, w) > 0.0 else 0
    return action

def calculate(X, w):
    result = np.dot(X, w) # 返り値は配列ではなく、１つの値になる。
    return result

env = gym.make('CartPole-v0')

# env.render()
# ゲームの様子を見たいときは env.render()を実行すれば良い

eta = 0.2
sigma = 0.05 # パラメーターを変動させる値の標準偏差

max_episodes = 5000 # 学習を行う最大エピソード数
max_number_of_steps = 200
n_states = 4 # 入力のパラメーター数
num_batch = 10
num_consecutive_iterations = 100 # 評価の範囲のエピソード数


w = np.random.randn(n_states)
reward_list = np.zeros(num_batch)
last_time_steps = np.zeros(num_consecutive_iterations)
mean_list = [] # 学習の進行具合を過去100エピソードのステップ数の平均で記録する

for episode in range(max_episodes//num_batch):
    N = np.random.normal(scale=sigma,size=(num_batch, w.shape[0]))
    # パラメーターの値を変動させるための値。これが偏差になる。

    for i in range(num_batch):
        w_try = w + N[i]
        reward, steps = do_episode(w_try, env)
        if i == num_batch-1:
            print('%d Episode finished after %d steps / mean %f' %(episode*num_batch, steps, last_time_steps.mean()))
        last_time_steps = np.hstack((last_time_steps[1:], [steps]))
        reward_list[i] = reward
        mean_list.append(last_time_steps.mean())
    if last_time_steps.mean() >= 195: break # 平均が195超えたら学習終了

    std = np.std(reward_list)
    if std == 0: std = 1
    # 報酬の値を正規化する
    A = (reward_list - np.mean(reward_list))/std
    # ここでパラメーターの更新を行う
    w_delta = eta /(num_batch*sigma) * np.dot(N.T, A)
    # 振れ幅を調整するためにsigmaをかけている。
    w += w_delta


env.close()
# グラフの表示

plt.plot(mean_list)
plt.show()