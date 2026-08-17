import gymnasium as gym
import pykachu_env
import numpy as np
import random
import time
import pickle
import matplotlib.pyplot as plt

env = gym.make("PykachuVolleyball-v0", render_mode="human", is_player_2_computer=False)

def get_state(info):
    ball_x = int(info["ball"]["x"] // 20)
    ball_y = int(info["ball"]["y"] // 20)
    p2_x = int(info["player2"]["x"] // 20)
    p2_y = int(info["player2"]["y"] // 20)
    return (ball_x, ball_y, p2_x, p2_y)

action_space = [tuple(a) for a in np.ndindex((3, 3, 2))]  # 총 18가지 조합
Q = {}

# 하이퍼파라미터
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.1
episodes = 10000

episode_rewards = []
epsilons = []

for episode in range(episodes):
    obs = env.reset()
    state = get_state(env.unwrapped.info)
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy
        if random.random() < epsilon or state not in Q:
            action = random.choice(action_space)
        else:
            action = max(Q[state], key=Q[state].get)

        obs, reward, terminated, info = env.step(action)
        done = terminated
        next_state = get_state(info)
        total_reward += reward

        # Q-table 초기화
        if state not in Q:
            Q[state] = {a: 0 for a in action_space}
        if next_state not in Q:
            Q[next_state] = {a: 0 for a in action_space}

        # Q-learning 업데이트
        old_q = Q[state][action]
        next_max = max(Q[next_state].values())
        Q[state][action] = old_q + alpha * (reward + gamma * next_max - old_q)

        state = next_state
        env.render()
        time.sleep(0.02)

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)
    epsilons.append(epsilon)
    print(f"🎮 Episode {episode+1} 완료 | epsilon={epsilon:.3f} | reward={total_reward:.2f}")

env.close()

# Q-table 저장
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)
print("✅ Q-table 저장 완료: q_table.pkl")

# 리워드 비교
print(f"📊 초기 평균 리워드: {np.mean(episode_rewards[:10]):.2f}")
print(f"📈 마지막 평균 리워드: {np.mean(episode_rewards[-10:]):.2f}")

# 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label="Episode Reward")
plt.title("Episode Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epsilons, label="Epsilon", color="orange")
plt.title("Epsilon Decay Over Time")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
