import gymnasium as gym
import pykachu_env
import pickle
import random
import numpy as np
import time

# 환경 생성 (왼쪽: AI, 오른쪽: Q-learning agent)
env = gym.make("PykachuVolleyball-v0", render_mode="human", is_player_2_computer=False)

# 상태 추출 함수 (오른쪽 피카츄 기준)
def get_state(info):
    ball_x = int(info["ball"]["x"] // 20)
    ball_y = int(info["ball"]["y"] // 20)
    p2_x = int(info["player2"]["x"] // 20)
    p2_y = int(info["player2"]["y"] // 20)
    return (ball_x, ball_y, p2_x, p2_y)

# 행동 공간 정의
action_space = [tuple(a) for a in np.ndindex((3, 3, 2))]

# Q-table 불러오기
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)
print("✅ Q-table 불러오기 완료")

# 테스트 5회 진행
for episode in range(5):
    obs = env.reset()
    state = get_state(env.unwrapped.info)
    done = False
    total_reward = 0

    while not done:
        # 오른쪽 에이전트 행동 선택
        if state in Q:
            action = max(Q[state], key=Q[state].get)
        else:
            action = random.choice(action_space)

        # step()은 player2(오른쪽)의 action만 받는다.
        # player1(왼쪽)은 env 내부에서 빈 입력 + 내장 AI가 처리한다.
        obs, reward, terminated, info = env.step(action)
        done = terminated
        total_reward += reward
        state = get_state(info)

        env.render()
        time.sleep(0.02)

    print(f"🎮 테스트 Episode {episode+1} 완료 | 총 리워드: {total_reward:.2f}")

env.close()
