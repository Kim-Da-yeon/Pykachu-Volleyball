# Pykachu-Volleyball

Pikachu Volleyball `gymnasium` 환경에 표 기반 Q-learning 적용. 표 학습용 밀집 보상 재설계 포함.
[Frog-Slayer/Pykachu-Volleyball](https://github.com/Frog-Slayer/Pykachu-Volleyball) 포크 — 환경은 upstream, 학습 부분이 이 포크에서 추가된 것.

---

![pikachu](./pika.gif)

## 환경

`gym.make('PykachuVolleyball-v0', render_mode="human", is_player_2_computer=False)`

| | |
|---|---|
| Action Space | `MultiDiscrete([3, 3, 2])` — 좌우 · 상하 · 파워히트 |
| Observation Space | `Box(0, 255, (432, 304, 3), uint8)` — 화면 RGB |
| `step()` 인자 | player 2 (오른쪽) 행동 하나. player 1은 내장 AI |
| `step()` 반환 | `(observation, reward, terminated, info)` — 4-tuple |
| `terminated` | 공이 바닥에 닿으면 `True` |
| `info` | player1 · player2 좌표와 dive_direction, ball 좌표와 속도 |

| 값 | `action_space[0]` 좌우 | `action_space[1]` 상하 | `action_space[2]` 파워히트 |
|---|---|---|---|
| 0 | 왼쪽 | 위 | NOOP |
| 1 | NOOP | NOOP | 파워히트 |
| 2 | 오른쪽 | 아래 | — |

## Q-learning

| 파일 | 내용 |
|---|---|
| `train_qlearning.py` | 학습 후 `q_table.pkl` 저장, reward·epsilon 곡선 |
| `test_qlearning.py` | `q_table.pkl` 로드, greedy 5 에피소드 |
| `q_table.pkl` | 학습된 Q-table |
| `sample.py` | upstream 랜덤 행동 예제 |

- **상태** — 관측이 432×304×3 RGB라 표에 부적합. `info`의 좌표를 20px 격자로 이산화
  ```python
  state = (ball.x // 20, ball.y // 20, player2.x // 20, player2.y // 20)
  ```
  Q-table은 이 4-tuple을 키로 18개 행동(3×3×2)에 매핑. 미방문 상태는 지연 생성
- **보상** — upstream은 승 `+1` / 패 `-1`. 표 학습에는 신호가 희소해 밀집 보상으로 교체

  | 항목 | 값 | 목적 |
  |---|---|---|
  | 승 / 패 | `+15` / `-10` | 주 목표 |
  | 랠리 | `+0.1`/step | 공 유지 |
  | 이동 | `+0.01 × 거리` | 정지 억제 |
  | 타격 | `+1.0` | 공 접촉 |
  | 근접 | `+0.1 / 거리` | 공 방향 기울기 |

- **하이퍼파라미터** — `alpha` 0.1 · `gamma` 0.95 · `epsilon` 1.0에서 0.999 감쇠, 하한 0.1 · 10,000 에피소드

## 실행

```bash
pip install -e .
pip install matplotlib

python train_qlearning.py
python test_qlearning.py
```

두 스크립트 모두 `render_mode="human"` + step당 `time.sleep(0.02)`. 10,000 에피소드는 매우 오래 걸리므로 표만 필요하면 sleep과 렌더러 제거.

## Limitations

- **The state is not Markov.** It holds only positions — ball `(x, y)` and player
  `(x, y)`, binned to 20px — and drops `ball.x_velocity` / `ball.y_velocity`, which
  `info` does expose. The agent therefore cannot distinguish a ball travelling
  toward it from the same ball travelling away, and those two situations need
  opposite actions. Adding binned velocity is the single most likely improvement.
- **No results are reported.** The training script plots reward and epsilon curves
  but nothing is saved or summarized, and win rate against the built-in AI was
  never measured. The shipped `q_table.pkl` has no accompanying score.
- **Rendering during training.** As noted above, 10,000 episodes at 0.02 s per step
  is impractical; the table shipped here was necessarily trained on far fewer.
- **The reward's shaping terms may fight the objective.** The `+0.1` per-step rally
  bonus rewards prolonging a rally, which is not the same as winning it, and the
  `+0.1 / dist` proximity term grows without bound as the player approaches the
  ball. Neither weight was tuned.
- **`step()` returns a 4-tuple**, not gymnasium's standard
  `(obs, reward, terminated, truncated, info)`, so standard RL libraries
  (Stable-Baselines3 and similar) will not accept this environment as-is.

## TODOs
For the code may not be fully compliant with `gymnasium` standard APIs, there may be some bugs or issues. Please let me know if there is an error or if you need additional functions. Good luck.

- [X] reset and reward 
- [X] refactor to Python convention 
- [X] draw background
- [X] ~~show game start message~~
- [X] transparent Pikachu background 
- [X] trail, punch, hyper sprites 
- [X] fix computer's improper movements
- [X] ball rotation 
- [X] draw shadows
- [ ] support render_mode 'rgb_array'
- [ ] specific comments 
- [ ] add sound(and make it optional) 
