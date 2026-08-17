# Pykachu-Volleyball

**A tabular Q-learning agent for the Pikachu Volleyball `gymnasium` environment, with a dense reward reshaped for table-based learning**

> Fork of [Frog-Slayer/Pykachu-Volleyball](https://github.com/Frog-Slayer/Pykachu-Volleyball) · Environment: `gymnasium`, single-agent vs the game's built-in AI · Everything new here is under [Q-learning agent](#q-learning-agent)

---

![pikachu](./pika.gif)

## Overview

The upstream repository provides the `gymnasium` environment; this fork adds the
learning agent and the reward function it needed.

The environment is an adaption of [this code](https://github.com/gorisanson/pikachu-volleyball), obtained by reverse engineering the original game developed by "(C) SACHI SOFT". It is a `gymnasium` environment for single-agent reinforcement learning with a computer as an opponent. Multi-agent support will be added later, using `pettingzoo`.

## Environment API
 
```python
#this is in sample.py
import gymnasium as gym
import pykachu_env

env = gym.make('PykachuVolleyball-v0', 
               render_mode= "human", 
               is_player_2_computer=False)
```

Create your environment through `gym.make('PykachuVolleyball-v0')`. Options include `render_mode`, and `is_player_2_computer`, which determines which player the computer will control.

|                   |                          |
|-------------------|--------------------------|
| Action Space      | MultiDiscrete([3, 3, 2]) |
| Observation Space | (432, 304, 3)            |
| Observation High  | 255                      |
| Observation Low   | 0                        |


### Action Space
The action space is `MultiDiscrete([3, 3, 2])`, and each corresponds to left-right input, up-down input, and power hit input in order.

#### `action_space[0]` (left-right movement)
| Value             | Meaning                  |
|-------------------|--------------------------|
| 0                 | input left movement      |
| 1                 | NOOP                     |
| 2                 | input right movement     |

#### `action_space[1]` (up-down movement)
| Value             | Meaning                  |
|-------------------|--------------------------|
| 0                 | input up movement        |
| 1                 | NOOP                     |
| 2                 | input down movement      |

#### `action_space[2]` (power hit)
| Value             | Meaning                  |
|-------------------|--------------------------|
| 0                 | NOOP                     |
| 1                 | input power hit          |


### Observation Space
The observation space is `Box(low=0, high=255, shape=(432, 304, 3), dtype=np.uint8)`. It is the RGB image, displayed to a human player. 

```python
#this is in sample.py
for episode in range(5):
    env.reset()

    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, terminated, info = env.step(action)
        if terminated:
            break

env.close()
```

The `step()` funcion also returns `reward`, `terminated`, and `info`, along with the above `observation`. 

`step()` takes **one** action, which controls **player 2 (the right-hand Pikachu)**.
Player 1 is driven by the built-in computer AI, so you never pass an input for it.

#### `reward`
See [Reward shaping](#reward-shaping) — this fork replaces the upstream `+1 / -1`
win-loss reward with a dense, shaped reward.

#### `terminated`
`True` if the ball touches the ground, otherwise `False`. 

#### `info`
You can get additional information about the players and tha ball.
```json
{
    "player1": {
        "x": player1.x,
        "y": player1.y,
        "dive_direction" : player1.dive_direction 
    },
    "player2":{
        "x": player2.x,
        "y": player2.y,
        "dive_direction" : player2.dive_direction 
    },
    "ball": {
        "x": ball.x,
        "x_velocity": ball.x_velocity,
        "y": ball.y,
        "y_velocity": ball.y_velocity,
    }
}
```


## Q-learning agent

This is the part added in this fork. A tabular Q-learning agent learns to play
**player 2 (right side)** against the game's built-in computer AI on the left.

| File                 | Purpose                                                        |
|----------------------|----------------------------------------------------------------|
| `train_qlearning.py` | Trains the agent and writes the Q-table to `q_table.pkl`        |
| `test_qlearning.py`  | Loads `q_table.pkl` and plays 5 greedy episodes                 |
| `q_table.pkl`        | A pre-trained Q-table, so you can run the test script directly  |
| `sample.py`          | Upstream random-action example                                  |

### State discretization

The raw observation is a `(432, 304, 3)` RGB frame, which is far too large for a
table. Instead the agent builds its state from `info`, binning positions into a
20-pixel grid:

```python
state = (ball.x // 20, ball.y // 20, player2.x // 20, player2.y // 20)
```

The Q-table is a `dict` keyed by that 4-tuple, mapping to a `dict` over all
18 actions (`3 x 3 x 2`). Unseen states are created lazily, so the table only
holds states the agent has actually visited.

### Reward shaping

The upstream environment returns `+1` on a win and `-1` on a loss. That signal is
too sparse for tabular Q-learning — most steps carry no information at all — so
`PykachuEnv.step()` in this fork returns a dense reward instead:

| Term          | Value                   | Intent                                            |
|---------------|-------------------------|---------------------------------------------------|
| Win           | `+15` (on termination)  | Main objective                                    |
| Loss          | `-10` (on termination)  | Main objective                                    |
| Rally bonus   | `+0.1` per step         | Reward keeping the ball alive                     |
| Move bonus    | `+0.01 * distance`      | Discourage standing still                         |
| Hit bonus     | `+1.0` on ball contact  | Reward actually touching the ball                 |
| Proximity     | `+0.1 / distance`       | Dense gradient toward the ball when not touching  |

Note that `step()` returns the 4-tuple `(observation, reward, terminated, info)`,
not the gymnasium-standard 5-tuple.

### Hyperparameters

Defaults in `train_qlearning.py`:

| Parameter       | Value   |
|-----------------|---------|
| `alpha`         | `0.1`   |
| `gamma`         | `0.95`  |
| `epsilon`       | `1.0` decaying by `0.999` per episode, floor `0.1` |
| `episodes`      | `10000` |

### Usage

```bash
pip install -e .
pip install matplotlib     # only needed for the training plots

python train_qlearning.py  # trains, saves q_table.pkl, plots reward + epsilon
python test_qlearning.py   # replays 5 greedy episodes from q_table.pkl
```

Both scripts run with `render_mode="human"` and a `time.sleep(0.02)` per step, so
they play back at watchable speed. Training all 10,000 episodes this way takes a
very long time — drop the `sleep` (and the renderer) if you only want the table.

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
