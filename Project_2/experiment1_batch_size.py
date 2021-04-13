import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import deque
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as k
tf.compat.v1.disable_eager_execution() # disable tensorflow's eager execution

def get_eps_greedy_action(state, epsilon):
    p = np.random.random()
    if p >= epsilon:
        action = np.argmax(model.predict_on_batch(state)[0])
    else:
        action = np.random.randint(4)
    return action

# solution
batch_sizes = [1, 8, 16]
final_scores = defaultdict(list)
for batch_size in batch_sizes:
    env = gym.make('LunarLander-v2')
    nn_nodes = 64
    epsilon = 1
    epsilon_decay = .995
    epsilon_min = .01
    alpha = 5e-4
    gamma = .99
    episodes = 500
    steps = 500
    scores = []

    # replay_buffer = []
    replay_buffer = deque(maxlen=int(1e5))
    C = 4

    # initialize model and target model
    model = Sequential()
    model.add(Dense(nn_nodes, input_dim=8, activation="relu"))
    model.add(Dense(nn_nodes, activation="relu"))
    model.add(Dense(4))
    model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=alpha),
                  metrics=["accuracy"])

    target_model = Sequential()
    target_model.add(Dense(nn_nodes, input_dim=8, activation="relu"))
    target_model.add(Dense(nn_nodes, activation="relu"))
    target_model.add(Dense(4))
    target_model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=alpha),
                  metrics=["accuracy"])

    w = model.get_weights()
    t = 0
    for episode in range(episodes):
        score=0
        state = env.reset()
        print('eps {} bs {}'.format(episode, batch_size))
        for step in range(steps):
            action = get_eps_greedy_action(state.reshape(1,8), epsilon)
            prev_state = state
            state, reward, done, info = env.step(action)
            prev_action = action
            score += reward
            replay_buffer.append([prev_state, prev_action, reward, state, done])
            if len(replay_buffer) >= batch_size:

                # sample random minibatch of (prev_state, prev_action, reward, state) from replay buffer
                minibatch = random.sample(replay_buffer, batch_size)
                states = []
                targets = []
                for j in range(len(minibatch)):
                    this_prev_state = minibatch[j][0]
                    this_prev_action = minibatch[j][1]
                    this_reward = minibatch[j][2]
                    this_state = minibatch[j][3]
                    this_done = minibatch[j][4]
                    if this_done:
                        yj_target = this_reward
                    else:
                        yj_target = this_reward + gamma*np.amax(target_model.predict_on_batch(this_state.reshape(1, 8)))

                    # model.set_weights(w)
                    states.append(this_prev_state)
                    target = model.predict_on_batch(this_prev_state.reshape(1,8))
                    target[0][this_prev_action] = yj_target
                    targets.append(target)
                # model.fit(np.array(states), np.array(targets).reshape(batch_size, 4), epochs=1, verbose=0)
                model.train_on_batch(np.array(states), np.array(targets).reshape(batch_size, 4))
                # print('step {} of episode {}'.format(step, episode))
                w = model.get_weights()
                t += 1
                if t % C == 0:
                    target_model.set_weights(w)

            if done:
                break

        scores.append(score)
        epsilon = epsilon*epsilon_decay
        epsilon = max(epsilon_min, epsilon)

    final_scores[batch_size] = scores



plt.figure()
plt.plot(range(len(scores)), final_scores[batch_sizes[0]])
plt.plot(range(len(scores)), final_scores[batch_sizes[1]])
plt.plot(range(len(scores)), final_scores[batch_sizes[2]])
plt.legend(["batch size 1", "batch size 8", "batch size 16"])
plt.xlabel("episode")
plt.ylabel("score")