import gym
import numpy as np
env = gym.make("Taxi-v3").env
env.render()
print(env.action_space)
print(env.observation_space)
observation = env.reset()
action = 0
observation, reward, done, info = env.step(action)


# solution
env = gym.make("Taxi-v3").env
Q = np.zeros((500, 6))
alpha = .1
gamma = .9
epsilon = .1
diff = .00001
while diff >= .00001:
    Qprev = Q.copy()
    observation = env.reset()
    done = False
    while not done:
        prev_obs = observation
        action = np.argmax(Q[prev_obs])
        observation, reward, done, info = env.step(action)

        # do the update
        if done:
            Q[prev_obs, action] = Q[prev_obs, action] + alpha * (
                        reward - Q[prev_obs, action])
            break
        else:
            Q[prev_obs, action] = Q[prev_obs, action] + alpha * (
                        reward + gamma * np.max(Q[observation]) - Q[prev_obs, action])

    diff = np.linalg.norm(Qprev - Q)




# solution 2
env = gym.make("Taxi-v3").env
Q = np.zeros((500, 6))
alpha = .15
gamma = .9
epsilon = .25
diff = .00001
for _ in range(300000):
    observation = env.reset()
    done = False
    while not done:
        prev_obs = observation
        p = np.random.random()
        if p >= epsilon:
            action = np.argmax(Q[prev_obs])
        else:
            action = np.random.randint(6)

        observation, reward, done, info = env.step(action)
        # do the update
        if done:
            Q[prev_obs, action] = Q[prev_obs, action] + alpha * (
                        reward - Q[prev_obs, action])
            break
        else:
            Q[prev_obs, action] = Q[prev_obs, action] + alpha * (
                        reward + gamma * np.max(Q[observation]) - Q[prev_obs, action])

