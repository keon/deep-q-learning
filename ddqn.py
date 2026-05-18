# -*- coding: utf-8 -*-
import json
import os
import random
import gymnasium as gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.losses import Huber
from keras.optimizers import Adam

EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=Huber(delta=1.0),
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([m[0] for m in minibatch])
        next_states = np.vstack([m[3] for m in minibatch])
        # Double DQN: online net picks the action, target net evaluates it.
        targets = self.model.predict(states, verbose=0)
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        next_q = self.target_model.predict(next_states, verbose=0)
        for i, (_, action, reward, _, done) in enumerate(minibatch):
            targets[i][action] = reward if done else reward + self.gamma * next_q[i][next_actions[i]]
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()
        meta = name + ".meta.json"
        if os.path.exists(meta):
            with open(meta) as f:
                self.epsilon = json.load(f)["epsilon"]

    def save(self, name):
        self.model.save_weights(name)
        with open(name + ".meta.json", "w") as f:
            json.dump({"epsilon": self.epsilon}, f)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.weights.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.weights.h5")
