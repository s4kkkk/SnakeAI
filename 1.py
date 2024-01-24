#!/bin/python

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from collections import deque
import random
import torch.cuda
import matplotlib.pyplot as plt
import pdb

from snake_env import Snake

t.autograd.set_detect_anomaly(True)

class SnakeAI(nn.Module):

    # структура сети
    def __init__(self, LearningRate=0.01, Epsilon=0.3):

        super().__init__()

        # Слои
        self.inp = nn.Linear(12, 1024)
        self.hidden1 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, 4)
        
        # Инициализация слоев
        init.uniform_(self.inp.weight, -1, 1)
        init.uniform_(self.hidden1.weight, -1, 1)
        init.uniform_(self.out.weight, -1, 1)

        
        # Память
        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_next_states = []
        self.memory_isdones = []
        self.memory_len = 0
        self.loses = []
        self.rewards_per_epoch = []

        self.optimizer = optim.Adam(self.parameters(), lr=LearningRate)
        self.criterion = nn.MSELoss()
        self.epsilon = Epsilon
        self.activation = nn.LeakyReLU()

    # прямой проход
    def forward(self, x):
        x = t.sigmoid(self.inp(x))
        x = t.sigmoid(self.hidden1(x))
        x = self.activation(self.out(x))
        return x

    # метод для запоминания i-го состояния, действия, награды и следующего состояния после выполнения
    def remember(self, state, action, reward, next_state, isDone):
        self.memory_states.append(state)
        self.memory_actions.append(action)
        self.memory_rewards.append(reward)
        self.memory_next_states.append(next_state)
        self.memory_isdones.append(1 - isDone)
        self.memory_len += 1
    

    # метод для получения тензоров 
    def samplebatch(self):
        states = t.FloatTensor(self.memory_states).cuda()
        self.memory_states.clear()

        actions = t.IntTensor(self.memory_actions).cuda()
        self.memory_actions.clear()

        rewards = t.FloatTensor(self.memory_rewards).cuda()
        self.memory_rewards.clear()

        next_states = t.FloatTensor(self.memory_next_states).cuda()
        self.memory_next_states.clear()
        
        isdones = t.IntTensor(self.memory_isdones).cuda()
        self.memory_isdones.clear()
        
        self.memory_len = 0
        return (states, actions, rewards, next_states, isdones)
        
    
    # метод для игры. Первый аргумент - среда, второй - максимальное кол-во эпизодов, третий - максимальное
    # число шагов в одном эпизоде
    def game(self, env, episodes=1000, maxsteps=1000):
        for i in range(episodes):
            state = env.reset()
            for j in range(maxsteps):
                if(random.random() < self.epsilon):
                    action = random.choice(range(4))
                else:
                    answer = self.forward((t.FloatTensor(state) * t.FloatTensor([5])).cuda())
                    print(answer)
                    action = t.argmax(answer).item()
                (next_state, reward, isDone, _) = env.step(action)
                self.remember(state, action, reward,  next_state, isDone)
                state = next_state
                if (isDone):
                    break

    def train_snake(self):
        if (self.memory_len == 0):
            return
        memory_len = self.memory_len
        states, actions, rewards, next_states, isdones = self.samplebatch()

        NeuroNowAnswer = self.forward(states)
        NeuroNextAnswer = self.forward(next_states)
        predicted_now_value = NeuroNowAnswer[range(memory_len), actions]
        predicted_future_value = t.max(NeuroNextAnswer, dim=1)[0]
        predict_target = rewards + predicted_future_value*isdones
        loss = self.criterion(predict_target, predicted_now_value)
        self.loses.append(loss.cpu().item())
        self.rewards_per_epoch.append(t.sum(rewards.cpu()).item())
        self.optimizer.zero_grad()
        loss.backward()
        if (self.inp.weight.grad.norm() < 0.0001):
            self.inp.weight.grad.data += t.FloatTensor([0.001]).cuda()
        self.optimizer.step()
        print(f"Ошибка: {loss}")
                    

            

if __name__ == '__main__':
    DEVICE = "cuda"


    snake = Snake()
    agent = SnakeAI(LearningRate=0.001, Epsilon=0.2).cuda()
    epoch = 10000
    epochs = list(range(epoch))
    for i in range(epoch):
        agent.game(snake, 1, 100)
        agent.train_snake()
        print(f"Завершено [{i}/{epoch}]")
    print("Тренировка завершена")
    plt.figure(1)
    plt.plot(epochs,agent.loses)
    plt.xlabel("Эпохи")
    plt.ylabel("Ошибка")
    plt.grid(True)

    plt.figure(2)
    plt.plot(epochs,agent.rewards_per_epoch)
    plt.xlabel("Эпохи")
    plt.ylabel("Награда за эпоху")
    plt.grid(True)
    plt.show()
