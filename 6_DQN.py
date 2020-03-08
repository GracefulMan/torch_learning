import gym
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import time
# Hyper parameters
BATCH_SIZE = 128
LR = 0.01 #learning rate
EPSION = 0.7 # greedy policy
GAMMA = 0.9 # reward discount
TARGET_REPLACE_ITER = 50 # target update frequency
MEMORY_CAPACITY = 5000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTION = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1) # initializtion
        self.out = nn.Linear(100, N_ACTION)
        self.out.weight.data.normal_(0, 0.1) # initializtion

    def forward(self,x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, memory=None):
        self.eval_net, self.target_net = NET(), NET()
        self.learn_step_counter = 0 # for target updating
        self.memory_counter = 0
        if memory == None:
            self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) # initizalize memory
        else:
            self.memory = memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSION: # greedy
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else: #random
            action = np.random.randint(0, N_ACTION)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        return action

    def store_transition(self,s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES +1 : N_STATES +2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES :]))
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




dqn = DQN()
print("\n Collecting experience...")
for i_episode in range(550):
    s = env.reset()
    print('episode: {}'.format(i_episode))
    while True:
        env.render()
        a = dqn.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot =s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.8
        r = r1 + r2
        dqn.store_transition(s, a, r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
        if done:
            break
        s = s_
print('test...')
env.reset()
total_zero =0
EPSION = 1
for i_episode in range(10000):
    print('epision:{}'.format(i_episode))
    print('episode: {}'.format(i_episode))
    while True:
        env.render()
        a = dqn.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)
        if r < 0.1:
            total_zero += 1
        else:
            total_zero = 0
        s = s_
        if total_zero >= 100:
            env.reset()

env.close()


