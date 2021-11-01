import argparse
import pickle
from collections import namedtuple
from itertools import count
import pandas as pd
import os, time
import numpy as np
import matplotlib.pyplot as plt
from Env import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from torchviz import make_dot

# Parameters
gamma = 0.995
seed = 1

ref_data = pd.read_csv(r'data/软件园300.csv', encoding='gbk')
arrive = ref_data['arrive'].values
dwell = ref_data['dwell'].values
depart = np.array([arrive[i] + dwell[i] for i in range(len(arrive))])
event = np.append(arrive, depart)
ind = np.argsort(event)
ind = [i + 1 for i in ind]
event = np.sort(event)
demand = len(arrive)
k = [2,2,2]
stack = 10
abbr = '10-C'
env = Simulation(event, ind, demand, dwell, k, stack)
action_space = env.action_space
layout_space = env.layout_space
veh_space = env.veh_space

torch.manual_seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 6, 1, 1) # (stack + 2 - 3 - 1) / 1 + 1 = stack-1
        self.conv2 = nn.Conv2d(32, 64, 6, 1, 1) # stack - 3
        self.conv3 = nn.Conv2d(64, 128, 6, 1, 1) # stack - 5
        self.flatten = nn.Flatten() # out = 16 * 5 * 7 = 560
        self.fc1 = nn.Linear(128 * (layout_space[1] - 9) * (layout_space[2] - 9), 600)
        # self.fc2 = nn.Linear(1000, 600)
        self.fc3 = nn.Linear(600, 400)
        self.fc4 = nn.Linear(400, 200)
        self.fc5 = nn.Linear(200, 40)
        self.fcv = nn.Linear(veh_space, 40)
        self.fco1 = nn.Linear(80, 60)
        self.fco2 = nn.Linear(60, 40)
        self.fco3 = nn.Linear(40, action_space)

    def forward(self, layout_input, veh_input):
        x = self.conv1(layout_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(self.fc5(x))
        y = F.relu(self.fcv(veh_input))
        z = torch.cat((x, y),1)
        o = F.relu(self.fco1(z))
        o = self.fco2(o)
        o = self.fco3(o)
        o = F.softmax(o, dim=1)
        return o

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 6, 1, 1) # (stack + 2 - 3 - 1) / 1 + 1 = stack-1
        self.conv2 = nn.Conv2d(32, 64, 6, 1, 1) # stack - 3
        self.conv3 = nn.Conv2d(64, 128, 6, 1, 1) # stack - 5
        self.flatten = nn.Flatten() # out = 16 * 5 * 7 = 560
        self.fc1 = nn.Linear(128 * (layout_space[1] - 9) * (layout_space[2] - 9), 600)
        # self.fc2 = nn.Linear(1000, 600)
        self.fc3 = nn.Linear(600, 400)
        self.fc4 = nn.Linear(400, 200)
        self.fc5 = nn.Linear(200, 40)
        self.fcv = nn.Linear(veh_space, 40)
        self.fco1 = nn.Linear(80, 40)
        self.fco2 = nn.Linear(40, 20)
        self.fco3 = nn.Linear(20, 10)
        self.fco4 = nn.Linear(10, 1)

    def forward(self, layout_input, veh_input):
        x = self.conv1(layout_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(self.fc5(x))
        y = F.relu(self.fcv(veh_input))
        z = torch.cat((x, y),1)
        o = F.relu(self.fco1(z))
        o = self.fco2(o)
        o = self.fco3(o)
        o = self.fco4(o)
        return o

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 5
    buffer_capacity = 10000
    batch_size = 64

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().cuda()
        self.critic_net = Critic().cuda()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        # self.writer = SummaryWriter('./exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-5)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-5)
        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    def select_action(self, state):
        layout_input = torch.from_numpy(state[0]).float().unsqueeze(0).cuda()
        veh_input = torch.from_numpy(state[1]).float().unsqueeze(0).cuda()
        with torch.no_grad():
            action_prob = self.actor_net(layout_input, veh_input)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        layout_input = torch.from_numpy(state[0]).float().unsqueeze(0)
        veh_input = torch.from_numpy(state[1]).float().unsqueeze(0)
        with torch.no_grad():
            value = self.critic_net(layout_input, veh_input)
        return value.item()

    def save_param(self, episode):
        folder = './param/net_param/'+ abbr
        actor_path = folder +'/actor_net_{}.pkl'.format(episode)
        critic_path = folder +'/critic_net_{}.pkl'.format(episode)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)

    def load_param(self):
        actor_path = './param/net_param/'+ abbr +'/actor_net_1999.pkl'
        critic_path = './param/net_param/'+ abbr +'/critic_net_1999.pkl'
        self.actor_net.load_state_dict(torch.load(actor_path))
        self.critic_net.load_state_dict(torch.load(critic_path))


    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self, i_ep):
        layout_input = torch.tensor([t.state[0] for t in self.buffer], dtype=torch.float).cuda()
        veh_input = torch.tensor([t.state[1] for t in self.buffer], dtype=torch.float).cuda()
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).cuda()
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).cuda()

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1).cuda()
                V = self.critic_net(layout_input[index], veh_input[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(layout_input[index], veh_input[index]).gather(1, action[index]) # new policy
                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean().cuda()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V).cuda()
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

    
def train():
    agent = PPO()
    history = [[],[]]
    for i_epoch in range(2000):
        history[0].append(i_epoch)
        state = env.reset()
        reward_sum = 0
        for t in count():
            action, action_prob = agent.select_action(state)
            action_y = (action) % layout_space[1]
            action_x = (action) // layout_space[1]
            action_convert = [int(action_x), int(action_y)]
            next_state, reward, done, _ = env.step(action_convert)
            reward_sum += reward
            trans = Transition(state, action, action_prob, reward, next_state)
            agent.store_transition(trans)
            state = next_state

            if done :
                if i_epoch % 1 == 0:
                    print('i_epoch{} ---- reward:{}\n'.format(i_epoch, reward_sum))
                if i_epoch % 100 == 0:
                    agent.save_param(i_epoch)
                history[1].append(reward_sum)
                if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)
                # agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
    env.reset()
    agent.save_param(1999)
    outdata = pd.DataFrame(history).T
    outdata.columns = ['episode','reward']
    out_folder = './res/'+ abbr
    outdata_path = out_folder + '/outdata.csv'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    outdata.to_csv(outdata_path)


def evaluate():
    agent = PPO()
    agent.load_param()
    state = env.reset()
    reward_sum = 0
    for t in count():
        action, action_prob = agent.select_action(state)
        action_y = (action) % layout_space[1]
        action_x = (action) // layout_space[1]
        action_convert = [int(action_x), int(action_y)]
        next_state, reward, done, _ = env.step(action_convert)
        reward_sum += reward
        trans = Transition(state, action, action_prob, reward, next_state)
        agent.store_transition(trans)
        state = next_state

        if done :
            print('reward:{}---relocate:{}----reject:{}----block:{}\n'.format(reward_sum, env.sim_relocate, env.sim_reject, env.sim_block))
            break


if __name__ == '__main__':
    # train()
    evaluate()
