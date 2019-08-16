
import tensorflow as tf

import numpy as np
import os
import shutil
import argparse
from argparse import ArgumentParser

#from arm_env import ArmEnv
import avant_para
from crane_env import c_env
from glob import glob
import random

np.random.seed(1)
tf.set_random_seed(1)

global env
global STATE_DIM
global ACTION_DIM
global ACTION_BOUND

# all placeholder for tf
global S
global R
global S_


class Actor(object):

    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        global S
        global R
        global S_

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                # Scale output to -action_bound to action_bound
                scaled_a = tf.multiply(
                    actions, self.action_bound, name='scaled_a')
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e)
                           for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(
                ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            # (- learning rate) for ascent policy
            opt = tf.train.RMSPropOptimizer(-self.lr)
            self.train_op = opt.apply_gradients(
                zip(self.policy_grads, self.e_params))


class Critic(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        global S
        global R
        global S_

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            # target_q is based on a_ from Actor's target_net
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)

            self.e_params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            # tensor of gradients of each sample (None, a_dim)
            self.a_grads = tf.gradients(self.q, a)[0]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable(
                    'w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable(
                    'w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w,
                                    bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e)
                           for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):

    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def store_prev_transition(self, transition):
        #transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


class DDPG():
    MAX_EPISODES = avant_para.MAX_EPISODES
    MAX_EP_STEPS = avant_para.MAX_EP_STEPS
    DONE_STEPS = avant_para.DONE_STEPS
    LR_A = avant_para.LR_A  # learning rate for actor
    LR_C = avant_para.LR_C  # learning rate for critic
    GAMMA = avant_para.GAMMA  # reward discount
    REPLACE_ITER_A = avant_para.REPLACE_ITER_A
    REPLACE_ITER_C = avant_para.REPLACE_ITER_C
    MEMORY_CAPACITY = avant_para.MEMORY_CAPACITY
    BATCH_SIZE = avant_para.BATCH_SIZE

    VAR_MIN = avant_para.VAR_MIN
    VAR_MAX = avant_para.VAR_MAX  # control exploration

    #env = ArmEnv(mode=MODE[n_model])
    env = None
    STATE_DIM = None
    ACTION_DIM = None
    ACTION_BOUND = None

    # all placeholder for tf
    S = None
    R = None
    S_ = None

    def __init__(self,
                 env_args,
                 load_exps=False,
                 load_on_weights=False,
                 load_off_weights=True,
                 batch_size=64,
                 memory_capacity=3000,
                 max_eps=1200,
                 max_steps=90):

        self.states = None
        self.reward = None
        self.action = None
        self.next_states = None
        self.ep_reward = None
        self.done = False
        self.step_count = 0
        self.success_count = 0  # count how many frames are classified as Success
        self.ep_num = 0

        self.env = None

        self.LOAD_EXP = load_exps
        self.LOAD_ON_WEIGHTS = load_on_weights
        self.LOAD_OFF_WEIGHTS = load_off_weights
        self.BATCH_SIZE = batch_size
        self.MEMORY_CAPACITY = memory_capacity
        self.MAX_EPISODES = max_eps
        self.MAX_EP_STEPS = max_steps

        self.saver = None

        self.sess = tf.Session()
        self.set_env_arg(t_type=arguments.t_type,
                         n_type=arguments.n_type,
                         r_type=arguments.r_type,
                         proj=str_to_bool(arguments.proj),
                         cam_r_noise=str_to_bool(arguments.cam_r_noise),
                         cam_t_noise=str_to_bool(arguments.cam_t_noise),
                         cam_in_noise=str_to_bool(arguments.cam_in_noise),
                         test=False)

        # making paths
        # create the name for path based on different configurations
        self.extention = 'weights/RL_weights/DDPG_'
        self.path = './' + self.extention
        self.off_policy_weights_path = self.path + '/off-policy'
        # create output path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        if not os.path.isdir(self.off_policy_weights_path):
            os.mkdir(self.off_policy_weights_path)

        self.ckpt_path = os.path.join('./' + self.extention, 'DDPG.ckpt')
        self.log_path = os.path.join('./' + self.extention, 'train_log')
        self.exp_path = 'exps/'

    def set_env_arg(self, t_type='easy', n_type="small", r_type="2_corners", proj=True, cam_r_noise=False, cam_t_noise=False, cam_in_noise=False, test=False):

        global S
        global R
        global S_
        # input MODE = ['easy', '2_corners', '4_corners',  'all', 'partial_all']
        # noise mode = [small,middle,big,hardest]
        print(t_type, n_type, r_type, proj, cam_r_noise, cam_t_noise, cam_in_noise)
        self.env = c_env(obs_mode=t_type,
                         noise_mode=n_type,
                         reward_type=r_type,
                         projection=proj,
                         cam_R_noise=cam_r_noise,
                         cam_T_noise=cam_t_noise,
                         cam_IN_noise=cam_in_noise,
                         test=test)

        self.STATE_DIM = self.env.state_dim
        self.ACTION_DIM = self.env.action_dim
        self.ACTION_BOUND = self.env.action_bound

        # all placeholder for tf
        with tf.name_scope('S'):
            S = tf.placeholder(tf.float32, shape=[None, self.STATE_DIM], name='s')
            print(self.S)
        with tf.name_scope('R'):
            R = tf.placeholder(tf.float32, [None, 1], name='r')
        with tf.name_scope('S_'):
            S_ = tf.placeholder(tf.float32, shape=[None, self.STATE_DIM], name='s_')

        # Create actor and critic.
        self.actor = Actor(self.sess, self.ACTION_DIM, self.ACTION_BOUND[1], self.LR_A, self.REPLACE_ITER_A)
        self.critic = Critic(self.sess, self.STATE_DIM, self.ACTION_DIM, self.LR_C,
                             self.GAMMA, self.REPLACE_ITER_C, self.actor.a, self.actor.a_)
        self.actor.add_grad_to_graph(self.critic.a_grads)
        self.M = Memory(self.MEMORY_CAPACITY, dims=2 * self.STATE_DIM + self.ACTION_DIM + 1)

        self.saver = tf.train.Saver()

    def load_weights(self, on_policy=False, off_policy=True):

        # if load pretrained weights
        if on_policy and not off_policy:
            # if the pretrained-weights exists, then load
            print('loading on policy trained weights')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path))
        elif off_policy and not on_policy:
            print('loading off policy trained weights')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.off_policy_weights_path))
        else:
            self.sess.run(tf.global_variables_initializer())

    def load_exps(self, iteration=50, save_weights=False):
        print("--------------load exps------------------")
        # if load experiences
        if not os.path.isdir(self.exp_path):
            os.mkdir(self.exp_path)
        else:
            # if experiences are stored in the folder, then pretrain
            exp_all = glob(self.exp_path + '/*')
            #print(exp_path, exp_all)
            exp_to_load = random.sample(exp_all, self.MEMORY_CAPACITY)
            for i in exp_to_load:
                exp = np.load(i)
                # print(exp)
                self.M.store_prev_transition(exp)

            self.train_AC(iteration)

            if save_weights:
                # save the off-policy pretrained weights
                ckpt_ep_path = os.path.join(self.off_policy_weights_path, 'DDPG_offPolicy' + '.ckpt')
                save_ep_path = self.saver.save(self.sess, self.ckpt_ep_path, write_meta_graph=False)

    def train_AC(self, iteration):
        if self.M.pointer > avant_para.MEMORY_CAPACITY - 1:
            for e in range(iteration):
                b_M = self.M.sample(self.BATCH_SIZE)
                b_s = b_M[:, :self.STATE_DIM]
                b_a = b_M[:, self.STATE_DIM: self.STATE_DIM + self.ACTION_DIM]
                b_r = b_M[:, -self.STATE_DIM - 1: -self.STATE_DIM]
                b_s_ = b_M[:, -self.STATE_DIM:]

                self.critic.learn(b_s, b_a, b_r, b_s_)
                self.actor.learn(b_s)
                #print('loss', self.critic.loss)

    def get_states(self, sensors):
        states = sensors[:len(sensors) - 1]
        reward = sensors[len(sensors) - 1]

        return states, reward

    def train_steps(self,  sensors, new_ep):
        #print(self.ep_num, new_ep, self.step_count)
        if new_ep:
            self.ep_num += 1
            self.step_count = 0
            self.success_count = 0
            self.done = False
            self.next_states = None
            self.reward = None
            self.ep_reward = 0

            self.states, _ = self.get_states(sensors)
            a = self.actor.choose_action(self.states)
            self.action = np.clip(np.random.normal(a, self.VAR_MAX), *self.ACTION_BOUND)

        else:
            self.next_states, self.reward = self.get_states(sensors)
            if len(self.next_states) != 0 and self.reward != None and len(self.action) != 0:
                self.M.store_transition(self.states, self.action, self.reward, self.next_states)

            a = self.actor.choose_action(self.states)
            self.action = np.clip(np.random.normal(a, self.VAR_MAX), *self.ACTION_BOUND)
            self.states = self.next_states
            self.ep_reward += self.reward
            if self.reward > 0.9:
                self.success_count += 1

        if self.M.pointer > self.MEMORY_CAPACITY:  # if the experience pool is full then update A,C nets
            # print(M.pointer)
            # decay the action randomness
            self.VAR_MAX = max([self.VAR_MAX * .9999, self.VAR_MIN])

            self.train_AC(2)

        if self.step_count > self.MAX_EP_STEPS:
            self.train_AC(5)
            print('Ep:', self.ep_num,
                  done,
                  '| R: %i' % int(self.ep_reward),
                  '| Explore: %.2f' % self.VAR_MAX,
                  '| Steps %i', self.step_count,
                  self.step_count > self.MAX_EP_STEPS
                  )
            if self.ep_num % 10 == 0 and self.ep_num != 0:
                ckpt_ep_path = os.path.join('./' + self.extention, 'DDPG_' + str(self.ep_num) + '.ckpt')
                save_ep_path = self.saver.save(self.sess, ckpt_ep_path, write_meta_graph=False)

        finished = self.step_count > self.MAX_EP_STEPS
        self.step_count += 1

        return self.action, finished


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t_type', type=str, default="no_symmetric_corners_simple")
    parser.add_argument('--n_type', type=str, default="none")
    parser.add_argument('--r_type', type=str, default="2_corners")
    parser.add_argument('--proj', type=str, default="True")
    parser.add_argument('--test', type=str, default="True")
    parser.add_argument('--load_path', type=str, default="None")
    parser.add_argument('--cam_r_noise', type=str, default="False")
    parser.add_argument('--cam_t_noise', type=str, default="False")
    parser.add_argument('--cam_in_noise', type=str, default="False")
    return parser.parse_args()


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError  # evil ValueError


if __name__ == '__main__':

    arguments = get_args()
    proj = False

    rl = DDPG(arguments)
    s, _ = rl.env.reset()
    rl.load_weights(on_policy=False, off_policy=False)

    s = np.hstack((s, 0))

    RESET_DONE = True  # tell if the avant is reset
    finished = False  # tell if current ep finished
    new_ep = True  # tell RL if it is a new episode
    episode_cnt = 0
    while True:  # 50ms loop
        if RESET_DONE and not finished:
            rl.env.render()
            action, done = rl.train_steps(s, new_ep)
            finished = done
            new_ep = False
            #print("finished", finished)
            st, rw, d, _ = rl.env.step(action)
            s = np.hstack((st, np.array([rw])))
        else:
            new_ep = True
            s, _ = rl.env.reset()
            s = np.hstack((s, 0))
            print("reset")
            RESET_DONE = True
            finished = False

        episode_cnt += 1


"""
 # checking whether to run RL or Avant reset
        if RESET_DONE: # Avant is ready to run RL
            # do RL step, save the history and continue RL until reached end of episode
            cmd, done_flag = rl.train_steps(s, new_ep)
            if(done_flag)
                RL_DONE = done_flag
                
        else: # reset avant
            #cmd, done_flag = avant_reset(inData.data)
            RESET_DONE = done_flag

        episode_cnt += 1
"""
