
import tensorflow as tf
import numpy as np
import os
import shutil
import argparse
from argparse import ArgumentParser

#from arm_env import ArmEnv
from avant_env import *
from glob import glob
import random
import avant_para

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


def train(obs_mode=avant_para.state_modes[2]):
    global env
    if obs_mode == avant_para.state_modes[2]:
        STATE_DIM = 11  # 8 vis + 4 senor states: Angle boom, Angle bucket, Length telescope, TransmissionPressureSensor16 -TransmissionPressureSensor13
    elif obs_mode == avant_para.state_modes[0]:
        STATE_DIM = 3  # 4 senor states: Angle boom, Angle bucket, Length telescope, TransmissionPressureSensor16 -TransmissionPressureSensor13
    elif obs_mode == avant_para.state_modes[1]:
        STATE_DIM = 8  # 8 vis
    else:
        print('wrong states modes\n choose one from:', avant_para.state_modes)
    ACTION_DIM = 3  # 4 actions: Steering command, Boom command, Bucket command, Telescope command

    ACTION_BOUND = [-1, 1]

    global S
    global R
    global S_
    var = avant_para.VAR_MAX

    # all placeholder for tf
    with tf.name_scope('S'):
        S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
        print(S)
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')

    sess = tf.Session()

    # Create actor and critic.
    actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], avant_para.LR_A, avant_para.REPLACE_ITER_A)
    critic = Critic(sess, STATE_DIM, ACTION_DIM, avant_para.LR_C,
                    avant_para.GAMMA, avant_para.REPLACE_ITER_C, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    M = Memory(avant_para.MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

    saver = tf.train.Saver()

    # create the name for path based on different configurations
    extention = 'weights/RL_weights/DDPG_' + obs_mode
    path = './' + extention
    off_policy_weights_path = path + '/off-policy'
    # create output path
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(off_policy_weights_path):
        os.mkdir(off_policy_weights_path)

    ckpt_path = os.path.join('./' + extention, 'DDPG.ckpt')
    log_path = os.path.join('./' + extention, 'train_log')
    if obs_mode == avant_para.state_modes[2]:
        exp_path = 'exps/vis_sensor'
    elif obs_mode == avant_para.state_modes[1]:
        exp_path = 'exps/vis'
    elif obs_mode == avant_para.state_modes[0]:
        exp_path = 'exps/sensor'
    else:
        print('wrong state mode')

    # if load pretrained weights
    if avant_para.LOAD_ON_WEIGHTS:
        # if the pretrained-weights exists, then load
        print('loading on policy trained weights')
        saver.restore(sess, tf.train.latest_checkpoint(path))
    elif avant_para.LOAD_OFF_WEIGHTS:
        print('loading off policy trained weights')
        saver.restore(sess, tf.train.latest_checkpoint(off_policy_weights_path))
    else:
        sess.run(tf.global_variables_initializer())

    # if load experiences
    if not os.path.isdir(exp_path) and not avant_para.LOAD_EXP:
        os.mkdir(exp_path)
    elif avant_para.LOAD_EXP:
        # if experiences are stored in the folder, then pretrain
        exp_all = glob(exp_path + '/*')
        #print(exp_path, exp_all)
        exp_to_load = random.sample(exp_all, avant_para.MEMORY_CAPACITY)
        for i in exp_to_load:
            exp = np.load(i)
            print(exp)
            M.store_prev_transition(exp)

        for e in range(50):
            b_M = M.sample(avant_para.BATCH_SIZE)
            b_s = b_M[:, :STATE_DIM]
            b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
            b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
            b_s_ = b_M[:, -STATE_DIM:]

            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)
            print('loss', critic.loss)

        # save the off-policy pretrained weights
        ckpt_ep_path = os.path.join(off_policy_weights_path, 'DDPG_offPolicy' + '.ckpt')
        save_ep_path = saver.save(sess, ckpt_ep_path, write_meta_graph=False)

    for ep in range(avant_para.MAX_EPISODES):
        s, gt_ = reset_avant(obs_mode)  # give the command to Avant, so it can move back to start point
        ep_reward = 0

        for t in range(avant_para.MAX_EP_STEPS):
            # Added exploration noise
            a = actor.choose_action(s)
            # add randomness to action selection for exploration
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)
            s_, r = command_to_avant(a, obs_mode)
            M.store_transition(s, a, r, s_)

            if M.pointer > avant_para.MEMORY_CAPACITY:  # if the experience pool is full then update A,C nets
                # print(M.pointer)
                # decay the action randomness
                var = max([var * .9999, avant_para.VAR_MIN])
                for i in range(2):
                    b_M = M.sample(avant_para.BATCH_SIZE)
                    b_s = b_M[:, :STATE_DIM]
                    b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                    b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                    b_s_ = b_M[:, -STATE_DIM:]

                    critic.learn(b_s, b_a, b_r, b_s_)
                    actor.learn(b_s)

            s = s_
            ep_reward += r

            if t == avant_para.MAX_EP_STEPS - 1:  # when 1 episode is over

                if M.pointer > avant_para.MEMORY_CAPACITY:
                    for i in range(2):
                        b_M = M.sample(avant_para.BATCH_SIZE)
                        b_s = b_M[:, :STATE_DIM]
                        b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                        b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                        b_s_ = b_M[:, -STATE_DIM:]

                        critic.learn(b_s, b_a, b_r, b_s_)
                        actor.learn(b_s)
                # if done:
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )

                if ep % 100 == 0 and ep != 0:
                    ckpt_ep_path = os.path.join('./' + extention, 'DDPG_' + str(ep) + '.ckpt')
                    save_ep_path = saver.save(sess, ckpt_ep_path, write_meta_graph=False)

                break

    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


train(obs_mode=avant_para.state_modes[2])
