import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac.sac import SAC
from tensorboardX import SummaryWriter
from sac.replay_memory import ReplayMemory

from crane_env import c_env

global env
global STATE_DIM
global ACTION_DIM
global ACTION_BOUND
global action_dim
global state_dim

MAX_STEP = 120
MAX_EP_STEPS = 1200
sample_numsteps = 10
var = .95
VAR_MIN = 0.11


def set_env_arg(t_type='easy', n_type="small", r_type="2_corners", proj=True, cam_r_noise=False, cam_t_noise=False, cam_in_noise=False, test=False):
    # input MODE = ['easy', '2_corners', '4_corners',  'all', 'partial_all']
    # noise mode = [small,middle,big,hardest]

    global env
    global STATE_DIM
    global ACTION_DIM
    global ACTION_BOUND

    global S
    global R
    global S_

    print(t_type, n_type, r_type, proj, cam_r_noise, cam_t_noise, cam_in_noise)
    env = c_env(obs_mode=t_type,
                noise_mode=n_type,
                reward_type=r_type,
                projection=proj,
                cam_R_noise=cam_r_noise,
                cam_T_noise=cam_t_noise,
                cam_IN_noise=cam_in_noise,
                test=test)

    STATE_DIM = env.state_dim
    ACTION_DIM = env.action_dim
    ACTION_BOUND = env.action_bound


def get_args():
    parser = argparse.ArgumentParser(
        description='Train or test neural net motor controller.')
    parser.add_argument('--t_type', type=str, default="no_symmetric_corners_simple")
    parser.add_argument('--n_type', type=str, default="none")
    parser.add_argument('--r_type', type=str, default="2_corners")
    parser.add_argument('--proj', type=str, default="True")
    parser.add_argument('--test', type=str, default="True")
    parser.add_argument('--load_path', type=str, default="None")
    parser.add_argument('--cam_r_noise', type=str, default="False")
    parser.add_argument('--cam_t_noise', type=str, default="False")
    parser.add_argument('--cam_in_noise', type=str, default="False")

    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--policy', default="Gaussian",
                        help='algorithm to use: Gaussian | Deterministic')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default:True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--seed', type=int, default=456, metavar='N',
                        help='random seed (default: 456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        metavar='G', help='learning rate (default: 0.0003)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Temperature parameter α automaically adjusted.')
    args = parser.parse_args()

    return parser.parse_args()


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError  # evil ValueError


def train():
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    #env = gym.make(args.env_name)
    global var
    args = get_args()
    set_env_arg(t_type=args.t_type,
                n_type=args.n_type,
                r_type=args.r_type,
                proj=str_to_bool(args.proj),
                cam_r_noise=str_to_bool(args.cam_r_noise),
                cam_t_noise=str_to_bool(args.cam_t_noise),
                cam_in_noise=str_to_bool(args.cam_in_noise),
                test=str_to_bool(args.test))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.seed(args.seed)

    # Agent
    agent = SAC(env.state_dim, env.action_space, args)
    agent.load_model('models/sac_actor_crane70_', 'models/sac_critic_crane70_')

    # TesnorboardX
    writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for ep in range(MAX_EP_STEPS):
        state, gt = env.reset()
        episode_reward = 0

        for t in range(MAX_STEP):
                    # while True:
            env.render()

            # Added exploration noise
            if ep < sample_numsteps:
                print('sample')
                action = env.action_space.sample()  # Sample random action
            else:
                # Sample action from policy
                action = agent.select_action(state)

            # add randomness to action selection for exploration
            action = np.clip(np.random.normal(action, var), *ACTION_BOUND)
            next_state, reward, done, _ = env.step(action)  # Step
            if done:
                mask = 1
            else:
                mask = 0
            memory.push(state, action, reward, next_state,
                        mask)  # Append transition to memory

            """# store experience
                    trans = np.hstack((s, a, [r], s_))
                    outfile = exp_path + '/' + str(ep) + '_' + str(t)
                    np.save(outfile, trans)
                    """

            if len(memory) > sample_numsteps * MAX_STEP:
                # Number of updates per step in environment
                var = max([var * .9999, VAR_MIN])
                for i in range(1):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, 512, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar(
                        'entropy_temprature/alpha', alpha, updates)
                    updates += 1

            state = next_state

            episode_reward += reward

            if t == MAX_STEP - 1 or done:
                if len(memory) > sample_numsteps * MAX_STEP:
                    for i in range(10):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                            memory, 512, updates)

                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar(
                            'entropy_temprature/alpha', alpha, updates)
                        updates += 1

                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(episode_reward),
                      '| Explore: %.2f' % var,
                      )

                out_s = 'Ep: ' + str(ep) + ' result: ' + str(done) + \
                    " R: " + str(episode_reward) + " Explore " + str(var) + " \n"
                break
                """
                    f = open(log_path, "a+")
                    f.write(out_s)
                    f.close()
                    """
            if ep % 10 == 0:
                agent.save_model(env_name='crane' + str(ep))

    agent.save_model(env_name='crane')


def test():
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    #env = gym.make(args.env_name)
    args = get_args()
    args.eval = True
    set_env_arg(t_type=args.t_type,
                n_type=args.n_type,
                r_type=args.r_type,
                proj=str_to_bool(args.proj),
                cam_r_noise=str_to_bool(args.cam_r_noise),
                cam_t_noise=str_to_bool(args.cam_t_noise),
                cam_in_noise=str_to_bool(args.cam_in_noise),
                test=str_to_bool(args.test))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.seed(args.seed)

    # Agent
    agent = SAC(env.state_dim, env.action_space, args)
    agent.load_model('models/sac_actor_crane_', 'models/sac_critic_crane_')

    # TesnorboardX
    writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for ep in range(MAX_EP_STEPS):
        state, gt = env.reset()
        episode_reward = 0

        for t in range(MAX_STEP):
                    # while True:
            env.render()

            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)  # Step
            if done:
                mask = 1
            else:
                mask = 0
            memory.push(state, action, reward, next_state,
                        mask)  # Append transition to memory

            """# store experience
                    trans = np.hstack((s, a, [r], s_))
                    outfile = exp_path + '/' + str(ep) + '_' + str(t)
                    np.save(outfile, trans)
                    """

            state = next_state

            episode_reward += reward

            if t == MAX_STEP - 1 or done:

                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(episode_reward),
                      '| Explore: %.2f' % var,
                      )

                out_s = 'Ep: ' + str(ep) + ' result: ' + str(done) + \
                    " R: " + str(episode_reward) + " Explore " + str(var) + " \n"
                """
                    f = open(log_path, "a+")
                    f.write(out_s)
                    f.close()
                    """

train()
