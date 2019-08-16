state_modes = ['senor', 'vis', 'sensor_vis']

# parameters for DDPG
MAX_EPISODES = 1200
MAX_EP_STEPS = 240
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 3000
BATCH_SIZE = 64
VAR_MIN = 0.2
VAR_MAX = 1.5  # control exploration
DONE_STEPS = 20

LOAD_EXP = True
LOAD_OFF_WEIGHTS = False  # load off-policy trained
LOAD_ON_WEIGHTS = False  # load on-policy trained
