# =============================================================================
# config.py - Central configuration for simulation, training, and evaluation.
# =============================================================================

# --- SUMO / Simulation ---
SUMO_BINARY = "sumo"
SUMO_CFG = "nets/grid_tls.sumocfg"
SIMULATION_STEPS = 1500
CONTROL_INTERVAL = 5
TLS_ID = None                     # If set, only control this one traffic light
CONTROLLED_TLS_IDS = None         # Optional explicit list of traffic lights to control
MAX_CONTROLLED_TLS = None         # Optional cap when auto-detecting multi-phase signals
SEED = 42

# --- State space ---
# Per-intersection local state:
# [Q_N, Q_S, Q_E, Q_W,
#  speed_N, speed_S, speed_E, speed_W,
#  arrival_N, arrival_S, arrival_E, arrival_W,
#  cluster_growth,
#  accident_flag,
#  risk_score]
STATE_DIM = 15

# --- Action space ---
# 0 = Give green to North-South
# 1 = Give green to East-West
# 2 = Extend current green phase
# 3 = Force short green on the most congested axis
NUM_ACTIONS = 4

# --- DQN Hyperparameters ---
GAMMA = 0.95
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_FREQ = 200
LEARN_UPDATES_PER_STEP = 4

# --- Exploration ---
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995

# --- Neural Network ---
HIDDEN_DIM = 128

# --- Reward weights ---
ALPHA = 1.0
BETA = 0.1
GAMMA_ACC = 5.0
DELTA = 0.2
Q_NORM = 50.0
D_NORM = 100.0
G_NORM = 10.0
GLOBAL_QUEUE_WEIGHT = 0.30
JAM_PENALTY = 1.50

# --- Training ---
NUM_EPISODES = 500
SAVE_FREQ = 50
LOG_FREQ = 10
MODEL_SAVE_PATH = "checkpoints/dqn_checkpoint.pt"
TRAIN_USE_GUI = False
TRAIN_STEP_LOG = False
TRAIN_STEP_LOG_INTERVAL = 1
TRAIN_STEP_SLEEP_S = 0.0
PRE_JAM_TRANSITION_WINDOW = 3
PRE_JAM_REPLAY_MULTIPLIER = 2

# --- Phase durations (seconds) ---
MIN_GREEN = 5
MAX_GREEN = 60
DEFAULT_GREEN = 30

# --- Accident modeling ---
ACCIDENT_RATE_SCALE = 0.015
ACCIDENT_MAX_PROB_PER_STEP = 0.10
ACCIDENT_BLOCK_SPEED = 0.20

# --- Jam detection / dynamic demand throttling ---
JAM_QUEUE_THRESHOLD = 180.0
JAM_QUEUE_JUMP_THRESHOLD = 35.0
JAM_RECOVERY_THRESHOLD = 90.0
JAM_KEEP_PROB = 0.60
JAM_KEEP_PROB_MIN = 0.30
JAM_KEEP_PROB_RECOVER = 0.05
JAM_KEEP_PROB_STEP_DOWN = 0.10

# --- Normalization bounds for state ---
MAX_QUEUE = 100.0
MAX_SPEED = 15.0
MAX_ARRIVAL = 2.0
MAX_GROWTH = 5.0
