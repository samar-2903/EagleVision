import numpy as numpy
import traci
import subprocess
from sklearn.cluster import DBSCAN
import time 
import random 

DBSCAN_BASE_EPS = 20
DBSCAN_BASE_MIN_SAMPLES = 2
EPS_GROWTH_FACTOR = 0.1
EPS_DECAY_FACTOR = 0.1
MIN_SAMPLES_SCALE = 1