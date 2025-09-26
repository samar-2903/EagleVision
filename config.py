"""
Configuration constants for the traffic management pipeline.
"""

SUMO_BINARY = "sumo-gui"
SUMO_CFG = "grid_tls.sumocfg"
SIMULATION_STEPS = 3600
CONTROL_INTERVAL = 1

# Clustering parameters
DBSCAN_BASE_EPS = 20
DBSCAN_BASE_MIN_SAMPLES = 2
EPS_GROWTH_FACTOR = 0.1
EPS_DECAY_FACTOR = 0.1
MIN_SAMPLES_SCALE = 1

# Traffic light selection (None auto-detects the first available TLS)
TLS_ID = None


