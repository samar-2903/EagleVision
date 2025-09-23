import traci
import sumolib
import numpy as np
from sklearn.cluster import DBSCAN

# Set SUMO binary
SUMO_BINARY = "sumo-gui"  # or sumo if you want headless

# SUMO configuration file
SUMO_CFG = "grid_tls.sumocfg"

# Start SUMO as a subprocess
traci.start([SUMO_BINARY, "-c", SUMO_CFG])
