import subprocess
import time
import random
import traci
import config as cfg


def start_sumo():
    port = random.randint(55000, 56000)
    sumo_cmd = [cfg.SUMO_BINARY, "-c", cfg.SUMO_CFG, "--start", "--remote-port", str(port)]
    print(f"Launching SUMO on port {port}...")
    subprocess.Popen(sumo_cmd)
    time.sleep(2)
    traci.init(port)
    print("Connected to SUMO.")
    return port


