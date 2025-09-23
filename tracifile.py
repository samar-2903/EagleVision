import traci
import traci.constants as tc

sumoBinary = "sumo-gui"  # or "sumo" for CLI
sumoCmd = [sumoBinary, "-c", "grid_tls.sumocfg"]

traci.start(sumoCmd)

tls_data = []

for step in range(3600):
    traci.simulationStep()
    for tls_id in traci.trafficlight.getIDList():
        phase = traci.trafficlight.getRedYellowGreenState(tls_id)
        tls_data.append((step, tls_id, phase))

traci.close()

# tls_data now has (time step, TL id, phase state) for all TLs
