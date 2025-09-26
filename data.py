import traci
from collections import deque, defaultdict

_prev_pos = {}
_prev_t = None


def _estimate_speed(veh_id, pos_xy, t_s):
    global _prev_pos, _prev_t
    if _prev_t is None:
        _prev_t = float(t_s)
    dt = max(1e-6, float(t_s) - float(_prev_t))
    px, py = _prev_pos.get(veh_id, pos_xy)
    vx = float(pos_xy[0]) - float(px)
    vy = float(pos_xy[1]) - float(py)
    dist = (vx * vx + vy * vy) ** 0.5
    _prev_pos[veh_id] = pos_xy
    return float(dist / dt)


def get_vehicle_positions():
    vehicle_data = []
    for vehID in traci.vehicle.getIDList():
        v_type = traci.vehicle.getTypeID(vehID)
        pos = traci.vehicle.getPosition(vehID)
        speed = traci.vehicle.getSpeed(vehID)
        lane_id = traci.vehicle.getLaneID(vehID)
        wait_t = traci.vehicle.getWaitingTime(vehID)
        vehicle_data.append({
            "id": vehID,
            "type": v_type,
            "position": pos,
            "speed": speed,
            "lane": lane_id,
            "wait": wait_t
        })
    return vehicle_data


def get_vehicle_features(t_s: float, s_th: float = 0.5, D_q: float = 10.0):
    feats = []
    for vehID in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition(vehID)
        lane_id = traci.vehicle.getLaneID(vehID)
        lane_pos = traci.vehicle.getLanePosition(vehID)
        lane_len = traci.lane.getLength(lane_id) if lane_id else 0.0
        d_stop = max(0.0, float(lane_len) - float(lane_pos)) if lane_len else 0.0
        spd_est = _estimate_speed(vehID, pos, t_s)
        stopped = int(spd_est < s_th)
        feats.append({
            "id": vehID,
            "position": pos,
            "lane": lane_id,
            "lane_pos": lane_pos,
            "lane_len": lane_len,
            "d_stop": d_stop,
            "speed_est": spd_est,
            "stopped": stopped,
        })
    return feats


class LaneArrivalCounter:
    def __init__(self, window_s: float = 60.0):
        self.window_s = float(window_s)
        self.events = defaultdict(deque)  # lane_id -> deque[timestamps]
        self.last_lane = {}

    def update(self, t_s: float):
        # track lane entry events per vehicle
        for vid in traci.vehicle.getIDList():
            lane = traci.vehicle.getLaneID(vid)
            prev = self.last_lane.get(vid)
            if lane and lane != prev:
                self.events[lane].append(float(t_s))
            self.last_lane[vid] = lane
        # garbage collect old events
        cutoff = float(t_s) - self.window_s
        for lane, dq in self.events.items():
            while dq and dq[0] < cutoff:
                dq.popleft()

    def rate(self, lane_id: str, window_s: float | None = None) -> float:
        w = float(window_s) if window_s is not None else self.window_s
        if w <= 0.0:
            return 0.0
        return float(len(self.events.get(lane_id, ())) / w)


