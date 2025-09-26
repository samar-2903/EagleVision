import traci
import config as cfg
from agent import RLAgent
from forecast import ForecastModel

_rl_agent = RLAgent()
_forecast_model = ForecastModel()


def update_forecast(congestion_value):
    _forecast_model.update(congestion_value)


def decide_traffic_light(state, tl_id=None):
    if tl_id is None:
        try:
            tl_ids = list(traci.trafficlight.getIDList())
            if not tl_ids:
                print("No traffic lights found.")
                return
            tl_id = cfg.TLS_ID if cfg.TLS_ID else tl_ids[0]
        except Exception as e:
            print(f"Failed to fetch TLS IDs: {e}")
            return
    action_duration = _rl_agent.select_action({
        "queue_len_total": state.get("queue_len_total", 0),
        "avg_wait": state.get("avg_wait_total", 0.0),
        "avg_speed": state.get("avg_speed", 0.0),
        "risk_score": state.get("risk_score", 0.0)
    })

    if _forecast_model.predict_high_congestion(threshold=0.7):
        action_duration = min(60, int(action_duration * 1.2))

    traci.trafficlight.setPhaseDuration(tl_id, int(action_duration))
    print(
        f"[{tl_id}] phase_dur={int(action_duration)}s, q={state.get('queue_len_total',0)}, "
        f"risk={state.get('risk_score',0):.2f}, v={state.get('avg_speed',0):.1f}"
    )


