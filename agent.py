class RLAgent:
    def __init__(self):
        self.last_action = None

    def select_action(self, features):
        base = 15
        extension = 0
        extension += min(25, int(features.get("queue_len_total", 0) * 0.5))
        extension += min(10, int(features.get("risk_score", 0) * 10))
        extension -= int(max(0.0, features.get("avg_speed", 0) - 5) * 0.5)
        action = max(10, min(60, base + extension))
        self.last_action = action
        return action


