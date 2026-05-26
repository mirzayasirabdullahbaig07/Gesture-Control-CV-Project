# ── Gesture names ────────────────────────────────────────────
DRAW  = "draw"
MOVE  = "move"
ERASE = "erase"
CLEAR = "clear"

class GestureEngine:
    """Converts fingers_up list → stable gesture with cooldown."""
    def __init__(self, cooldown=18):
        self.cooldown = cooldown
        self._cd      = 0
        self.gesture  = MOVE

    def update(self, f):
        if not f or len(f)<5:
            return self.gesture
        raw = self._classify(f)
        if self._cd > 0:
            self._cd -= 1
            return self.gesture
        if raw != self.gesture:
            self.gesture = raw
            self._cd     = self.cooldown
        return self.gesture

    def force_cooldown(self, frames=40):
        self._cd = frames

    @staticmethod
    def _classify(f):
        n = sum(f)
        if n == 0:                                  return ERASE
        if n >= 5:                                  return CLEAR
        if f[1]==1 and f[2]==0 and f[3]==0 and f[4]==0: return DRAW
        if f[1]==1 and f[2]==1 and f[3]==0 and f[4]==0: return MOVE
        return MOVE