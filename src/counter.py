"""Line-crossing vehicle counter with per-category and per-direction stats."""
from collections import defaultdict


class LineCounter:
    def __init__(self, p1, p2):
        self.p1, self.p2 = p1, p2
        self.prev_side = {}                 # track_id -> sign of side
        self.counted = set()                # track ids already counted
        self.in_counts = defaultdict(int)   # class_name -> count
        self.out_counts = defaultdict(int)
        self.events = []                    # list of dicts for CSV

    def _side(self, pt):
        (x1, y1), (x2, y2) = self.p1, self.p2
        val = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
        return 1 if val > 0 else -1 if val < 0 else 0

    def _within_segment(self, pt):
        (x1, y1), (x2, y2) = self.p1, self.p2
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom == 0:
            return False
        t = ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / denom
        return 0.0 <= t <= 1.0

    def update(self, track_id, cls_name, centroid, frame_idx, fps):
        side = self._side(centroid)
        prev = self.prev_side.get(track_id)
        self.prev_side[track_id] = side
        if prev is None or side == 0 or prev == side:
            return None
        if track_id in self.counted or not self._within_segment(centroid):
            return None
        self.counted.add(track_id)
        direction = "in" if prev < 0 else "out"
        (self.in_counts if direction == "in" else self.out_counts)[cls_name] += 1
        self.events.append({
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 2),
            "track_id": track_id,
            "class": cls_name,
            "direction": direction,
        })
        return direction

    def totals(self):
        classes = set(self.in_counts) | set(self.out_counts)
        return {
            c: {"in": self.in_counts[c], "out": self.out_counts[c],
                "total": self.in_counts[c] + self.out_counts[c]}
            for c in sorted(classes)
        }
