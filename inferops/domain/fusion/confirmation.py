"""Media-time confirmation, isolated by entity and condition."""
from __future__ import annotations


class Confirmation:
    def __init__(self):
        self.pending = {}

    def details(self, key, timestamp, required_ms):
        first, _, samples = self.pending.get(key, (timestamp, timestamp, 1))
        return {"required_ms": required_ms, "observed_ms": timestamp - first, "samples": samples}

    def ready(self, key, matched, timestamp, *, confirm_ms=0, min_samples=1, max_gap_ms=1500):
        if not matched:
            self.pending.pop(key, None)
            return False
        if confirm_ms == 0 and min_samples == 1:
            return True
        previous = self.pending.get(key)
        if previous is None or timestamp < previous[1] or timestamp - previous[1] > max_gap_ms:
            first, last, samples = timestamp, timestamp, 1
        else:
            first, last, samples = previous
            if timestamp > last:
                samples += 1
                last = timestamp
        self.pending[key] = first, last, samples
        return timestamp - first >= confirm_ms and samples >= min_samples
