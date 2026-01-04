from pathlib import Path
import csv
from typing import Dict


class ReplayPredictionWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._ensure_file()

    def _ensure_file(self):
        if not self.output_path.exists():
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "prediction",
                    "confidence_level",
                    "p_down",
                    "p_sideways",
                    "p_up",
                ])

    def write(self, state: Dict):
        probs = state.get("probabilities", {})

        with open(self.output_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                state.get("timestamp"),
                state.get("prediction"),
                state.get("confidence_level"),
                probs.get("DOWN"),
                probs.get("SIDEWAYS"),
                probs.get("UP"),
            ])
