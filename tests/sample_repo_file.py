import os
import json
from pathlib import Path
from collections import defaultdict


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_risk_score(complexity: float, churn: int) -> float:
    return complexity * 0.6 + churn * 0.4


class RiskAnalyzer:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.history = defaultdict(list)

    def analyze(self, file_path: str) -> bool:
        score = compute_risk_score(2.5, 10)
        self.history[file_path].append(score)
        return score > self.threshold

    def report(self) -> dict:
        return dict(self.history)
