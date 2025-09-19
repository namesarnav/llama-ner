"""Metric helpers for token classification."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score


def build_compute_metrics(id2label: Dict[int, str]) -> Callable[[tuple], Dict[str, float]]:
    label_ids = list(id2label.keys())

    def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions: List[List[str]] = []
        true_labels: List[List[str]] = []

        for prediction, label in zip(predictions, labels):
            current_preds: List[str] = []
            current_labels: List[str] = []

            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                current_preds.append(id2label[int(pred_id)])
                current_labels.append(id2label[int(label_id)])

            if current_labels:
                true_predictions.append(current_preds)
                true_labels.append(current_labels)

        if not true_labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics
