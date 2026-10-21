"""Small linear Expected SARSA controller for historical agentic-RL replay."""

from __future__ import annotations

import math
import random

import numpy as np


ACTIONS = (
    "study:finish",
    "study:query",
    "fusion:p9",
    "fusion:fundamental",
    "fusion:research",
    "fusion:joint",
)
PROFILES = {
    "fusion:p9": "p9",
    "fusion:fundamental": "fundamental",
    "fusion:research": "research",
    "fusion:joint": "joint",
}
STATE_FEATURES = (
    "bias",
    "market_return_20s",
    "expert_disagreement",
    "fundamental_anomaly",
    "fundamental_coverage",
    "query_available",
    "evidence_conflict",
    "query_trade_importance",
    "account_cash_weight",
    "account_holdings_weight",
    "risk_to_budget",
    "days_since_query",
    "fusion_stage",
    "research_performed",
    "research_direction_score",
    "research_evidence_sufficiency",
    "research_conflict_resolved",
    "research_supporting_evidence_count",
    "research_contrary_evidence_count",
)


def state_vector(state: dict[str, float]) -> np.ndarray:
    """Encode observed decision context; all non-bias values are pre-scaled."""
    return np.asarray([1.0, *(float(state.get(name, 0.0)) for name in STATE_FEATURES[1:])], dtype=float)


class LinearExpectedSarsa:
    """Linear action values with an epsilon-greedy Expected SARSA update."""

    def __init__(self, alpha: float = 0.015, gamma: float = 0.96):
        self.alpha = alpha
        self.gamma = gamma
        self.weights = {action: np.zeros(len(STATE_FEATURES), dtype=float) for action in ACTIONS}
        self.updates = 0
        self.frozen = False

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LinearExpectedSarsa":
        """Restore a frozen policy artifact for inference."""
        if value.get("algorithm") != "linear_expected_sarsa_two_stage":
            raise ValueError("Unsupported Agentic RL policy algorithm")
        if tuple(value.get("state_features", ())) != STATE_FEATURES:
            raise ValueError("Agentic RL policy state features do not match this implementation")
        if tuple(value.get("actions", ())) != ACTIONS:
            raise ValueError("Agentic RL policy actions do not match this implementation")
        alpha = float(value["alpha"])
        gamma = float(value["gamma"])
        updates = int(value["updates"])
        if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Agentic RL policy has invalid learning parameters")
        if updates < 0:
            raise ValueError("Agentic RL policy has an invalid update count")
        raw_weights = value.get("weights_by_action")
        if not isinstance(raw_weights, dict) or set(raw_weights) != set(ACTIONS):
            raise ValueError("Agentic RL policy weights do not cover the declared actions")
        policy = cls(alpha=alpha, gamma=gamma)
        for action in ACTIONS:
            weights = np.asarray(raw_weights[action], dtype=float)
            if weights.shape != (len(STATE_FEATURES),) or not np.isfinite(weights).all():
                raise ValueError(f"Agentic RL policy has invalid weights for {action}")
            policy.weights[action] = weights
        policy.updates = updates
        policy.frozen = True
        return policy

    def value(self, action: str, features: np.ndarray) -> float:
        return float(self.weights[action] @ features)

    def expected_value(
        self,
        features: np.ndarray,
        allowed_actions: tuple[str, ...],
        epsilon: float,
    ) -> float:
        values = {action: self.value(action, features) for action in allowed_actions}
        best = max(values.values())
        greedy = [action for action in allowed_actions if values[action] == best]
        probability = epsilon / len(allowed_actions)
        return math.fsum(probability * value for value in values.values()) + (
            (1 - epsilon) / len(greedy) * math.fsum(values[action] for action in greedy)
        )

    def choose(
        self,
        features: np.ndarray,
        allowed_actions: tuple[str, ...],
        epsilon: float,
        seed: int,
    ) -> tuple[str, bool, dict[str, float]]:
        estimates = {action: self.value(action, features) for action in allowed_actions}
        randomizer = random.Random(seed)
        explored = randomizer.random() < epsilon
        if explored:
            action = randomizer.choice(allowed_actions)
        else:
            best = max(estimates.values())
            action = min(
                (action for action in allowed_actions if estimates[action] == best),
                key=ACTIONS.index,
            )
        return action, explored, estimates

    def update_n_step(
        self,
        action: str,
        features: np.ndarray,
        rewards: list[float],
        next_features: np.ndarray | None,
        next_actions: tuple[str, ...],
        epsilon: float,
    ) -> dict[str, float]:
        """Update from matured n-step rewards; the account NAV already includes fees."""
        if self.frozen:
            raise RuntimeError("A restored frozen Expected SARSA policy cannot be updated")
        discounted = math.fsum(
            (self.gamma ** index) * reward for index, reward in enumerate(rewards)
        )
        bootstrap = (
            self.gamma ** len(rewards)
            * self.expected_value(next_features, next_actions, epsilon)
            if next_features is not None and next_actions else 0.0
        )
        old_value = self.value(action, features)
        td_error = discounted + bootstrap - old_value
        self.weights[action] += self.alpha * td_error * features
        self.updates += 1
        return {
            "discounted_matured_reward": discounted,
            "expected_sarsa_bootstrap": bootstrap,
            "td_error": td_error,
            "updated_action_value": self.value(action, features),
        }

    def update_one_step(
        self,
        action: str,
        features: np.ndarray,
        reward: float,
        next_features: np.ndarray | None,
        next_actions: tuple[str, ...],
        epsilon: float,
        intraday_discount: float = 1.0,
    ) -> dict[str, float]:
        """Update a same-day research decision without discounting across market time."""
        if self.frozen:
            raise RuntimeError("A restored frozen Expected SARSA policy cannot be updated")
        discounted = float(reward)
        bootstrap = (
            intraday_discount * self.expected_value(next_features, next_actions, epsilon)
            if next_features is not None and next_actions else 0.0
        )
        old_value = self.value(action, features)
        td_error = discounted + bootstrap - old_value
        self.weights[action] += self.alpha * td_error * features
        self.updates += 1
        return {
            "discounted_matured_reward": discounted,
            "expected_sarsa_bootstrap": bootstrap,
            "intraday_discount": intraday_discount,
            "td_error": td_error,
            "updated_action_value": self.value(action, features),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": "linear_expected_sarsa_two_stage",
            "state_features": list(STATE_FEATURES),
            "actions": list(ACTIONS),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "study_step_update": "same-day one-step Expected SARSA with zero immediate reward, unit intraday discount, and fusion-stage bootstrap",
            "n_step_sessions": 3,
            "fusion_step_update": "three matured daily NAV/risk rewards; account NAV includes research fee once",
            "updates": self.updates,
            "weights_by_action": {
                action: [float(value) for value in self.weights[action]]
                for action in ACTIONS
            },
        }


class RidgeResidualModel:
    """Small standardized ridge model for sequential residual calibration."""

    def __init__(self, feature_names: tuple[str, ...], alpha: float = 10.0, minimum_rows: int = 30):
        self.feature_names = feature_names
        self.alpha = alpha
        self.minimum_rows = minimum_rows
        self.rows: list[tuple[list[float], float, str]] = []
        self.mean = np.zeros(len(feature_names), dtype=float)
        self.scale = np.ones(len(feature_names), dtype=float)
        self.coef = np.zeros(len(feature_names) + 1, dtype=float)
        self.rmse = None
        self.fitted_rows = 0
        self.training_rows_count = 0
        self.training_record_ids: list[str] = []
        self.frozen = False

    def add(self, feature_values: list[float], target: float, record_id: str) -> None:
        if self.frozen:
            raise RuntimeError("A restored frozen residual model cannot be trained")
        self.rows.append((feature_values, float(target), record_id))
        self.training_rows_count = len(self.rows)
        self.training_record_ids = [row[2] for row in self.rows]

    def fit(self) -> None:
        if self.frozen:
            raise RuntimeError("A restored frozen residual model cannot be trained")
        if len(self.rows) < self.minimum_rows:
            self.fitted_rows = len(self.rows)
            return
        values = np.asarray([row[0] for row in self.rows], dtype=float)
        targets = np.asarray([row[1] for row in self.rows], dtype=float)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        standardized = (values - self.mean) / self.scale
        design = np.column_stack((np.ones(len(values)), standardized))
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        residuals = design @ self.coef - targets
        self.rmse = float(np.sqrt(np.mean(residuals**2)))
        self.fitted_rows = len(self.rows)

    def predict(self, feature_values: list[float]) -> float:
        if self.fitted_rows < self.minimum_rows:
            return 0.0
        standardized = (np.asarray(feature_values, dtype=float) - self.mean) / self.scale
        return float(np.clip(self.coef[0] + self.coef[1:] @ standardized, -0.05, 0.05))

    def to_dict(self) -> dict[str, object]:
        return {
            "model": "standardized_ridge_residual",
            "feature_names": list(self.feature_names),
            "alpha": self.alpha,
            "minimum_rows": self.minimum_rows,
            "training_rows": self.training_rows_count,
            "fitted_rows": self.fitted_rows,
            "residual_rmse": self.rmse,
            "feature_mean": [float(value) for value in self.mean],
            "feature_scale": [float(value) for value in self.scale],
            "coefficients_with_intercept": [float(value) for value in self.coef],
            "training_record_ids": list(self.training_record_ids),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RidgeResidualModel":
        """Restore a frozen residual model artifact for inference."""
        if value.get("model") != "standardized_ridge_residual":
            raise ValueError("Unsupported Agentic RL residual model")
        feature_names = tuple(str(name) for name in value.get("feature_names", ()))
        if not feature_names:
            raise ValueError("Residual model artifact has no features")
        model = cls(
            feature_names,
            alpha=float(value["alpha"]),
            minimum_rows=int(value["minimum_rows"]),
        )
        if not math.isfinite(model.alpha) or model.alpha < 0 or model.minimum_rows < 1:
            raise ValueError("Residual model artifact has invalid fit parameters")
        mean = np.asarray(value["feature_mean"], dtype=float)
        scale = np.asarray(value["feature_scale"], dtype=float)
        coef = np.asarray(value["coefficients_with_intercept"], dtype=float)
        rows = int(value["training_rows"])
        fitted_rows = int(value["fitted_rows"])
        record_ids = [str(record_id) for record_id in value.get("training_record_ids", ())]
        if (
            mean.shape != (len(feature_names),)
            or scale.shape != (len(feature_names),)
            or coef.shape != (len(feature_names) + 1,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(coef).all()
            or np.any(scale <= 0)
            or rows < 0
            or fitted_rows < 0
            or fitted_rows > rows
            or (record_ids and len(record_ids) != rows)
        ):
            raise ValueError("Residual model artifact has invalid fitted arrays or row counts")
        rmse = value.get("residual_rmse")
        if rmse is not None and (not math.isfinite(float(rmse)) or float(rmse) < 0):
            raise ValueError("Residual model artifact has invalid RMSE")
        model.mean = mean
        model.scale = scale
        model.coef = coef
        model.training_rows_count = rows
        model.training_record_ids = record_ids
        model.fitted_rows = fitted_rows
        model.rmse = None if rmse is None else float(rmse)
        model.frozen = True
        return model
