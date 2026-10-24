# RAAMA Risk Profile v1

## Purpose and limits

Risk profiles carry explicit portfolio constraints from the request form through the API queue, worker, decision, and persisted result. The canonical profile includes its schema and template versions, units, source evidence, unresolved text, and SHA-256 identity. The worker validates the profile and its hash before executing it; the profile identity is also part of task and recovery identity.

An annualized volatility budget is a portfolio construction limit. It is not a maximum drawdown or loss limit, and it does not guarantee a maximum loss. The parser does not convert drawdown or loss requests into volatility values. A profile is a user-selected constraint set, not a suitability assessment.

## Templates and source priority

The versioned templates are:

| Template | Annualized volatility budget | Maximum single-name weight | Maximum one-way turnover | Minimum cash |
| --- | ---: | ---: | ---: | ---: |
| Conservative | 12% | 5% | 10% | 25% |
| Balanced | 20% | 5% | 50% | 5% |
| Aggressive | 30% | 10% | 75% | 0% |

An explicit numeric field or supported numeric expression overrides a template value for that constraint. If explicit sources give different values for the same constraint, the profile needs clarification and cannot execute. Values from an unselected constraint use the chosen template; with no template, they use system defaults (20% annualized volatility, 5% single-name weight, 50% one-way turnover, and 0% minimum cash). The UI leaves an explicit annualized value blank unless the user supplies one.

## Supported language and fail-closed behavior

The offline parser recognizes the versioned conservative, balanced, and aggressive labels and positive numeric clauses for annualized risk/volatility upper bounds, single-name upper bounds, one-way turnover upper bounds, and minimum cash. Examples include “年化风险不超过0.15”, “单票不超过5%”, “换手率不超过30%”, “现金不低于20%”, “至少保留20%现金”, and “我偏保守”. Numeric percentages include `%` or `％`; bare numbers at or above 1 are rejected as ambiguous.

Negated, unsupported, contradictory, or reverse-direction clauses remain unresolved and block task submission. Examples include “不要保守”, “年化风险至少15%”, “不需要保留20%现金”, “单票不超过5%且不低于8%”, and “最多亏损10%”. The parser is deterministic and local; it does not call a model or a provider. New language patterns require parser and test changes together.

## Runtime enforcement

The optimizer checks each proposed target portfolio against the annualized volatility budget, single-name cap, turnover cap, and minimum post-cost cash floor. If a proposal is blocked, the hard-risk review resets the decision to a hold and checks the existing portfolio's estimated annualized risk, concentration, cash floor, and zero turnover; it reports any pre-existing violations. This review does not verify holdings after real fills. The manager can explain the profile and resulting decision, but does not change or authorize numeric limits. Warmup results retain the profile, its identity, source information, and the no-guarantee scope note.

The request preview and completed-task view display resolved values, their sources, and unresolved clauses. Unresolved profiles cannot create executable tasks. CLI, API, worker, and recovery paths validate the same canonical profile.

## Verification record

The focused risk-profile, optimizer, harness service, contextual collaboration, runtime, synthetic end-to-end, and decision-schema checks passed: 85 tests. The end-to-end test exercises the API-to-worker-to-CLI path using synthetic data, checks the final hold and persisted profile identity, and verifies recovery identity. The synthetic live review task `risk-profile-live-20261024` also completed on the isolated local service at `http://127.0.0.1:8876/`; provider execution was disabled, no broker was connected, and it generated no orders. No user records or experiment artifacts were used by this verification.
