# Methodology

## Experimental design

Every synthetic run begins on a fixed UTC timeline. The generator combines daily and shorter-period
components with seeded random variation to create three correlated signals. Defect locations are
expressed as fractions of the run length, remain outside the initial calibration segment, and are
kept far enough apart for unambiguous event matching.

The generator writes an event table with these fields:

| Field | Meaning |
| --- | --- |
| `event_id` | Stable identifier within a generated run |
| `defect_type` | Expected detector category |
| `signal` | Affected signal or `__timestamp__` for timeline defects |
| `start`, `end` | Inclusive UTC event boundaries |
| `severity` | Qualitative injection strength |

The manifest records the scenario, seed, requested duration and frequency, row count, and signal
names. It contains no machine-specific path or hidden source data.

## Causality

A detector is causal when its decision at time `t` uses only observations at or before `t`.
The spike detector explicitly shifts its rolling baseline by one point, while the shift and noise
detectors use trailing windows ending at the current point. Persistence checks delay the alert until
the configured number of consecutive scores exists. Each finding stores this time as `detected_at`.

The exported `start` may precede `detected_at`: it describes the interval implicated by the evidence,
not a claim that the detector knew the full interval in advance.

The final pipeline consolidates overlapping diagnoses after each causal detector has run. A
persistent flatline, shift, or noise interval can explain instantaneous spike alerts on the same
signal. Nearby shift regions are treated as the onset and recovery of one excursion, with their
boundaries corrected by half of the comparison window. The original first alert time remains the
event's `detected_at` value.

## Robust statistics

For observations `x`, the median absolute deviation is:

```text
MAD(x) = median(|x - median(x)|)
```

The package multiplies MAD by `1.4826` when using it as a robust scale estimate. A trailing standard
deviation is used only when a Hampel window has zero MAD, with a small numerical floor for a fully
constant window.

The level-shift score compares medians in two adjacent trailing windows and normalizes their distance
by the robust scale in the clean calibration segment. The noise score applies the same scale concept
to first differences, separating rapid variation from gradual movement.

## Event matching

Truth and predicted events are grouped by defect type and signal. Candidate pairs must overlap or
fall within one expected cadence of each other. Candidates are ordered by temporal overlap and then
boundary distance. Greedy one-to-one assignment prevents a broad prediction from receiving credit
for several truth events.

For each defect type:

```text
event precision = matched predictions / all predictions
event recall    = matched truth events / all truth events
event F1        = harmonic mean of precision and recall
```

When both truth and predictions are empty, precision and recall are defined as one for that category.
An unexpected prediction in a clean category reduces precision to zero.

## Point evaluation

The observed range is expanded to a regular grid at the inferred cadence. Inclusive truth and
predicted intervals become Boolean masks on that grid. Their intersections produce TP, FP, FN, and
TN counts. This approach includes timestamps removed by an injected outage instead of silently
excluding them from evaluation.

Point metrics complement event metrics: a detector can locate every event yet cover too much clean
time, or miss the beginning of long failures while still receiving event-level credit.

## Detection delay

For a matched pair:

```text
delay = max(detected_at - truth_start, 0)
```

The benchmark reports the mean delay in seconds for each defect type. It does not mix delay from an
unmatched prediction into the average; those predictions are already represented as false positives.

## Limitations

The injected scenarios are controlled stress tests, not a statistical model of every physical
system. The quality score is a transparent prioritization aid rather than a probability. Event
matching uses a deterministic greedy assignment rather than an optimization algorithm, which is
appropriate for the deliberately separated benchmark events but may need refinement for dense,
overlapping labels.

## Chronological forecasting benchmark

The optional forecast command compares two dependency-light one-step-ahead baselines on the final
holdout segment:

- **Seasonal naive:** the observation from one configured season earlier.
- **Rolling mean:** the mean of a fixed number of observations immediately before the forecast time.

Predictions are calculated with `shift` before rolling or seasonal lookup, so neither baseline can
read the value it is predicting. The benchmark reports MAE, RMSE, and symmetric MAPE over timestamps
where both the actual value and prediction are present. This is a trend-analysis reference point,
not a claim that simple baselines are appropriate for every operating process.
