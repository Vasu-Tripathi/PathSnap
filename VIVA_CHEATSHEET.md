# PathSnap viva cheat sheet

Know these answers cold. All numbers come from `results/`.

**1. What is the task?**
Noisy 1 Hz GPS → the road arcs actually driven. Each trip has its own road network: `.nodes` (lon, lat) and `.arcs` (from → to, which are *directed*; the row index is the arc_id). `.route` is the ordered list of arc ids driven.

**2. Preprocessing**
- Trips are spread around the world, so each one is projected to a **local Azimuthal Equidistant projection** centred on its GPS centroid, giving metres. This avoids the wrong UTM zone and the distortion of using raw degrees.
- Sort by time and de-duplicate. The data had no NaNs and no duplicate timestamps.
- **Labels:** `.route` gives arcs, not one arc per GPS point. We align points to route arcs with a **monotone dynamic program**: the k-th route arc assigned can only stay the same or increase over time, and total point-to-segment distance is minimised. Median distance is 5.6 m.
- Anomalies: trips 33 and 54 have a constant ~120–140 m GPS offset, so they're excluded from training.
- For training only, every 3rd point is kept (consecutive 1 Hz points are near-duplicates).

**3. Candidates and features**
- Candidates: arcs within 50 m (max 8). If there are none, the 3 nearest within 300 m. Lookup uses a KD-tree over arc samples spaced 10 m apart.
- Miss rate (the true arc isn't among the candidates): 0.41% on student_test.
- Features (17):
  - distance and relative distance to the closest candidate
  - **heading vs arc bearing** (±2 and ±5 points)
  - mean/max distance of the ±3 neighbouring points to the same arc
  - projection position t, arc length, speed, jump size
  - number of candidates, whether the road is two-way, node degrees
- Why heading matters: two-way roads are two opposite arcs at identical distance. Nearest-arc alone gets only 53% Top-1.

**4. Models**
- Binary classification of (point, candidate) pairs. 7 models: LogReg, GaussianNB, KNN, RandomForest, LightGBM, XGBoost, MLP.
- CV is 5-fold **GroupKFold by trip**, so no trip leaks between folds. Student_test is used for comparison only.
- Primary metric: **observation Top-1**, i.e. is the highest-scoring candidate the reference arc? That's what the route is judged on, and it needs no threshold. Macro F1 uses a 0.5 threshold. Log loss matters because the probabilities become HMM emissions.
- **Why LightGBM:** best CV Macro F1 (0.938), CV Top-1, and log loss (calibration). It trains about 16× faster than RF and runs inference in about 40 ms per 1k points. XGBoost is basically tied with it.
- Fit diagnosis:
  - LogReg and NB underfit (train ≈ CV ≈ test, all low). NB is badly calibrated because it assumes correlated distance features are independent.
  - KNN has train F1 = 1.0, so it memorises, and it's very slow.
  - XGBoost's validation loss bottoms out at round 247.
  - The MLP's train/val gap grows after epoch 23.

**5. Bayesian optimisation**
- Optuna TPE, 30 trials. Trial 0 was the untuned config, followed by 8 random startup trials.
- Objective: 3-fold grouped CV Top-1.
- Search space: n_estimators, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, reg_lambda and reg_alpha. Learning rate, leaves, child samples and both regularisers use log scales.
- Result: Top-1 improved by +0.17 points on test. F1 was flat and log loss slightly worse. After the HMM, tuned and untuned are equal. **Honest takeaway:** the features are the limit, not the hyperparameters.
- TPE models the good and bad trial regions as two densities and picks the next trial where the good/bad ratio is high. It's smarter than grid search.

**6. Leakage**
- Trip-grouped splits.
- Scalers sit inside the pipelines.
- The evaluator data is never read in development.
- Nothing is hard-coded.
