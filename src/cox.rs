use std::fmt;
use std::sync::Arc;

use faer::linalg::matmul::matmul;
use faer::{Accum, ColMut, ColRef, Mat, MatMut, MatRef, Par};
use fearless_simd::{Level, Simd, dispatch};

use crate::inference::{Estimate, LrtResult, ProfileBound, ProfileOptions, Tail};
use crate::linalg::{
    CholeskySummary, MatrixScratch, cholesky_factor_logdet, cholesky_inverse_logdet,
    normalized_qr_pivots, pivoted_cholesky_inverse_logdet, quadratic_form_rows, rows_mul,
    weighted_gram,
};
use crate::simd::{axpy, dot, exp_shifted};
use crate::solver::{FitStatus, IterationQuantities, NewtonOptions, newton_raphson};
use crate::sum::CompensatedSum;
use crate::validation::{
    validate_feature_count, validate_finite_features, validate_nonnegative_weights,
    validate_optional_vector, validate_penalty_weight, validate_times, validate_vector,
};
use crate::{Error, MatrixView, Result};

/// Calls `$f::<_, K>(simd, ...)` for `K = $k` inside `dispatch!` when the feature count `$k` is at
/// most 16. Otherwise, the macro evaluates `$fallback`. The running sums in `$f` are arrays of
/// length `K` that the compiler keeps in registers.
macro_rules! with_const_k {
    ($k:expr, $f:ident($($arg:expr),* $(,)?), $fallback:expr $(,)?) => {
        with_const_k!(
            @arms ($k) $f ($($arg),*) ($fallback) 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        )
    };
    (@arms ($k:expr) $f:ident $args:tt ($fallback:expr) $($n:literal)*) => {
        match $k {
            $($n => dispatch!(Level::new(), simd => with_const_k!(@call $f $n simd $args)),)*
            _ => $fallback,
        }
    };
    (@call $f:ident $n:literal $simd:ident ($($arg:expr),*)) => {
        $f::<_, $n>($simd, $($arg),*)
    };
}

/// Options for fitting a Firth-penalized Cox model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoxOptions {
    /// Weight of the penalty term. The standard Firth weight is 0.5. Set to 0.0 for an unpenalized
    /// Cox partial likelihood fit.
    pub penalty_weight: f64,
    pub solver: NewtonOptions,
}

impl Default for CoxOptions {
    fn default() -> Self {
        Self {
            penalty_weight: 0.5,
            solver: NewtonOptions {
                max_iterations: 50,
                max_step: 5.0,
                max_half_steps: 5,
                gradient_tolerance: 1e-4,
                step_tolerance: 1e-6,
            },
        }
    }
}

/// Survival data for Cox regression, sorted and grouped once. Every model from
/// [`model`](Self::model) reuses the sorted data.
///
/// Clones share one copy of the data.
#[derive(Clone)]
pub struct CoxData {
    inner: Arc<CoxPrecomputed>,
}

impl CoxData {
    /// Checks the data, sorts it by descending time, and groups tied times.
    ///
    /// `time` is the event or censoring time of each sample. `event` is `true` where the event
    /// occurred and `false` where the sample was censored.
    ///
    /// # Errors
    ///
    /// Returns `InvalidDimensions` for fewer than two samples or mismatched lengths, or
    /// `InvalidInput` for a non-finite feature, a negative or non-finite time, or no events.
    pub fn new(features: MatrixView, time: &[f64], event: &[bool]) -> Result<Self> {
        validate_finite_features(features)?;
        validate_survival_data(features.nrows(), event, time)?;
        Ok(Self {
            inner: Arc::new(CoxPrecomputed::from_data(features, time, event)),
        })
    }

    /// Returns the Cox model of this data. The model has unit weights, zero offsets, and default
    /// options until [`sample_weight`](CoxModel::sample_weight), [`offset`](CoxModel::offset), or
    /// [`options`](CoxModel::options) changes them.
    pub fn model(&self) -> CoxModel<'_> {
        CoxModel {
            data: self,
            sample_weight: None,
            offset: None,
            options: CoxOptions::default(),
        }
    }
}

// A derived `Debug` implementation would print every row of the sorted data.
impl fmt::Debug for CoxData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoxData")
            .field("retained_rows", &self.inner.n_samples())
            .field("n_features", &self.inner.n_features())
            .finish_non_exhaustive()
    }
}

/// A Cox proportional hazards model before it is fitted. [`CoxData::model`] creates the model.
#[derive(Clone, Copy)]
#[must_use = "call `fit` to fit the model"]
pub struct CoxModel<'a> {
    data: &'a CoxData,
    sample_weight: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    options: CoxOptions,
}

impl<'a> CoxModel<'a> {
    /// `sample_weight[i]` is the nonnegative weight of input row `i`. Weights are counts. An
    /// integer weight `m` gives the same estimates, standard errors, likelihood ratio tests, and
    /// profile intervals as `m` copies of the row. A zero weight gives the same results as removing
    /// the row.
    pub fn sample_weight(mut self, sample_weight: &'a [f64]) -> Self {
        self.sample_weight = Some(sample_weight);
        self
    }

    /// `offset[i]` is added to the linear predictor of input row `i`.
    pub fn offset(mut self, offset: &'a [f64]) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn options(mut self, options: CoxOptions) -> Self {
        self.options = options;
        self
    }

    /// Fits the Cox proportional hazards model with Firth's bias reduction.
    ///
    /// # Errors
    ///
    /// Returns `InvalidDimensions` if the weights or offsets do not have one value per input row.
    /// Returns `InvalidInput` for an out-of-range option, a negative or non-finite weight, a
    /// non-finite offset, or no positive-weight event. Returns `RankDeficient` when some linear
    /// combination of the columns is constant over the positive-weight rows. For example, a full
    /// set of indicator columns sums to one on every row. The partial likelihood does not depend on
    /// such a combination. Returns a numerical error if a factorization fails at the starting
    /// coefficients or while the solver computes a Newton step.
    pub fn fit(self) -> Result<CoxFit> {
        let options = self.options;
        options.solver.validate()?;
        validate_penalty_weight(options.penalty_weight)?;

        let data = &*self.data.inner;
        let inputs = CaseInputs::new(data, self.sample_weight, self.offset)?;
        let mut objective = CoxObjective::new(data, &inputs, options.penalty_weight);
        let result = newton_raphson(options.solver, vec![0.0; data.n_features()], |beta| {
            objective.evaluate(beta)
        })?;
        let estimate = Estimate::new(result);

        // The solver's last evaluation is at the accepted beta unless step halving failed.
        let pass = &mut objective.workspace.pass;
        if pass.beta != estimate.beta {
            risk_set_pass(data, &inputs, &estimate.beta, pass);
        }
        let (unique_times, log_cumulative_baseline_hazard) = baseline_hazard(data, pass);
        let cumulative_baseline_hazard: Vec<f64> = log_cumulative_baseline_hazard
            .iter()
            .map(|&value| value.exp())
            .collect();
        let baseline_survival: Vec<f64> = cumulative_baseline_hazard
            .iter()
            .map(|&value| (-value).exp())
            .collect();

        Ok(CoxFit {
            estimate,
            unique_times,
            log_cumulative_baseline_hazard,
            cumulative_baseline_hazard,
            baseline_survival,
            data: self.data.clone(),
            inputs,
            options,
        })
    }
}

// A derived `Debug` implementation would print every row of the weights and offsets.
impl fmt::Debug for CoxModel<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoxModel")
            .field("data", self.data)
            .field("sample_weight_len", &self.sample_weight.map(<[f64]>::len))
            .field("offset_len", &self.offset.map(<[f64]>::len))
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

pub struct CoxFit {
    estimate: Estimate,
    unique_times: Vec<f64>,
    log_cumulative_baseline_hazard: Vec<f64>,
    cumulative_baseline_hazard: Vec<f64>,
    baseline_survival: Vec<f64>,
    // saved data for likelihood ratio tests and profile CIs
    data: CoxData,
    inputs: CaseInputs,
    options: CoxOptions,
}

// A derived `Debug` implementation would print the baseline hazard at every event time.
impl fmt::Debug for CoxFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoxFit")
            .field("coefficients", &self.estimate.beta)
            .field("standard_errors", &self.estimate.standard_errors)
            .field("log_likelihood", &self.estimate.log_likelihood)
            .field("status", &self.estimate.status)
            .field("iterations", &self.estimate.iterations)
            .field("data", &self.data)
            .finish_non_exhaustive()
    }
}

impl CoxFit {
    fn objective(&self) -> CoxObjective<'_> {
        CoxObjective::new(&self.data.inner, &self.inputs, self.options.penalty_weight)
    }

    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.estimate.beta
    }

    #[must_use]
    pub fn log_likelihood(&self) -> f64 {
        self.estimate.log_likelihood
    }

    #[must_use]
    pub fn standard_errors(&self) -> &[f64] {
        &self.estimate.standard_errors
    }

    /// Returns the inverse Fisher information at the penalized estimate in row-major order.
    ///
    /// Rows and columns are in the order of [`coefficients`](Self::coefficients). The result is
    /// `None` if the fit did not converge or if the information is singular.
    #[must_use]
    pub fn covariance(&self) -> Option<&[f64]> {
        self.estimate.covariance.as_deref()
    }

    #[must_use]
    pub fn p_values(&self) -> &[f64] {
        &self.estimate.p_values
    }

    /// Base-10 logs of the Wald p-values. The logs are finite even where `p_values` underflow to
    /// zero.
    #[must_use]
    pub fn log10_p_values(&self) -> Vec<f64> {
        self.estimate.log10_p_values()
    }

    #[must_use]
    pub fn iterations(&self) -> usize {
        self.estimate.iterations
    }

    #[must_use]
    pub fn status(&self) -> FitStatus {
        self.estimate.status
    }

    /// Unique event times in ascending order.
    #[must_use]
    pub fn unique_times(&self) -> &[f64] {
        &self.unique_times
    }

    /// Breslow cumulative baseline hazard at each unique event time. The baseline is a sample whose
    /// features and offset are all zero.
    #[must_use]
    pub fn cumulative_baseline_hazard(&self) -> &[f64] {
        &self.cumulative_baseline_hazard
    }

    /// Baseline survival function `exp(-cumulative_baseline_hazard)` at each unique event time.
    #[must_use]
    pub fn baseline_survival(&self) -> &[f64] {
        &self.baseline_survival
    }

    /// Computes `X @ coefficients + offset` (log relative hazard) for each row.
    ///
    /// # Errors
    ///
    /// Returns `InvalidDimensions` for mismatched shapes and `InvalidInput` for non-finite values.
    pub fn linear_predictor(
        &self,
        features: MatrixView,
        offset: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let nrows = features.nrows();
        let n_features = self.estimate.beta.len();
        validate_feature_count(features, n_features)?;
        validate_finite_features(features)?;
        let offset = validate_optional_vector(offset, nrows, 0.0, "offset")?;

        Ok((0..nrows)
            .map(|i| {
                (0..n_features)
                    .map(|j| features.at(i, j) * self.estimate.beta[j])
                    .sum::<f64>()
                    + offset[i]
            })
            .collect())
    }

    /// Computes Harrell's concordance index (C-index). [`Self::linear_predictor`] gives the risk
    /// scores.
    ///
    /// # Errors
    ///
    /// Returns `InvalidDimensions` for mismatched shapes and `InvalidInput` for non-finite values
    /// or negative times.
    pub fn concordance(
        &self,
        features: MatrixView,
        time: &[f64],
        event: &[bool],
        offset: Option<&[f64]>,
    ) -> Result<f64> {
        let risk = self.linear_predictor(features, offset)?;
        concordance_index(time, event, &risk)
    }

    /// Predicts cumulative hazard curves at [`Self::unique_times`].
    ///
    /// Returns one row-major buffer with a row per sample and a column per unique time. The value
    /// for row `i` at time `j` is at index `i * unique_times().len() + j`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::linear_predictor`].
    pub fn predict_cumulative_hazard(
        &self,
        features: MatrixView,
        offset: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let eta = self.linear_predictor(features, offset)?;
        let times = self.log_cumulative_baseline_hazard.len();
        let mut hazard = Vec::with_capacity(eta.len() * times);
        for eta in eta {
            hazard.extend(
                self.log_cumulative_baseline_hazard
                    .iter()
                    .map(|&log_baseline| (log_baseline + eta).exp()),
            );
        }
        Ok(hazard)
    }

    /// Predicts survival curves at [`Self::unique_times`]. The buffer has the same layout as
    /// [`Self::predict_cumulative_hazard`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::linear_predictor`].
    pub fn predict_survival(
        &self,
        features: MatrixView,
        offset: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let mut survival = self.predict_cumulative_hazard(features, offset)?;
        for value in &mut survival {
            *value = (-*value).exp();
        }
        Ok(survival)
    }

    /// Wald confidence intervals for all coefficients.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `alpha` is outside the open interval (0, 1).
    pub fn wald_confidence_intervals(&self, alpha: f64) -> Result<Vec<[f64; 2]>> {
        self.estimate.wald_confidence_intervals(alpha)
    }

    /// Penalized likelihood ratio test that the coefficients at `indices` are all zero. The test
    /// has `indices.len()` degrees of freedom.
    ///
    /// The test refits the model with the coefficients at `indices` fixed to zero.
    ///
    /// When `warm_start` is false, the other coefficients start at zero.
    /// When `warm_start` is true, the other coefficients start at their conditional mean given that the
    /// fixed coefficients are zero. Let `C` be the inverse Fisher information of the full fit,
    /// returned by [`covariance`](Self::covariance). The conditional mean is computed as if the
    /// full estimate `beta` were normal with covariance `C`. Let `S` be the positions in `indices`.
    /// Let `F` be the other positions. The start is then `beta_F - C_FS C_SS^-1 beta_S`. With one
    /// tested coefficient `j`, each free coefficient `f` starts at `beta_f - beta_j C_fj / C_jj`.
    ///
    /// Equivalently, the start is the result of one Newton step from the full estimate with the
    /// coefficients at `indices` set to zero. The step changes only the other coefficients. It uses
    /// a quadratic approximation of the penalized log-likelihood around the full estimate.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `indices` is empty, repeats an index, or contains an index past
    /// the last parameter. This method does not return an error when the full or constrained fit
    /// does not converge. The returned `LrtResult` then has status `FullFitNonconvergence` or
    /// `ConstrainedFitNonconvergence`.
    pub fn likelihood_ratio_test(&self, indices: &[usize], warm_start: bool) -> Result<LrtResult> {
        let mut objective = self.objective();
        self.estimate
            .likelihood_ratio_test(indices, warm_start, self.options.solver, |beta| {
                objective.evaluate(beta)
            })
    }

    /// Computes one bound of a profile-likelihood confidence interval with the Venzon-Moolgavkar
    /// (1988) algorithm.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `index`, `alpha`, or a field of `options` is out of range. Returns
    /// a numerical error if a solve fails during root finding. When the fit or the profile
    /// iteration does not converge, this method returns a `ProfileBound` with status
    /// `FullFitNonconvergence` or `ProfileBoundNonconvergence`.
    pub fn profile_confidence_bound(
        &self,
        index: usize,
        tail: Tail,
        alpha: f64,
        options: ProfileOptions,
    ) -> Result<ProfileBound> {
        let mut objective = self.objective();
        self.estimate
            .profile_bound(index, tail, alpha, options, |beta| objective.evaluate(beta))
    }

    /// Returns both profile-likelihood confidence bounds for a parameter as `[lower, upper]`.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`profile_confidence_bound`](Self::profile_confidence_bound).
    pub fn profile_confidence_interval(
        &self,
        index: usize,
        alpha: f64,
        options: ProfileOptions,
    ) -> Result<[ProfileBound; 2]> {
        let mut objective = self.objective();
        self.estimate
            .profile_interval(index, alpha, options, |beta| objective.evaluate(beta))
    }
}

/// Computes Harrell's concordance index (C-index).
///
/// # Errors
///
/// Returns `InvalidDimensions` if the slices have different lengths, or `InvalidInput`
/// for a negative or non-finite time or a non-finite risk score.
#[expect(clippy::float_cmp)]
pub fn concordance_index(time: &[f64], event: &[bool], risk: &[f64]) -> Result<f64> {
    let n = time.len();
    if event.len() != n || risk.len() != n {
        return Err(Error::InvalidDimensions(format!(
            "time, event, and risk must have equal lengths, got {}, {}, and {}",
            time.len(),
            event.len(),
            risk.len()
        )));
    }
    validate_times(time)?;
    if risk.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidInput("risk must be finite".into()));
    }

    // Rank the risks so that equal risks have equal ranks. -0.0 and 0.0 are ties.
    let mut distinct_risks = risk.to_vec();
    distinct_risks.sort_by(f64::total_cmp);
    distinct_risks.dedup();
    let rank: Vec<usize> = risk
        .iter()
        .map(|&value| distinct_risks.partition_point(|&other| other < value))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| time[b].total_cmp(&time[a]));

    // Go through the times from latest to earliest so that `seen` holds the ranks of all rows with
    // later times. Each event forms a pair with every row in `seen`. The pair is concordant if the
    // event has the higher risk, a tie if the risks are equal, and discordant otherwise.
    let mut seen = RankCounts::new(distinct_risks.len());
    let (mut concordant, mut discordant, mut tied) = (0_usize, 0_usize, 0_usize);
    for block in order.chunk_by(|&a, &b| time[a] == time[b]) {
        // Add the censored rows at this time first so that the events at this time pair with them.
        // Events at the same time do not pair with each other.
        for &i in block.iter().filter(|&&i| !event[i]) {
            seen.insert(rank[i]);
        }
        for &i in block.iter().filter(|&&i| event[i]) {
            let below = seen.below(rank[i]);
            let at = seen.counts[rank[i]];
            concordant += below;
            tied += at;
            discordant += seen.total - below - at;
        }
        for &i in block.iter().filter(|&&i| event[i]) {
            seen.insert(rank[i]);
        }
    }

    let total = concordant + discordant + tied;
    if total == 0 {
        return Ok(0.5);
    }
    Ok((concordant as f64 + 0.5 * tied as f64) / total as f64)
}

/// Counts of inserted risk ranks. A Fenwick tree gives the count below a rank in O(log n).
struct RankCounts {
    tree: Vec<usize>,
    counts: Vec<usize>,
    total: usize,
}

impl RankCounts {
    fn new(n_ranks: usize) -> Self {
        Self {
            tree: vec![0; n_ranks + 1],
            counts: vec![0; n_ranks],
            total: 0,
        }
    }

    fn insert(&mut self, rank: usize) {
        self.counts[rank] += 1;
        self.total += 1;
        let mut i = rank + 1;
        while i < self.tree.len() {
            self.tree[i] += 1;
            // `i & i.wrapping_neg()` is the lowest set bit of `i`.
            i += i & i.wrapping_neg();
        }
    }

    /// Number of inserted ranks below `rank`.
    fn below(&self, rank: usize) -> usize {
        let mut sum = 0;
        let mut i = rank;
        while i > 0 {
            sum += self.tree[i];
            i -= i & i.wrapping_neg();
        }
        sum
    }
}

/// Computes the Breslow cumulative baseline hazard from a risk-set pass at the fitted coefficients.
///
/// Returns the unique event times in ascending order and the log cumulative
/// baseline hazard at each of those times.
fn baseline_hazard(data: &CoxPrecomputed, pass: &RiskSetPass) -> (Vec<f64>, Vec<f64>) {
    let center_linear_predictor: f64 = pass
        .beta
        .iter()
        .zip(&data.feature_centers)
        .map(|(coefficient, center)| coefficient * center)
        .sum();

    // The blocks are in descending time order.
    let mut unique_times: Vec<f64> = pass
        .blocks
        .iter()
        .map(|block| data.time[block.end - 1])
        .collect();
    let mut log_hazard: Vec<f64> = pass
        .blocks
        .iter()
        .map(|block| block.event_weight.ln() - block.log_s0 - center_linear_predictor)
        .collect();
    unique_times.reverse();
    log_hazard.reverse();

    // The log cumulative hazard is the prefix log-sum-exp of the log hazard increments. `sum` is
    // the prefix sum of `exp(increment - max)`, where `max` is the running maximum of the
    // increments. The loop rescales `sum` when `max` grows. No `exp` or `ln` call depends on the
    // result of the previous `ln` call. The processor can therefore overlap consecutive steps. An
    // increment of negative infinity adds nothing to `sum`.
    let mut log_cumulative_hazard = log_hazard;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for value in &mut log_cumulative_hazard {
        if *value > max {
            sum = sum * (max - *value).exp() + 1.0;
            max = *value;
        } else if *value > f64::NEG_INFINITY {
            sum += (*value - max).exp();
        }
        *value = max + sum.ln();
    }

    (unique_times, log_cumulative_hazard)
}

/// Precomputed data for Cox likelihood evaluation.
struct CoxPrecomputed {
    /// Centered feature matrix sorted by time descending, row-major `(n_samples, n_features)`.
    x: Vec<f64>,
    /// Center (median) subtracted from each feature column.
    feature_centers: Vec<f64>,
    n_features: usize,
    /// Observed times, sorted descending.
    time: Vec<f64>,
    /// Event indicators in the order of `time`.
    event: Vec<bool>,
    /// Input row of each sorted sample.
    order: Vec<usize>,
    /// Number of input rows, including the rows censored before the earliest event.
    n_input_rows: usize,
    /// End index (exclusive) of each time block in the sorted arrays.
    block_ends: Vec<usize>,
}

const MAX_CENTER_SAMPLES: usize = 1024;

/// Returns the median of each column of `features` over at most `MAX_CENTER_SAMPLES` of the rows in
/// `rows`. The sampled rows are evenly spaced in `rows`.
fn sampled_column_centers(features: MatrixView, rows: &[usize]) -> Vec<f64> {
    let active_rows = rows.len();
    let n_features = features.ncols();
    let sample_count = active_rows.min(MAX_CENTER_SAMPLES);
    let mut sample = vec![0.0; sample_count];
    let sample_intervals = sample_count.saturating_sub(1).max(1);
    let last_row = active_rows - 1;

    (0..n_features)
        .map(|column| {
            for (sample_index, value) in sample.iter_mut().enumerate() {
                let row = sample_index * last_row / sample_intervals;
                *value = features.at(rows[row], column);
            }
            let (_, median, _) = sample.select_nth_unstable_by(sample_count / 2, f64::total_cmp);
            *median
        })
        .collect()
}

impl CoxPrecomputed {
    /// Sorts the samples by descending time and groups the samples with exactly equal times into
    /// blocks.
    ///
    /// Times that differ only by rounding error, such as `1.0` and `1.0 + 1e-15`, are in
    /// different blocks unless the user rounds them before fitting.
    ///
    /// The inputs must pass `validate_survival_data`.
    #[expect(clippy::float_cmp)]
    // Keep setup out of the hot fit path; benchmark before changing this boundary.
    #[inline(never)]
    fn from_data(x: MatrixView, time: &[f64], event: &[bool]) -> Self {
        let n = x.nrows();
        let k = x.ncols();

        // Tied samples stay in input order because `sort_by` is stable.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .expect("validated times are finite and comparable")
        });
        let mut time_sorted: Vec<f64> = order.iter().map(|&i| time[i]).collect();
        let mut event_sorted: Vec<bool> = order.iter().map(|&i| event[i]).collect();

        let mut block_ends = Vec::new();
        for i in 1..n {
            if time_sorted[i] != time_sorted[i - 1] {
                block_ends.push(i);
            }
        }
        block_ends.push(n);

        let last_event_row = event_sorted
            .iter()
            .rposition(|&value| value)
            .expect("validated data contains an event");
        let risk_set_rows = block_ends
            .iter()
            .copied()
            .find(|&end| end > last_event_row)
            .expect("the final block contains the final event");

        // Samples censored before the earliest event time are in no risk set of an event. Samples
        // censored at the earliest event time are in the risk set of that event.
        let n = risk_set_rows;
        order.truncate(n);
        time_sorted.truncate(n);
        event_sorted.truncate(n);
        block_ends.retain(|&end| end <= n);

        let feature_centers = sampled_column_centers(x, &order);
        let mut x_sorted = vec![0.0; n * k];
        for (row, &source) in x_sorted.chunks_exact_mut(k).zip(&order) {
            if let Some(input) = x.contiguous_row(source) {
                for ((value, &feature), &center) in row.iter_mut().zip(input).zip(&feature_centers)
                {
                    *value = feature - center;
                }
            } else {
                for (column, (value, &center)) in row.iter_mut().zip(&feature_centers).enumerate() {
                    *value = x.at(source, column) - center;
                }
            }
        }

        Self {
            x: x_sorted,
            feature_centers,
            n_features: k,
            time: time_sorted,
            event: event_sorted,
            order,
            n_input_rows: x.nrows(),
            block_ends,
        }
    }

    fn n_samples(&self) -> usize {
        self.time.len()
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    /// Feature row of the `i`-th sample in descending time order.
    fn row(&self, i: usize) -> &[f64] {
        &self.x[i * self.n_features..(i + 1) * self.n_features]
    }
}

/// Weights, offsets, and weighted event sums for one fit.
struct CaseInputs {
    /// `offset + ln(weight)` for each sorted sample, or `None` when the fit has neither weights nor
    /// offsets. A zero weight gives negative infinity.
    eta_shift: Option<Vec<f64>>,
    /// Summed weight of the events in each time block.
    event_weights: Vec<f64>,
    /// Weighted sum of the centered features over all events.
    event_feature_totals: Vec<f64>,
    /// Weighted sum of the offsets over all events.
    event_offset_total: f64,
    /// Number of time blocks with positive event weight.
    n_event_blocks: usize,
}

impl CaseInputs {
    /// Checks the weights and offsets of the input rows and arranges them in the order of `data`.
    fn new(
        data: &CoxPrecomputed,
        sample_weight: Option<&[f64]>,
        offset: Option<&[f64]>,
    ) -> Result<Self> {
        if let Some(sample_weight) = sample_weight {
            validate_vector(sample_weight, data.n_input_rows, "sample_weight")?;
            validate_nonnegative_weights(sample_weight)?;
        }
        if let Some(offset) = offset {
            validate_vector(offset, data.n_input_rows, "offset")?;
        }
        // Constant closures let the compiler drop the row lookups and multiplications from the
        // unweighted sums.
        let inputs = if sample_weight.is_none() && offset.is_none() {
            Self::with_event_sums(data, None, |_| 1.0, |_| 0.0)
        } else {
            let weight_at = |i: usize| sample_weight.map_or(1.0, |weight| weight[data.order[i]]);
            let offset_at = |i: usize| offset.map_or(0.0, |offset| offset[data.order[i]]);
            let eta_shift = (0..data.n_samples())
                .map(|i| offset_at(i) + weight_at(i).ln())
                .collect();
            Self::with_event_sums(data, Some(eta_shift), weight_at, offset_at)
        };
        if inputs.n_event_blocks == 0 {
            return Err(Error::InvalidInput(
                "at least one positive-weight event is required".into(),
            ));
        }
        Ok(inputs)
    }

    /// Computes the weighted event sums of `CaseInputs` from the weight and offset of each sorted
    /// sample.
    fn with_event_sums(
        data: &CoxPrecomputed,
        eta_shift: Option<Vec<f64>>,
        weight: impl Fn(usize) -> f64,
        offset: impl Fn(usize) -> f64,
    ) -> Self {
        let mut event_weights = vec![0.0; data.block_ends.len()];
        let mut event_feature_totals = vec![0.0; data.n_features];
        let mut event_offset_total = 0.0;
        let mut start = 0;
        for (event_weight, &end) in event_weights.iter_mut().zip(&data.block_ends) {
            for i in start..end {
                if data.event[i] {
                    let weight = weight(i);
                    *event_weight += weight;
                    event_offset_total += weight * offset(i);
                    for (total, &x) in event_feature_totals.iter_mut().zip(data.row(i)) {
                        *total += weight * x;
                    }
                }
            }
            start = end;
        }
        let n_event_blocks = event_weights.iter().filter(|&&weight| weight > 0.0).count();
        Self {
            eta_shift,
            event_weights,
            event_feature_totals,
            event_offset_total,
            n_event_blocks,
        }
    }
}

/// Computes the penalized log-likelihood, modified score, and Fisher information at `beta` for one
/// iteration of Newton-Raphson.
///
/// The Fisher information is `X.T @ diag(weights) @ X - x_bar.T @ diag(event_weights) @ x_bar`.
fn cox_quantities(
    data: &CoxPrecomputed,
    inputs: &CaseInputs,
    penalty_weight: f64,
    beta: &[f64],
    workspace: &mut CoxWorkspace,
) -> Result<IterationQuantities> {
    let n = data.n_samples();
    let k = data.n_features();
    let design = MatRef::from_row_major_slice(&data.x, n, k);

    let CoxWorkspace {
        matrix_scratch,
        pass,
        normalized_risk,
        weights,
        cumulative_weight,
        event_weights,
        ..
    } = &mut *workspace;
    risk_set_pass(data, inputs, beta, pass);
    let n_event_blocks = pass.blocks.len();
    // Samples past the end of the last event block are in no risk set of a positive-weight event.
    let risk_set_rows = pass.blocks.last().map_or(0, |block| block.end);
    let x_bar = MatRef::from_row_major_slice(&pass.x_bar, n_event_blocks, k);
    event_weights.clear();
    event_weights.extend(pass.blocks.iter().map(|block| block.event_weight));

    // `cumulative_weight[e]` is the sum of `event_weights[f] * S0[e] / S0[f]` over the blocks
    // `f >= e`. `S0[e]` is the risk-set sum of block `e`.
    cumulative_weight.resize(n_event_blocks, 0.0);
    for e in (0..n_event_blocks).rev() {
        let block = &pass.blocks[e];
        cumulative_weight[e] = block.event_weight
            + if e + 1 < n_event_blocks {
                pass.carry(e) * cumulative_weight[e + 1]
            } else {
                0.0
            };
    }

    // Divide the risk of each sample by the risk-set sum of the first event block whose risk set
    // contains the sample.
    *normalized_risk = vec![0.0; n];
    *weights = vec![0.0; n];
    let normalized_risk = normalized_risk.as_mut_slice();
    let weights = weights.as_mut_slice();
    let running_max_eta = pass.running_max_eta.as_slice();
    let pass_scaled_risk = pass.scaled_risk.as_slice();
    let mut start = 0;
    for (e, block) in pass.blocks.iter().enumerate() {
        let block_max = running_max_eta[block.end - 1];
        let inverse_s0 = 1.0 / block.s0;
        // Convert a risk from the scale of `max` to the scale of `block_max`.
        let factor_at = |max: f64| {
            if max < block_max {
                (max - block_max).exp() * inverse_s0
            } else {
                inverse_s0
            }
        };
        let mut factor_max = running_max_eta[start];
        let mut factor = factor_at(factor_max);
        for i in start..block.end {
            if running_max_eta[i] > factor_max {
                factor_max = running_max_eta[i];
                factor = factor_at(factor_max);
            }
            normalized_risk[i] = pass_scaled_risk[i] * factor;
            weights[i] = normalized_risk[i] * cumulative_weight[e];
        }
        start = block.end;
    }

    let mut information = weighted_gram(design, weights, matrix_scratch);
    let gram_diagonal: Vec<f64> = (0..k).map(|j| information[(j, j)]).collect();
    let mean_gram = weighted_gram(x_bar, event_weights, matrix_scratch);
    for r in 0..k {
        for s in 0..k {
            information[(r, s)] -= mean_gram[(r, s)];
        }
    }

    let mut quantities = IterationQuantities {
        log_likelihood: pass.log_likelihood,
        modified_score: pass.score.clone(),
        step_matrix: information,
        information: None,
        gram_diagonal: Some(gram_diagonal),
    };

    // A rank-deficient design makes the information singular at every beta. Because a full-rank
    // design can also make the information numerically singular at extreme beta, only the first
    // evaluation of a workspace checks the rank.
    if !workspace.rank_checked {
        check_design_rank(data, inputs, risk_set_rows, &quantities.step_matrix)?;
        workspace.rank_checked = true;
    }
    if penalty_weight == 0.0 {
        return Ok(quantities);
    }

    let cholesky = factor_information(&quantities.step_matrix)?;
    firth_correction(
        data,
        &cholesky.inverse,
        penalty_weight,
        &mut quantities.modified_score,
        workspace,
    );
    quantities.log_likelihood += penalty_weight * cholesky.log_determinant;

    Ok(quantities)
}

/// Returns `RankDeficient` when some combination of the columns, such as the sum of a full set of
/// indicator columns, is constant over the positive-weight samples among the first `risk_set_rows`
/// samples. Those samples form the largest risk set of a positive-weight event. The information is
/// then singular. `cox_quantities` forms the information as a difference of two Gram matrices.
/// Rounding in that difference can put the smallest pivot above the cutoff of `factor_information`.
/// The QR factorization of the design with a column of ones runs only when the Cholesky
/// factorization of the information fails or when its `minimum_relative_pivot` is at most
/// `sqrt(max(n, k + 1)·eps)`.
fn check_design_rank(
    data: &CoxPrecomputed,
    inputs: &CaseInputs,
    risk_set_rows: usize,
    information: &Mat<f64>,
) -> Result<()> {
    let k = data.n_features();
    let tolerance = (data.n_samples().max(k + 1) as f64 * f64::EPSILON).sqrt();
    match cholesky_factor_logdet(information) {
        Ok(cholesky) if cholesky.minimum_relative_pivot > tolerance => return Ok(()),
        Ok(_) | Err(Error::NotPositiveDefinite) => {},
        Err(error) => return Err(error),
    }
    // Zero-weight samples have an `eta_shift` of negative infinity.
    let rows: Vec<usize> = (0..risk_set_rows)
        .filter(|&i| {
            inputs
                .eta_shift
                .as_ref()
                .is_none_or(|shift| shift[i] > f64::NEG_INFINITY)
        })
        .collect();
    let pivots = normalized_qr_pivots(rows.len(), k + 1, |i, j| {
        if j == k { 1.0 } else { data.row(rows[i])[j] }
    });
    let cutoff = rows.len().max(k + 1) as f64 * f64::EPSILON * pivots[0];
    let rank = pivots.iter().filter(|&&pivot| pivot > cutoff).count();
    if rank < k + 1 {
        return Err(Error::RankDeficient {
            rank: rank.saturating_sub(1),
            required: k,
        });
    }
    Ok(())
}

/// Returns the inverse and log-determinant of the information, or `RankDeficient` when it is
/// numerically singular.
///
/// Plain Cholesky factors the information first. If the plain factor's smallest pivot is at or
/// below the rank cutoff of `pivoted_cholesky_inverse_logdet`, this function calls
/// `pivoted_cholesky_inverse_logdet` instead.
fn factor_information(information: &Mat<f64>) -> Result<CholeskySummary> {
    let k = information.nrows();
    let diagonal_scale = (0..k)
        .map(|i| information[(i, i)].abs())
        .fold(0.0_f64, f64::max);
    let pivot_cutoff = k as f64 * f64::EPSILON * diagonal_scale;
    match cholesky_inverse_logdet(information) {
        Ok(cholesky) if cholesky.minimum_pivot * cholesky.minimum_pivot > pivot_cutoff => {
            Ok(cholesky)
        },
        Ok(_) | Err(Error::NotPositiveDefinite) => pivoted_cholesky_inverse_logdet(information),
        Err(error) => Err(error),
    }
}

/// Adds the Firth correction to `modified_score`.
///
/// For coefficient `t`, the correction is
/// `penalty_weight * trace(fisher_inverse @ d_fisher/d_beta[t])`.
// `#[inline(never)]` limits the code size of `cox_quantities`. Benchmark fits before removing it.
#[inline(never)]
fn firth_correction(
    data: &CoxPrecomputed,
    fisher_inverse: &Mat<f64>,
    penalty_weight: f64,
    modified_score: &mut [f64],
    workspace: &mut CoxWorkspace,
) {
    let n = data.n_samples();
    let k = data.n_features();
    let design = MatRef::from_row_major_slice(&data.x, n, k);
    let CoxWorkspace {
        matrix_scratch,
        pass,
        normalized_risk,
        weights,
        firth,
        ..
    } = workspace;
    let FirthWorkspace {
        hat,
        h_bar,
        transformed,
    } = firth;

    // `hat[i] = dot(x[i] @ fisher_inverse, x[i])`.
    *hat = vec![0.0; n];
    quadratic_form_rows(design, fisher_inverse, hat, matrix_scratch);

    // `h_bar[e]` is the risk-weighted mean of `hat` over the risk set of block `e`. Like the
    // risk-set sums, its numerator accumulates forward over the samples.
    h_bar.clear();
    h_bar.reserve_exact(pass.blocks.len());
    let mut s0_h = 0.0;
    let mut start = 0;
    for block in &pass.blocks {
        for (i, &h) in hat.iter().enumerate().take(block.end).skip(start) {
            s0_h = pass.rescale(i) * s0_h + pass.scaled_risk[i] * h;
        }
        start = block.end;
        h_bar.push(s0_h / block.s0);
    }

    // Accumulate design' residual - x_bar' block_residual over the blocks in reverse.
    let hat = hat.as_slice();
    let weights = weights.as_slice();
    let normalized_risk = normalized_risk.as_slice();
    let mut correction = vec![0.0; k];
    let x_bar = MatRef::from_row_major_slice(&pass.x_bar, pass.blocks.len(), k);
    let chunk_rows = (FIRTH_CHUNK_BYTES / (size_of::<f64>() * k)).max(1);
    with_const_k!(
        k,
        reverse_rows(
            data,
            pass,
            x_bar,
            fisher_inverse,
            chunk_rows,
            hat,
            h_bar,
            weights,
            normalized_risk,
            transformed,
            &mut correction,
        ),
        dispatch!(Level::new(), simd => {
            let mut cumulative = vec![0.0; k];
            for_each_transformed_mean(x_bar, fisher_inverse, chunk_rows, transformed, |e, transformed_e| {
                let block = &pass.blocks[e];
                let carry = if e + 1 < pass.blocks.len() { pass.carry(e) } else { 0.0 };
                for (value, &mean) in cumulative.iter_mut().zip(transformed_e) {
                    *value = block.event_weight * mean + carry * *value;
                }
                let start = if e == 0 { 0 } else { pass.blocks[e - 1].end };
                for i in (start..block.end).rev() {
                    let row = data.row(i);
                    let form = dot(simd, row, &cumulative);
                    let residual = weights[i] * hat[i] - 2.0 * normalized_risk[i] * form;
                    axpy(simd, &mut correction, residual, row);
                }
                let x_bar_row = &pass.x_bar[e * k..(e + 1) * k];
                let mean_form = dot(simd, x_bar_row, transformed_e);
                let block_residual = block.event_weight * (h_bar[e] - 2.0 * mean_form);
                axpy(simd, &mut correction, -block_residual, x_bar_row);
            });
        })
    );
    for (score, correction) in modified_score.iter_mut().zip(&correction) {
        *score += penalty_weight * correction;
    }
}

/// Summary of one positive-weight event block from `risk_set_pass`.
struct EventBlock {
    /// End index (exclusive) of the block in the sorted samples.
    end: usize,
    /// Summed weight of the events in the block.
    event_weight: f64,
    /// Risk-set sum in the scale of `running_max_eta[end - 1]`.
    s0: f64,
    /// Log of the absolute risk-set sum, `running_max + ln(s0)`.
    log_s0: f64,
}

/// Results of one streaming pass over the samples in descending time order.
struct RiskSetPass {
    /// Coefficients that `risk_set_pass` used to compute the other fields.
    beta: Vec<f64>,
    /// Scaled risk `weight_i * exp(eta_i - running_max_eta[i])` for each sample.
    scaled_risk: Vec<f64>,
    /// Running maximum of `eta + ln(weight)` over samples `0..=i`.
    running_max_eta: Vec<f64>,
    blocks: Vec<EventBlock>,
    /// Risk-weighted covariate means at each event block, row-major `(n_event_blocks, k)`.
    x_bar: Vec<f64>,
    log_likelihood: f64,
    score: Vec<f64>,
}

impl RiskSetPass {
    /// Ratio of the risk-set sums of event blocks `e` and `e + 1`.
    fn carry(&self, e: usize) -> f64 {
        let (block, next) = (&self.blocks[e], &self.blocks[e + 1]);
        let (max, next_max) = (
            self.running_max_eta[block.end - 1],
            self.running_max_eta[next.end - 1],
        );
        let ratio = block.s0 / next.s0;
        if max < next_max {
            ratio * (max - next_max).exp()
        } else {
            ratio
        }
    }

    /// Factor that converts the sums over samples `0..i` from the scale of `running_max_eta[i - 1]`
    /// to the scale of `running_max_eta[i]`. The factor is 1.0 at `i = 0`.
    fn rescale(&self, i: usize) -> f64 {
        if i > 0 && self.running_max_eta[i] > self.running_max_eta[i - 1] {
            (self.running_max_eta[i - 1] - self.running_max_eta[i]).exp()
        } else {
            1.0
        }
    }
}

/// Scratch for likelihood evaluations.
///
/// Evaluations reuse the per-block buffers and the matrix chunks. Each evaluation allocates new
/// per-sample buffers.
struct CoxWorkspace {
    matrix_scratch: MatrixScratch,
    pass: RiskSetPass,
    /// Risk of each sample divided by the risk-set sum of the first event block whose risk set
    /// contains the sample.
    normalized_risk: Vec<f64>,
    /// Gram weights per sample.
    weights: Vec<f64>,
    cumulative_weight: Vec<f64>,
    event_weights: Vec<f64>,
    firth: FirthWorkspace,
    rank_checked: bool,
}

/// The penalized partial log-likelihood of one Cox data set.
struct CoxObjective<'a> {
    data: &'a CoxPrecomputed,
    inputs: &'a CaseInputs,
    penalty_weight: f64,
    workspace: CoxWorkspace,
}

impl<'a> CoxObjective<'a> {
    fn new(data: &'a CoxPrecomputed, inputs: &'a CaseInputs, penalty_weight: f64) -> Self {
        Self {
            data,
            inputs,
            penalty_weight,
            workspace: CoxWorkspace::new(data.n_features()),
        }
    }

    fn evaluate(&mut self, beta: &[f64]) -> Result<IterationQuantities> {
        cox_quantities(
            self.data,
            self.inputs,
            self.penalty_weight,
            beta,
            &mut self.workspace,
        )
    }
}

struct FirthWorkspace {
    hat: Vec<f64>,
    h_bar: Vec<f64>,
    /// `x_bar @ fisher_inverse` for one chunk of event blocks.
    transformed: Vec<f64>,
}

impl CoxWorkspace {
    fn new(k: usize) -> Self {
        Self {
            matrix_scratch: MatrixScratch::default(),
            pass: RiskSetPass {
                beta: Vec::new(),
                scaled_risk: Vec::new(),
                running_max_eta: Vec::new(),
                blocks: Vec::new(),
                x_bar: Vec::new(),
                log_likelihood: 0.0,
                score: vec![0.0; k],
            },
            normalized_risk: Vec::new(),
            weights: Vec::new(),
            cumulative_weight: Vec::new(),
            event_weights: Vec::new(),
            firth: FirthWorkspace {
                hat: Vec::new(),
                h_bar: Vec::new(),
                transformed: Vec::new(),
            },
            rank_checked: false,
        }
    }
}

/// Computes the partial log-likelihood, score, and risk-set summaries.
#[expect(clippy::too_many_lines)]
fn risk_set_pass(data: &CoxPrecomputed, inputs: &CaseInputs, beta: &[f64], pass: &mut RiskSetPass) {
    let n = data.n_samples();
    let k = data.n_features();
    let RiskSetPass {
        beta: pass_beta,
        scaled_risk,
        running_max_eta,
        blocks,
        x_bar,
        log_likelihood,
        score,
    } = pass;
    pass_beta.clear();
    pass_beta.extend_from_slice(beta);
    *scaled_risk = vec![0.0; n];
    *running_max_eta = vec![0.0; n];
    let scaled_risk = scaled_risk.as_mut_slice();
    let running_max_eta = running_max_eta.as_mut_slice();
    // With fewer than 16 features, a scalar dot product per row is faster than the matrix-vector
    // product `X @ beta`.
    let use_matvec = k >= 16;
    if use_matvec {
        matmul(
            ColMut::from_slice_mut(scaled_risk),
            Accum::Replace,
            MatRef::from_row_major_slice(&data.x, n, k),
            ColRef::from_slice(beta),
            1.0,
            Par::Seq,
        );
    }
    // `scaled_risk` holds the linear predictors until `exp_shifted` overwrites them. Adding
    // `eta_shift` applies the offsets and makes each risk `weight * exp(eta)`.
    let eta_shift = inputs.eta_shift.as_deref();
    let mut max_eta = f64::NEG_INFINITY;
    for i in 0..n {
        if !use_matvec {
            scaled_risk[i] = data.row(i).iter().zip(beta).map(|(x, b)| x * b).sum();
        }
        if let Some(eta_shift) = eta_shift {
            scaled_risk[i] += eta_shift[i];
        }
        max_eta = max_eta.max(scaled_risk[i]);
        running_max_eta[i] = max_eta;
    }
    // The running maximum is negative infinity at the samples before the first positive weight.
    // Setting their risks to zero avoids the NaN result of `exp(-inf - (-inf))`.
    let first = running_max_eta.partition_point(|&max| max == f64::NEG_INFINITY);
    scaled_risk[..first].fill(0.0);
    exp_shifted(&mut scaled_risk[first..], &running_max_eta[first..]);

    blocks.clear();
    // Reserving the capacity once avoids reallocating `blocks` as it grows in a new workspace. The
    // macOS allocator does not reuse the freed buffers. Without the reservation, the peak memory of
    // Cox inference grows with the number of likelihood ratio tests and profile bounds.
    blocks.reserve_exact(inputs.n_event_blocks);
    x_bar.resize(inputs.n_event_blocks * k, 0.0);
    let x_bar = x_bar.as_mut_slice();

    // `CaseInputs` contains the event totals of the score and the log-likelihood. The scan computes
    // only the risk-set terms.
    let mut log_likelihood_sum = CompensatedSum::default();
    let score = score.as_mut_slice();
    score.copy_from_slice(&inputs.event_feature_totals);
    with_const_k!(
        k,
        scan_rows(
            data,
            inputs,
            scaled_risk,
            running_max_eta,
            blocks,
            x_bar,
            score,
            &mut log_likelihood_sum,
        ),
        {
            let mut s0 = CompensatedSum::default();
            let mut s1 = vec![0.0; k];

            let mut block = 0;
            for i in 0..n {
                let row = data.row(i);
                let risk = scaled_risk[i];
                // Rescale the earlier sums when the running maximum changes.
                if i > 0 && running_max_eta[i] > running_max_eta[i - 1] {
                    let rescale = (running_max_eta[i - 1] - running_max_eta[i]).exp();
                    s0.scale(rescale);
                    for s1 in &mut s1 {
                        *s1 *= rescale;
                    }
                }
                s0.add(risk);
                for (s1, &x) in s1.iter_mut().zip(row) {
                    *s1 += x * risk;
                }

                if i + 1 != data.block_ends[block] {
                    continue;
                }
                let event_weight = inputs.event_weights[block];
                block += 1;
                if event_weight == 0.0 {
                    continue;
                }
                let s0 = s0.total();
                let s0_inv = 1.0 / s0;
                // `x_bar` is the risk-weighted mean of the covariates over the risk set.
                let means = &mut x_bar[blocks.len() * k..(blocks.len() + 1) * k];
                for ((mean, &s1), score) in means.iter_mut().zip(&s1).zip(score.iter_mut()) {
                    *mean = s1 * s0_inv;
                    *score -= event_weight * *mean;
                }
                // Add the running maximum back to undo the risk-set scaling.
                let log_s0 = running_max_eta[i] + s0.ln();
                log_likelihood_sum.add(-event_weight * log_s0);
                blocks.push(EventBlock {
                    end: i + 1,
                    event_weight,
                    s0,
                    log_s0,
                });
            }
        },
    );
    let event_eta: f64 = inputs
        .event_feature_totals
        .iter()
        .zip(beta)
        .map(|(total, beta)| total * beta)
        .sum();
    *log_likelihood = event_eta + inputs.event_offset_total + log_likelihood_sum.total();
}

/// Runs the loop of `risk_set_pass` over the samples for `K` features.
#[inline(always)]
#[expect(clippy::too_many_arguments)]
fn scan_rows<S: Simd, const K: usize>(
    _simd: S,
    data: &CoxPrecomputed,
    inputs: &CaseInputs,
    scaled_risk: &[f64],
    running_max_eta: &[f64],
    blocks: &mut Vec<EventBlock>,
    x_bar: &mut [f64],
    score: &mut [f64],
    log_likelihood_sum: &mut CompensatedSum,
) {
    let (rows, _) = data.x.as_chunks::<K>();
    let (means, _) = x_bar.as_chunks_mut::<K>();
    let mut score_sums = [0.0; K];
    score_sums.copy_from_slice(score);
    let mut s0 = CompensatedSum::default();
    let mut s1 = [0.0; K];
    let mut block = 0;
    for (i, row) in rows.iter().enumerate() {
        let risk = scaled_risk[i];
        // Rescale the earlier sums when the running maximum changes.
        if i > 0 && running_max_eta[i] > running_max_eta[i - 1] {
            let rescale = (running_max_eta[i - 1] - running_max_eta[i]).exp();
            s0.scale(rescale);
            for s1 in &mut s1 {
                *s1 *= rescale;
            }
        }
        s0.add(risk);
        for j in 0..K {
            s1[j] += row[j] * risk;
        }
        if i + 1 != data.block_ends[block] {
            continue;
        }
        let event_weight = inputs.event_weights[block];
        block += 1;
        if event_weight == 0.0 {
            continue;
        }
        let s0 = s0.total();
        let s0_inv = 1.0 / s0;
        // `x_bar` is the risk-weighted mean of the covariates over the risk set.
        let mean = &mut means[blocks.len()];
        for j in 0..K {
            mean[j] = s1[j] * s0_inv;
            score_sums[j] -= event_weight * mean[j];
        }
        // Add the running maximum back to undo the risk-set scaling.
        let log_s0 = running_max_eta[i] + s0.ln();
        log_likelihood_sum.add(-event_weight * log_s0);
        blocks.push(EventBlock {
            end: i + 1,
            event_weight,
            s0,
            log_s0,
        });
    }
    score.copy_from_slice(&score_sums);
}

/// Bytes per chunk of `x_bar @ fisher_inverse` in `firth_correction`. A chunk this size stays in
/// the L2 cache while the reverse pass reads it.
const FIRTH_CHUNK_BYTES: usize = 256 * 1024;

/// Calls `visit(e, x_bar[e] @ fisher_inverse)` for the rows `e` of `x_bar` in reverse order.
/// `rows_mul` computes the products into `transformed` for `chunk_rows` rows at a time.
#[inline(always)]
fn for_each_transformed_mean(
    x_bar: MatRef<f64>,
    fisher_inverse: &Mat<f64>,
    chunk_rows: usize,
    transformed: &mut Vec<f64>,
    mut visit: impl FnMut(usize, &[f64]),
) {
    let (n_blocks, k) = x_bar.shape();
    transformed.resize(chunk_rows.min(n_blocks) * k, 0.0);
    let mut chunk_end = n_blocks;
    while chunk_end > 0 {
        let chunk_start = chunk_end.saturating_sub(chunk_rows);
        let rows = chunk_end - chunk_start;
        let chunk = &mut transformed[..rows * k];
        rows_mul(
            x_bar.subrows(chunk_start, rows),
            fisher_inverse,
            MatMut::from_row_major_slice_mut(chunk, rows, k),
        );
        for e in (chunk_start..chunk_end).rev() {
            let offset = (e - chunk_start) * k;
            visit(e, &chunk[offset..offset + k]);
        }
        chunk_end = chunk_start;
    }
}

/// Runs the reverse pass of `firth_correction` for `K` features.
#[inline(always)]
#[expect(clippy::too_many_arguments)]
fn reverse_rows<S: Simd, const K: usize>(
    simd: S,
    data: &CoxPrecomputed,
    pass: &RiskSetPass,
    x_bar: MatRef<f64>,
    fisher_inverse: &Mat<f64>,
    chunk_rows: usize,
    hat: &[f64],
    h_bar: &[f64],
    weights: &[f64],
    normalized_risk: &[f64],
    transformed: &mut Vec<f64>,
    correction: &mut [f64],
) {
    let (rows, _) = data.x.as_chunks::<K>();
    let (means, _) = pass.x_bar.as_chunks::<K>();
    let n_blocks = pass.blocks.len();
    // `cumulative` is the sum of `event_weight[f] * S0[e] / S0[f] * x_bar[f] @ fisher_inverse` over
    // the blocks `f >= e`. `S0[e]` is the risk-set sum of block `e`.
    let mut cumulative = [0.0; K];
    let mut sums = [0.0; K];
    for_each_transformed_mean(
        x_bar,
        fisher_inverse,
        chunk_rows,
        transformed,
        |e, transformed_e| {
            let block = &pass.blocks[e];
            let carry = if e + 1 < n_blocks { pass.carry(e) } else { 0.0 };
            for j in 0..K {
                cumulative[j] = block.event_weight * transformed_e[j] + carry * cumulative[j];
            }
            let start = if e == 0 { 0 } else { pass.blocks[e - 1].end };
            for i in (start..block.end).rev() {
                let row = &rows[i];
                let form = dot(simd, row, &cumulative);
                let residual = weights[i] * hat[i] - 2.0 * normalized_risk[i] * form;
                for j in 0..K {
                    sums[j] += residual * row[j];
                }
            }
            let mean_form = dot(simd, &means[e], transformed_e);
            let block_residual = block.event_weight * (h_bar[e] - 2.0 * mean_form);
            for j in 0..K {
                sums[j] -= block_residual * means[e][j];
            }
        },
    );
    correction.copy_from_slice(&sums);
}

fn validate_survival_data(n_samples: usize, event: &[bool], time: &[f64]) -> Result<()> {
    if n_samples < 2 {
        return Err(Error::InvalidDimensions(format!(
            "Cox regression requires at least 2 samples, got {n_samples}"
        )));
    }
    if event.len() != n_samples || time.len() != n_samples {
        return Err(Error::InvalidDimensions(format!(
            "expected {n_samples} samples, got {} in event and {} in time",
            event.len(),
            time.len()
        )));
    }
    validate_times(time)?;
    if !event.iter().any(|&value| value) {
        return Err(Error::InvalidInput(
            "at least one event is required to fit a Cox model".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformed_means_match_for_every_chunk_size() {
        let (n_blocks, k) = (10, 9);
        let values: Vec<f64> = (0..n_blocks * k)
            .map(|i| ((i * 7) % 11) as f64 - 5.0)
            .collect();
        let x_bar = MatRef::from_row_major_slice(&values, n_blocks, k);
        let matrix = Mat::from_fn(k, k, |i, j| 1.0 / (1.0 + i as f64 + j as f64));
        for chunk_rows in [1, 3, n_blocks, 64] {
            let mut visited = Vec::new();
            for_each_transformed_mean(x_bar, &matrix, chunk_rows, &mut Vec::new(), |e, row| {
                visited.push((e, row.to_vec()));
            });
            let order: Vec<usize> = visited.iter().map(|(e, _)| *e).collect();
            assert_eq!(order, (0..n_blocks).rev().collect::<Vec<_>>());
            for (e, row) in &visited {
                for (j, &value) in row.iter().enumerate() {
                    let direct: f64 = (0..k).map(|m| x_bar[(*e, m)] * matrix[(m, j)]).sum();
                    assert!(
                        (value - direct).abs() < 1e-12,
                        "chunk_rows {chunk_rows}, block {e}: {value} vs {direct}"
                    );
                }
            }
        }
    }

    fn block_fixture() -> CoxPrecomputed {
        let x_data = [
            10.0, 1.0, // time 2, event 1
            20.0, 0.0, // time 5, event 1
            30.0, 3.0, // time 2, event 0
            40.0, 2.0, // time 5, event 1
            50.0, 5.0, // time 3, event 1
            60.0, 4.0, // time 4, event 0
        ];
        let x = MatrixView::from_row_major(&x_data, 6, 2).unwrap();
        let time = [2.0, 5.0, 2.0, 5.0, 3.0, 4.0];
        let event = [true, true, false, true, true, false];
        CoxPrecomputed::from_data(x, &time, &event)
    }

    /// The data come from the example of two individuals and one covariate in Section 2 of Heinze
    /// and Schemper (2001).
    #[test]
    fn two_individual_example_matches_log3() {
        let x_data = [1.0, 0.0];
        let x = MatrixView::from_row_major(&x_data, 2, 1).unwrap();
        let time = [1.0, 2.0];
        let event = [true, false];
        let expected = 3.0_f64.ln();

        let data = CoxPrecomputed::from_data(x, &time, &event);
        let inputs = CaseInputs::new(&data, None, None).unwrap();
        let quantities =
            cox_quantities(&data, &inputs, 0.5, &[expected], &mut CoxWorkspace::new(1)).unwrap();
        assert!(
            quantities.modified_score[0].abs() < 1e-12,
            "got {}",
            quantities.modified_score[0]
        );

        let fit = CoxData::new(x, &time, &event)
            .unwrap()
            .model()
            .fit()
            .unwrap();
        assert!(fit.status().is_converged());
        assert!(
            (fit.coefficients()[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            fit.coefficients()[0]
        );
    }

    /// With one coefficient, the constrained fit has no free coefficients. The LRT then evaluates
    /// the penalized log-likelihood at beta = 0 without running the solver.
    #[test]
    fn lrt_single_parameter() {
        let x_data = [1.0, 0.0];
        let x = MatrixView::from_row_major(&x_data, 2, 1).unwrap();
        let time = [1.0, 2.0];
        let event = [true, false];

        let fit = CoxData::new(x, &time, &event)
            .unwrap()
            .model()
            .fit()
            .unwrap();
        let lrt = fit.likelihood_ratio_test(&[0], true).unwrap();
        assert!(lrt.is_valid(), "status {:?}", lrt.status());

        let data = CoxPrecomputed::from_data(x, &time, &event);
        let inputs = CaseInputs::new(&data, None, None).unwrap();
        let null = cox_quantities(&data, &inputs, 0.5, &[0.0], &mut CoxWorkspace::new(1)).unwrap();
        let chi_squared = 2.0 * (fit.log_likelihood() - null.log_likelihood);

        assert!(
            (lrt.chi_squared() - chi_squared).abs() < 1e-12,
            "got {}, expected {chi_squared}",
            lrt.chi_squared()
        );
    }

    #[test]
    fn baseline_hazard_matches_direct_log_sum_exp() {
        let n = 1_500;
        let k = 3;
        let data = generated_fixture(n, k);
        let beta = [0.7, -0.4, 0.25];
        let center_linear_predictor: f64 = data
            .feature_centers
            .iter()
            .zip(&beta)
            .map(|(center, coefficient)| center * coefficient)
            .sum();

        for weighted in [false, true] {
            let (inputs, weight, offset) = fixture_inputs(&data, weighted);
            let mut workspace = CoxWorkspace::new(k);
            cox_quantities(&data, &inputs, 0.5, &beta, &mut workspace).unwrap();
            let (unique_times, log_cumulative_hazard) = baseline_hazard(&data, &workspace.pass);

            let eta: Vec<f64> = (0..data.n_samples())
                .map(|i| {
                    data.row(i)
                        .iter()
                        .zip(&beta)
                        .map(|(x, b)| x * b)
                        .sum::<f64>()
                        + center_linear_predictor
                        + offset[i]
                })
                .collect();
            let mut expected_times = Vec::new();
            let mut increments = Vec::new();
            let mut start = 0;
            for &end in &data.block_ends {
                let d: f64 = (start..end)
                    .filter(|&i| data.event[i])
                    .map(|i| weight[i])
                    .sum();
                start = end;
                if d == 0.0 {
                    continue;
                }
                let shift = eta[..end].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let log_s0 = shift
                    + (0..end)
                        .map(|i| weight[i] * (eta[i] - shift).exp())
                        .sum::<f64>()
                        .ln();
                expected_times.push(data.time[end - 1]);
                increments.push(d.ln() - log_s0);
            }
            expected_times.reverse();
            increments.reverse();
            let mut expected = increments;
            for i in 1..expected.len() {
                expected[i] = log_add_exp(expected[i - 1], expected[i]);
            }

            assert_eq!(unique_times, expected_times);
            assert!(unique_times.len() > 100);
            for (observed, expected) in log_cumulative_hazard.iter().zip(&expected) {
                assert!(
                    (observed - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "weighted={weighted}: {observed} vs {expected}"
                );
            }
        }
    }

    fn log_add_exp(a: f64, b: f64) -> f64 {
        a.max(b) + (-(a - b).abs()).exp().ln_1p()
    }

    /// Data with tied times, eventless blocks, and one extreme linear predictor that raises the
    /// running maximum partway through the descending-time order.
    fn generated_fixture(n: usize, k: usize) -> CoxPrecomputed {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 11) as f64 / 9_007_199_254_740_992.0
        };
        let mut x_data: Vec<f64> = (0..n * k).map(|_| 2.0 * next() - 1.0).collect();
        let time: Vec<f64> = (0..n)
            .map(|_| (1.0 + (next() * 300.0).floor()) / 10.0)
            .collect();
        let event: Vec<bool> = (0..n).map(|_| next() < 0.6).collect();
        x_data[7 * k] = 60.0;
        let x = MatrixView::from_row_major(&x_data, n, k).unwrap();
        CoxPrecomputed::from_data(x, &time, &event)
    }

    /// Returns case inputs for `data`, and their weights and offsets in sorted order. When
    /// `weighted` is false, the inputs come from `CaseInputs::new(data, None, None)`. When
    /// `weighted` is true, the weights include zero and non-integer values. The five samples with
    /// the latest times also have zero weight.
    fn fixture_inputs(data: &CoxPrecomputed, weighted: bool) -> (CaseInputs, Vec<f64>, Vec<f64>) {
        let n = data.n_input_rows;
        let (mut weight, mut offset) = (vec![1.0; n], vec![0.0; n]);
        let inputs = if weighted {
            for row in 0..n {
                weight[row] = [0.0, 0.5, 1.0, 2.5][row % 4];
                offset[row] = 0.3 * ((row % 5) as f64 - 2.0);
            }
            for &row in &data.order[..5] {
                weight[row] = 0.0;
            }
            CaseInputs::new(data, Some(&weight), Some(&offset))
        } else {
            CaseInputs::new(data, None, None)
        };
        let sorted = |values: &[f64]| data.order.iter().map(|&row| values[row]).collect();
        (inputs.unwrap(), sorted(&weight), sorted(&offset))
    }

    /// Computes the penalized partial log-likelihood, the Fisher information, and the
    /// Firth-modified score directly from the moments `S0 = sum w exp(eta)`,
    /// `S1 = sum w exp(eta) x`, `S2 = sum w exp(eta) x x'`, and `S3_t = sum w exp(eta) x_t x x'` of
    /// each risk set. The risk set of a block contains every row of the sorted data up to the end
    /// of that block. The weights `w` and the offsets in `eta` are in sorted order.
    #[expect(clippy::needless_range_loop)]
    fn direct_cox_quantities(
        data: &CoxPrecomputed,
        weight: &[f64],
        offset: &[f64],
        penalty_weight: f64,
        beta: &[f64],
    ) -> (f64, Vec<f64>, Mat<f64>, Vec<Mat<f64>>, f64) {
        let n = data.n_samples();
        let k = data.n_features();
        let eta: Vec<f64> = (0..n)
            .map(|i| {
                data.row(i)
                    .iter()
                    .zip(beta)
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
                    + offset[i]
            })
            .collect();
        let shift = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let risk: Vec<f64> = (0..n).map(|i| weight[i] * (eta[i] - shift).exp()).collect();

        let mut log_likelihood = 0.0;
        let mut score = vec![0.0; k];
        let mut fisher = Mat::<f64>::zeros(k, k);
        let mut fisher_derivatives = vec![Mat::<f64>::zeros(k, k); k];
        let mut mean_gram_scale = 0.0_f64;
        let mut start = 0;
        for &end in &data.block_ends {
            let d: f64 = (start..end)
                .filter(|&i| data.event[i])
                .map(|i| weight[i])
                .sum();
            start = end;
            if d == 0.0 {
                continue;
            }
            let s0: f64 = risk[..end].iter().sum();
            let mut s1 = vec![0.0; k];
            let mut s2 = Mat::<f64>::zeros(k, k);
            let mut s3 = vec![Mat::<f64>::zeros(k, k); k];
            for i in 0..end {
                let row = data.row(i);
                for r in 0..k {
                    s1[r] += risk[i] * row[r];
                    for s in 0..k {
                        s2[(r, s)] += risk[i] * row[r] * row[s];
                        for t in 0..k {
                            s3[t][(r, s)] += risk[i] * row[t] * row[r] * row[s];
                        }
                    }
                }
            }
            let x_bar: Vec<f64> = s1.iter().map(|v| v / s0).collect();
            let max_mean = x_bar
                .iter()
                .fold(0.0_f64, |scale, value| scale.max(value.abs()));
            mean_gram_scale += d * max_mean * max_mean;
            for r in 0..k {
                score[r] -= d * x_bar[r];
                for s in 0..k {
                    let information = s2[(r, s)] / s0 - x_bar[r] * x_bar[s];
                    fisher[(r, s)] += d * information;
                    // d/d beta_t of `S2/S0 - x_bar x_bar'`.
                    for t in 0..k {
                        fisher_derivatives[t][(r, s)] += d
                            * (s3[t][(r, s)] / s0
                                - x_bar[t] * s2[(r, s)] / s0
                                - s2[(r, t)] / s0 * x_bar[s]
                                - x_bar[r] * s2[(t, s)] / s0
                                + 2.0 * x_bar[t] * x_bar[r] * x_bar[s]);
                    }
                }
            }
            log_likelihood -= d * (shift + s0.ln());
        }
        for i in (0..n).filter(|&i| data.event[i]) {
            log_likelihood += weight[i] * eta[i];
            for (score, &x) in score.iter_mut().zip(data.row(i)) {
                *score += weight[i] * x;
            }
        }
        let summary = cholesky_inverse_logdet(&fisher).unwrap();
        log_likelihood += penalty_weight * summary.log_determinant;
        let mut modified_score = score;
        for (t, entry) in modified_score.iter_mut().enumerate() {
            let mut trace = 0.0;
            for r in 0..k {
                for s in 0..k {
                    trace += summary.inverse[(r, s)] * fisher_derivatives[t][(s, r)];
                }
            }
            *entry += penalty_weight * trace;
        }
        (
            log_likelihood,
            modified_score,
            fisher,
            fisher_derivatives,
            mean_gram_scale,
        )
    }

    #[test]
    fn cox_quantities_match_direct_risk_set_moments() {
        // For k = 3, `risk_set_pass` computes the linear predictors with scalar dot products.
        // For k = 16 and k = 17, it computes them with a matrix-vector product. The loops for more
        // than 16 features run only for k = 17. The test checks each k with and without weights
        // and offsets.
        for (k, weighted) in [
            (3, false),
            (3, true),
            (16, false),
            (16, true),
            (17, false),
            (17, true),
        ] {
            let data = generated_fixture(80, k);
            let beta: Vec<f64> = (0..k).map(|j| [0.7, -0.4, 0.25][j % 3]).collect();
            let penalty_weight = 0.5;
            let (inputs, weight, offset) = fixture_inputs(&data, weighted);
            let direct = |penalty_weight, beta: &[f64]| {
                direct_cox_quantities(&data, &weight, &offset, penalty_weight, beta)
            };

            let (log_likelihood, modified_score, fisher, derivatives, mean_gram_scale) =
                direct(penalty_weight, &beta);

            // Check the derivatives of the reference Fisher information by finite differences when
            // k = 3.
            if k == 3 {
                let step = 1e-5;
                for t in 0..k {
                    let mut plus = beta.clone();
                    let mut minus = beta.clone();
                    plus[t] += step;
                    minus[t] -= step;
                    let (_, _, fisher_plus, _, _) = direct(0.0, &plus);
                    let (_, _, fisher_minus, _, _) = direct(0.0, &minus);
                    for r in 0..k {
                        for s in 0..k {
                            let numeric =
                                (fisher_plus[(r, s)] - fisher_minus[(r, s)]) / (2.0 * step);
                            let analytic = derivatives[t][(r, s)];
                            assert!(
                                (numeric - analytic).abs() <= 1e-5 * analytic.abs().max(1.0),
                                "dF[{r},{s}]/d beta_{t}: {numeric} vs {analytic}"
                            );
                        }
                    }
                }
            }

            // Evaluate at another beta first so the checked call reuses a workspace holding values
            // from a different beta. `CoxObjective` reuses its workspace in the same way between
            // Newton iterations.
            let mut workspace = CoxWorkspace::new(k);
            cox_quantities(
                &data,
                &inputs,
                penalty_weight,
                &vec![0.1; k],
                &mut workspace,
            )
            .unwrap();
            let quantities =
                cox_quantities(&data, &inputs, penalty_weight, &beta, &mut workspace).unwrap();
            assert!(
                (quantities.log_likelihood - log_likelihood).abs()
                    <= 1e-11 * log_likelihood.abs().max(1.0),
                "{} vs {log_likelihood}",
                quantities.log_likelihood
            );
            let scale = (0..k).map(|r| fisher[(r, r)]).fold(0.0_f64, f64::max);
            let fisher_tolerance = 1e-11 * scale + 64.0 * f64::EPSILON * mean_gram_scale;
            for r in 0..k {
                for s in 0..k {
                    assert!(
                        (quantities.step_matrix[(r, s)] - fisher[(r, s)]).abs() <= fisher_tolerance,
                        "F[{r},{s}]: {} vs {}",
                        quantities.step_matrix[(r, s)],
                        fisher[(r, s)]
                    );
                }
            }
            let score_scale = modified_score
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs()));
            for (t, (observed, expected)) in quantities
                .modified_score
                .iter()
                .zip(&modified_score)
                .enumerate()
            {
                assert!(
                    (observed - expected).abs() <= 1e-10 * score_scale,
                    "score[{t}]: {observed} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn baseline_hazard_skips_eventless_blocks() {
        let data = block_fixture();
        let inputs = CaseInputs::new(&data, None, None).unwrap();

        let mut workspace = CoxWorkspace::new(2);
        cox_quantities(&data, &inputs, 0.0, &[0.0, 0.0], &mut workspace).unwrap();
        let (unique_times, log_cumulative_hazard) = baseline_hazard(&data, &workspace.pass);

        assert_eq!(unique_times, vec![2.0, 3.0, 5.0]);
        let expected = [
            1.0 / 6.0,
            1.0 / 6.0 + 1.0 / 4.0,
            1.0 / 6.0 + 1.0 / 4.0 + 1.0,
        ];
        for (i, (actual, expected)) in log_cumulative_hazard.iter().zip(expected).enumerate() {
            assert!(
                (actual.exp() - expected).abs() < 1e-12,
                "index {i}: got {}, expected {expected}",
                actual.exp()
            );
        }
    }

    #[test]
    fn validation_rejects_bad_outcomes() {
        let event = [true, false];
        let time = [1.0, 2.0];

        assert!(matches!(
            validate_survival_data(1, &[true], &[1.0]),
            Err(Error::InvalidDimensions(_))
        ));
        assert!(matches!(
            validate_survival_data(2, &[true], &time),
            Err(Error::InvalidDimensions(_))
        ));
        assert!(matches!(
            validate_survival_data(2, &event, &[1.0, f64::NAN]),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            validate_survival_data(2, &event, &[1.0, -1.0]),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            validate_survival_data(2, &[false, false], &time),
            Err(Error::InvalidInput(_))
        ));
    }
}
