use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use faer::{Mat, MatRef};
use fearless_simd::{Level, dispatch, prelude::*};

use crate::inference::{Estimate, LrtResult, ProfileBound, ProfileOptions, Tail};
use crate::linalg::{
    MatrixScratch, add_transpose_mul_vec, add_weighted_gram, cholesky_factor_logdet,
    copy_lower_to_upper, mul_vec, normalized_qr_pivots, pivoted_cholesky_pivot_ratio,
    pivoted_qr_hat_logdet, weighted_gram,
};
use crate::solver::{FitStatus, IterationQuantities, NewtonOptions, newton_raphson};
use crate::sum::CompensatedSum;
use crate::validation::{
    validate_feature_count, validate_finite_features, validate_nonnegative_weights,
    validate_optional_vector, validate_penalty_weight,
};
use crate::{Error, MatrixView, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticOptions {
    /// Weight of the penalty term. The standard Firth weight is 0.5.
    pub penalty_weight: f64,
    pub solver: NewtonOptions,
}

impl Default for LogisticOptions {
    fn default() -> Self {
        Self {
            penalty_weight: 0.5,
            solver: NewtonOptions {
                max_iterations: 25,
                max_step: 5.0,
                max_half_steps: 25,
                gradient_tolerance: 1e-4,
                step_tolerance: 1e-4,
            },
        }
    }
}

/// A logistic regression design, checked and copied once. Every model from
/// [`model`](Self::model) reuses the copy.
///
/// Clones share one copy of the design.
#[derive(Clone)]
pub struct LogisticData {
    inner: Arc<LogisticDesign>,
}

struct LogisticDesign {
    /// The features, followed by a column of ones when an intercept is fitted.
    design: Mat<f64>,
    n_features: usize,
    fit_intercept: bool,
}

impl LogisticData {
    /// Checks `features` and copies them into a column-major matrix. The matrix has a final column
    /// of ones for the intercept.
    ///
    /// # Errors
    ///
    /// Returns `InvalidDimensions` for fewer than two rows, or `InvalidInput` if a feature is not
    /// finite.
    pub fn new(features: MatrixView) -> Result<Self> {
        Self::from_features(features, true)
    }

    /// Checks `features` and copies them into a column-major matrix without an intercept column.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`LogisticData::new`].
    pub fn without_intercept(features: MatrixView) -> Result<Self> {
        Self::from_features(features, false)
    }

    fn from_features(features: MatrixView, fit_intercept: bool) -> Result<Self> {
        let nrows = features.nrows();
        let n_features = features.ncols();
        if nrows < 2 {
            return Err(Error::InvalidDimensions(
                "logistic regression requires at least two samples".into(),
            ));
        }
        validate_finite_features(features)?;

        let k = n_features + usize::from(fit_intercept);
        // `resize_with` fills the new rows one column at a time. The loop copies row-major input in
        // blocks of 512 rows so that each block stays in cache while `resize_with` fills every
        // column. It copies column-major input in one block because the columns are contiguous.
        let block = if features.is_column_major() {
            nrows
        } else {
            512
        };
        let mut design = Mat::with_capacity(nrows, k);
        for start in (0..nrows).step_by(block) {
            design.resize_with((start + block).min(nrows), k, |i, j| {
                if j < n_features {
                    features.at(i, j)
                } else {
                    1.0
                }
            });
        }
        Ok(Self {
            inner: Arc::new(LogisticDesign {
                design,
                n_features,
                fit_intercept,
            }),
        })
    }

    /// Returns the logistic regression of `response` on this design. The model has unit weights,
    /// zero offsets, and default options until [`sample_weight`](LogisticModel::sample_weight),
    /// [`offset`](LogisticModel::offset), or [`options`](LogisticModel::options) changes them.
    pub fn model<'a>(&'a self, response: &'a [f64]) -> LogisticModel<'a> {
        LogisticModel {
            data: self,
            response,
            sample_weight: None,
            offset: None,
            options: LogisticOptions::default(),
        }
    }

    /// Returns a copy of the features in column-major order.
    #[must_use]
    pub fn features_column_major(&self) -> Vec<f64> {
        let mut features = Vec::with_capacity(self.nrows() * self.n_features());
        for j in 0..self.n_features() {
            features.extend_from_slice(self.inner.design.col_as_slice(j));
        }
        features
    }

    #[must_use]
    pub fn nrows(&self) -> usize {
        self.inner.design.nrows()
    }

    /// Returns the number of feature columns, not including the intercept.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.inner.n_features
    }

    /// Returns false if [`LogisticData::without_intercept`] created the data.
    #[must_use]
    pub fn has_intercept(&self) -> bool {
        self.inner.fit_intercept
    }
}

// A derived `Debug` implementation would print every entry of the design.
impl fmt::Debug for LogisticData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogisticData")
            .field("nrows", &self.nrows())
            .field("n_features", &self.n_features())
            .field("fit_intercept", &self.inner.fit_intercept)
            .finish_non_exhaustive()
    }
}

/// A logistic regression model before it is fitted. [`LogisticData::model`] creates the model.
#[derive(Clone, Copy)]
#[must_use = "call `fit` to fit the model"]
pub struct LogisticModel<'a> {
    data: &'a LogisticData,
    response: &'a [f64],
    sample_weight: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    options: LogisticOptions,
}

impl<'a> LogisticModel<'a> {
    /// `sample_weight[i]` is the nonnegative weight of row `i`. Weights are counts. An integer
    /// weight `m` gives the same estimates, standard errors, likelihood ratio tests, and profile
    /// intervals as `m` copies of the row. A zero weight gives the same results as removing the
    /// row.
    pub fn sample_weight(mut self, sample_weight: &'a [f64]) -> Self {
        self.sample_weight = Some(sample_weight);
        self
    }

    /// `offset[i]` is added to the linear predictor of row `i`.
    pub fn offset(mut self, offset: &'a [f64]) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn options(mut self, options: LogisticOptions) -> Self {
        self.options = options;
        self
    }

    /// Fits the Firth-penalized logistic regression.
    ///
    /// # Errors
    ///
    /// Returns `InvalidDimensions` for mismatched lengths or for fewer positive-weight rows than
    /// parameters. Returns `InvalidInput` for out-of-range values or options, or when a class has no
    /// positive-weight row. Returns `RankDeficient` or `IllConditionedDesign` when the fit finds that
    /// the positive-weight rows of the design are numerically rank deficient or ill-conditioned. The
    /// fit runs this check only when the information is close to singular. Returns a numerical error
    /// if a factorization fails at the starting coefficients or while the solver computes a Newton
    /// step.
    pub fn fit(self) -> Result<LogisticFit> {
        let Self {
            data,
            response,
            sample_weight,
            offset,
            options,
        } = self;
        let design = &data.inner.design;
        let (nrows, k) = design.shape();
        if response.len() != nrows {
            return Err(Error::InvalidDimensions(format!(
                "response has length {}, expected {nrows}",
                response.len()
            )));
        }
        validate_penalty_weight(options.penalty_weight)?;
        if response.iter().any(|&value| ![0.0, 1.0].contains(&value)) {
            return Err(Error::InvalidInput(
                "response must contain only 0.0 and 1.0".into(),
            ));
        }

        options.solver.validate()?;

        let sample_weight = validate_optional_vector(sample_weight, nrows, 1.0, "sample_weight")?;
        validate_nonnegative_weights(&sample_weight)?;
        let offset = validate_optional_vector(offset, nrows, 0.0, "offset")?;

        #[expect(clippy::float_cmp)]
        for class in [0.0, 1.0] {
            if !(0..nrows).any(|i| response[i] == class && sample_weight[i] > 0.0) {
                return Err(Error::InvalidInput(
                    "at least one positive-weight sample is required in each class".into(),
                ));
            }
        }

        let effective_n = sample_weight.iter().filter(|&&weight| weight > 0.0).count();
        if effective_n < k {
            return Err(Error::InvalidDimensions(format!(
                "number of parameters ({k}) exceeds the number of positive-weight samples ({effective_n})"
            )));
        }

        let design_checked = AtomicBool::new(false);
        let mut objective = LogisticObjective::new(
            design,
            response,
            &sample_weight,
            &offset,
            options.penalty_weight,
            &design_checked,
        );
        let result = newton_raphson(options.solver, vec![0.0; k], |beta| {
            objective.evaluate(beta)
        })?;
        objective.design_check.failure()?;

        Ok(LogisticFit {
            estimate: Estimate::new(result),
            data: data.clone(),
            response: response.to_vec(),
            sample_weight,
            offset,
            options,
            design_checked,
        })
    }
}

// A derived `Debug` implementation would print every row of the response, weights, and offset.
impl fmt::Debug for LogisticModel<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogisticModel")
            .field("data", self.data)
            .field("sample_weight_len", &self.sample_weight.map(<[f64]>::len))
            .field("offset_len", &self.offset.map(<[f64]>::len))
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

pub struct LogisticFit {
    estimate: Estimate,
    // Refits for likelihood ratio tests and profile bounds use these fields.
    data: LogisticData,
    response: Vec<f64>,
    sample_weight: Vec<f64>,
    offset: Vec<f64>,
    options: LogisticOptions,
    /// True when the positive-weight rows of the design have passed `check_design_conditioning`.
    design_checked: AtomicBool,
}

// A derived `Debug` implementation would print every row of the response, weights, and offset.
impl fmt::Debug for LogisticFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogisticFit")
            .field("parameters", &self.estimate.beta)
            .field("standard_errors", &self.estimate.standard_errors)
            .field("log_likelihood", &self.estimate.log_likelihood)
            .field("status", &self.estimate.status)
            .field("iterations", &self.estimate.iterations)
            .field("data", &self.data)
            .finish_non_exhaustive()
    }
}

impl LogisticFit {
    fn objective(&self) -> LogisticObjective<'_> {
        LogisticObjective::new(
            &self.data.inner.design,
            &self.response,
            &self.sample_weight,
            &self.offset,
            self.options.penalty_weight,
            &self.design_checked,
        )
    }

    /// Returns the feature matrix of the fit. A caller can store this along with
    /// [`response`](Self::response), [`sample_weight`](Self::sample_weight),
    /// [`offset`](Self::offset), and [`options`](Self::options) to rebuild the fit later.
    #[must_use]
    pub fn data(&self) -> &LogisticData {
        &self.data
    }

    #[must_use]
    pub fn response(&self) -> &[f64] {
        &self.response
    }

    /// Returns unit weights if the model had no sample weights.
    #[must_use]
    pub fn sample_weight(&self) -> &[f64] {
        &self.sample_weight
    }

    /// Returns zero offsets if the model had no offsets.
    #[must_use]
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    #[must_use]
    pub fn options(&self) -> LogisticOptions {
        self.options
    }

    /// Feature coefficients, followed by the intercept if present.
    #[must_use]
    pub fn parameters(&self) -> &[f64] {
        &self.estimate.beta
    }

    /// Feature coefficients, excluding the intercept.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.estimate.beta[..self.data.inner.n_features]
    }

    /// Returns 0.0 if [`LogisticData::without_intercept`] created the data.
    #[must_use]
    pub fn intercept(&self) -> f64 {
        if self.data.inner.fit_intercept {
            self.estimate.beta[self.data.inner.n_features]
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn log_likelihood(&self) -> f64 {
        self.estimate.log_likelihood
    }

    /// Standard errors from the inverse Fisher information at the penalized estimate.
    #[must_use]
    pub fn standard_errors(&self) -> &[f64] {
        &self.estimate.standard_errors
    }

    /// Returns the inverse Fisher information at the penalized estimate in row-major order.
    ///
    /// Rows and columns are in the order of [`parameters`](Self::parameters). The result is
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

    /// Wald confidence intervals for all parameters.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `alpha` is outside the open interval (0, 1).
    pub fn wald_confidence_intervals(&self, alpha: f64) -> Result<Vec<[f64; 2]>> {
        self.estimate.wald_confidence_intervals(alpha)
    }

    /// Penalized likelihood ratio test that the parameters at `indices` are all zero. The entries
    /// of `indices` are positions in [`parameters`](Self::parameters). The test has `indices.len()`
    /// degrees of freedom.
    ///
    /// The test refits the model with the parameters at `indices` fixed to zero.
    ///
    /// When `warm_start` is false, the other parameters start at zero.
    /// When `warm_start` is true, the other parameters start at their conditional mean given that the fixed
    /// parameters are zero. Let `C` be the inverse Fisher information of the full fit, which is
    /// returned by [`covariance`](Self::covariance). The conditional mean is computed as if
    /// the full estimate `beta` were normal with covariance `C`. Let `S` be the positions in `indices`.
    /// Let `F` be the other positions. The start is then `beta_F - C_FS C_SS^-1 beta_S`. With one
    /// tested parameter `j`, each free parameter `f` starts at `beta_f - beta_j C_fj / C_jj`.
    ///
    /// Equivalently, the start is the result of one Newton step from the full estimate with the
    /// parameters at `indices` set to zero. The step changes only the other parameters. It uses a
    /// quadratic approximation of the penalized log-likelihood around the full estimate.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `indices` is empty, repeats an index, or contains an index past
    /// the last parameter. Returns `RankDeficient` or `IllConditionedDesign` if the conditioning
    /// check of the design fails during the constrained fit. This method does not return an error
    /// when the full or constrained fit does not converge. The returned `LrtResult` then has status
    /// `FullFitNonconvergence` or `ConstrainedFitNonconvergence`.
    pub fn likelihood_ratio_test(&self, indices: &[usize], warm_start: bool) -> Result<LrtResult> {
        // The objective allocates buffers for every row. A test of a fit that did not converge never
        // evaluates it.
        let mut objective = None;
        let result = self.estimate.likelihood_ratio_test(
            indices,
            warm_start,
            self.options.solver,
            |beta| {
                objective
                    .get_or_insert_with(|| self.objective())
                    .evaluate(beta)
            },
        )?;
        if let Some(objective) = &objective {
            objective.design_check.failure()?;
        }
        Ok(result)
    }

    /// Computes one bound of a profile-likelihood confidence interval with the Venzon-Moolgavkar
    /// (1988) algorithm.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `index`, `alpha`, or a field of `options` is out of range. Returns
    /// a numerical error if a solve fails during root finding. Returns `RankDeficient` or
    /// `IllConditionedDesign` if the conditioning check of the design fails at an evaluation. When
    /// the fit or the profile iteration does not converge, this method returns a `ProfileBound`
    /// with status `FullFitNonconvergence` or `ProfileBoundNonconvergence`.
    pub fn profile_confidence_bound(
        &self,
        index: usize,
        tail: Tail,
        alpha: f64,
        options: ProfileOptions,
    ) -> Result<ProfileBound> {
        let mut objective = None;
        self.estimate
            .profile_bound(index, tail, alpha, options, |beta| {
                objective
                    .get_or_insert_with(|| self.objective())
                    .evaluate(beta)
            })
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
        let mut objective = None;
        self.estimate
            .profile_interval(index, alpha, options, |beta| {
                objective
                    .get_or_insert_with(|| self.objective())
                    .evaluate(beta)
            })
    }

    /// Computes the log-odds `X @ coefficients + intercept + offset` for each row.
    ///
    /// # Errors
    ///
    /// Returns `InvalidDimensions` for mismatched shapes and `InvalidInput` for non-finite values.
    pub fn decision_function(
        &self,
        features: MatrixView,
        offset: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let nrows = features.nrows();
        validate_feature_count(features, self.data.inner.n_features)?;
        validate_finite_features(features)?;

        let offset = validate_optional_vector(offset, nrows, 0.0, "offset")?;
        Ok((0..nrows)
            .map(|i| {
                let eta = (0..self.data.inner.n_features)
                    .map(|j| features.at(i, j) * self.estimate.beta[j])
                    .sum::<f64>();

                eta + self.intercept() + offset[i]
            })
            .collect())
    }

    /// Computes the fitted probability of the positive outcome (`y = 1`) for each row.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::decision_function`].
    pub fn predict_probability(
        &self,
        features: MatrixView,
        offset: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        Ok(self
            .decision_function(features, offset)?
            .into_iter()
            .map(expit)
            .collect())
    }
}

/// Checks the QR pivots of the positive-weight rows of `design` after each column is scaled to unit
/// norm. `n` is the number of positive-weight rows. Returns `RankDeficient` when a pivot is at most
/// `max(n, k) * eps` times the leading pivot. Returns `IllConditionedDesign` when a pivot is at
/// most `sqrt(max(n, k) * eps)` times the leading pivot.
fn check_design_conditioning(
    design: &Mat<f64>,
    sample_weight: &[f64],
    checked: &AtomicBool,
) -> Result<()> {
    if checked.load(Ordering::Relaxed) {
        return Ok(());
    }

    let n = sample_weight.iter().filter(|&&weight| weight > 0.0).count();
    let k = design.ncols();
    let rows: Vec<usize> = sample_weight
        .iter()
        .enumerate()
        .filter_map(|(i, &weight)| (weight > 0.0).then_some(i))
        .collect();
    let pivots = normalized_qr_pivots(n, k, |i, j| design[(rows[i], j)]);
    let qr_tolerance = n.max(k) as f64 * f64::EPSILON;
    let rank_cutoff = qr_tolerance * pivots[0];
    let rank = pivots.iter().filter(|&&pivot| pivot > rank_cutoff).count();
    if rank < k {
        return Err(Error::RankDeficient { rank, required: k });
    }

    // Forming the Fisher information squares the condition number of the design. The cutoff
    // therefore uses the square root of `qr_tolerance`.
    let normal_equations_cutoff = qr_tolerance.sqrt() * pivots[0];
    if pivots.iter().any(|&pivot| pivot <= normal_equations_cutoff) {
        return Err(Error::IllConditionedDesign);
    }
    checked.store(true, Ordering::Relaxed);
    Ok(())
}

/// The design conditioning check that one objective runs.
struct DesignCheck<'a> {
    /// Every objective of one fit uses the same flag.
    passed: &'a AtomicBool,
    /// The error of a failed check. Unlike `passed`, `error` is not shared between objectives. A
    /// failed check in one likelihood ratio test or profile bound therefore does not change the
    /// result of a later call on the same fit.
    error: Option<Error>,
}

impl DesignCheck<'_> {
    fn run(&mut self, design: &Mat<f64>, sample_weight: &[f64]) -> Result<()> {
        let result = check_design_conditioning(design, sample_weight, self.passed);
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
    }

    /// Returns the error of a failed check.
    ///
    /// `evaluate` calls this method first. After a failed check, every later evaluation therefore
    /// returns the error. When the check fails at a trial step, the solver rejects every halved
    /// step and returns `StepHalvingFailed`. The function that ran the solver then returns the
    /// error from this method.
    fn failure(&self) -> Result<()> {
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
}

/// Buffers reused across likelihood evaluations of one design.
struct LogisticWorkspace {
    matrix_scratch: MatrixScratch,
    eta: Vec<f64>,
    /// `p * (1 - p)` for each row, where `p` is the fitted probability.
    variance: Vec<f64>,
    y_minus_p: Vec<f64>,
    weights: Vec<f64>,
    h: Vec<f64>,
    residual: Vec<f64>,
    /// Indices of the rows `J_S` augments.
    augmented_rows: Vec<usize>,
    /// A column-major copy of the design rows whose indices are in `augmented_rows`.
    gathered_design: Vec<f64>,
    /// `2*penalty_weight*h_i * p_i*(1 - p_i)` for each row in `augmented_rows`.
    augmentation_weights: Vec<f64>,
}

impl LogisticWorkspace {
    fn new(n: usize) -> Self {
        Self {
            matrix_scratch: MatrixScratch::default(),
            eta: vec![0.0; n],
            variance: vec![0.0; n],
            y_minus_p: vec![0.0; n],
            weights: vec![0.0; n],
            h: vec![0.0; n],
            residual: vec![0.0; n],
            augmented_rows: Vec::new(),
            gathered_design: Vec::new(),
            augmentation_weights: Vec::new(),
        }
    }
}

/// The penalized log-likelihood of one logistic design.
struct LogisticObjective<'a> {
    design: &'a Mat<f64>,
    response: &'a [f64],
    sample_weight: &'a [f64],
    unit_weights: bool,
    offset: &'a [f64],
    penalty_weight: f64,
    design_check: DesignCheck<'a>,
    /// An unpenalized objective tests the information only at its first evaluation.
    evaluated: bool,
    workspace: LogisticWorkspace,
}

impl<'a> LogisticObjective<'a> {
    #[expect(clippy::float_cmp)]
    fn new(
        design: &'a Mat<f64>,
        response: &'a [f64],
        sample_weight: &'a [f64],
        offset: &'a [f64],
        penalty_weight: f64,
        design_checked: &'a AtomicBool,
    ) -> Self {
        Self {
            design,
            response,
            sample_weight,
            unit_weights: sample_weight.iter().all(|&weight| weight == 1.0),
            offset,
            penalty_weight,
            design_check: DesignCheck {
                passed: design_checked,
                error: None,
            },
            evaluated: false,
            workspace: LogisticWorkspace::new(design.nrows()),
        }
    }

    #[expect(clippy::too_many_lines)]
    fn evaluate(&mut self, beta: &[f64]) -> Result<IterationQuantities> {
        let (design, response, sample_weight, offset) =
            (self.design, self.response, self.sample_weight, self.offset);
        let penalty_weight = self.penalty_weight;
        self.design_check.failure()?;
        let first_evaluation = !std::mem::replace(&mut self.evaluated, true);
        let design_check = &mut self.design_check;
        let (n, k) = design.shape();
        let LogisticWorkspace {
            matrix_scratch,
            eta,
            variance,
            y_minus_p,
            weights,
            h,
            residual,
            ..
        } = &mut self.workspace;

        // Each loop below reads every block of the design more than once. The later reads come from
        // cache.
        let block_rows = if cfg!(accelerate) {
            ACCELERATE_EVALUATION_BLOCK
        } else {
            EVALUATION_BLOCK
        };
        let blocks = (0..n)
            .step_by(block_rows)
            .map(|start| start..n.min(start + block_rows));

        // I = X.T @ diag(w) @ X, with w = sample_weight * p * (1 - p). An unpenalized fit also adds
        // `X.T @ (sample_weight * (y - p))` to the score in this loop.
        let unpenalized = penalty_weight == 0.0;
        let mut information = Mat::zeros(k, k);
        let mut log_likelihood = CompensatedSum::default();
        let mut modified_score = vec![0.0; k];
        for rows in blocks.clone() {
            let x_block = design.subrows(rows.start, rows.len());
            // `eta` is `X @ beta` without the offset. `expit_loglik` adds the offset.
            mul_vec(x_block, beta, &mut eta[rows.clone()]);
            log_likelihood.add(expit_loglik(
                &eta[rows.clone()],
                &offset[rows.clone()],
                &sample_weight[rows.clone()],
                self.unit_weights,
                &response[rows.clone()],
                &mut variance[rows.clone()],
                &mut y_minus_p[rows.clone()],
            ));
            let w = if self.unit_weights {
                &variance[rows.clone()]
            } else {
                let w = &mut weights[rows.clone()];
                let (sample_weight, variance) =
                    (&sample_weight[rows.clone()], &variance[rows.clone()]);
                dispatch!(Level::new(), _ => {
                    for ((w_i, &s_i), &v_i) in w.iter_mut().zip(sample_weight).zip(variance) {
                        *w_i = s_i * v_i;
                    }
                });
                &*w
            };
            add_weighted_gram(information.as_mut(), x_block, w, matrix_scratch);
            if unpenalized {
                let residual = &mut residual[rows.clone()];
                let (sample_weight, y_minus_p) = (&sample_weight[rows.clone()], &y_minus_p[rows]);
                for ((r_i, &s_i), &d_i) in residual.iter_mut().zip(sample_weight).zip(y_minus_p) {
                    *r_i = s_i * d_i;
                }
                add_transpose_mul_vec(x_block, residual, &mut modified_score);
            }
        }
        copy_lower_to_upper(&mut information);
        let log_likelihood = log_likelihood.total();
        let w: &[f64] = if self.unit_weights { variance } else { weights };

        let cholesky_tolerance = (n.max(k) as f64 * f64::EPSILON).sqrt();

        if unpenalized {
            // A rank-deficient design makes the information singular at every beta. Because a
            // full-rank design can also make the information numerically singular at extreme beta,
            // only the first evaluation tests the information. Rounding in the information can make
            // its Cholesky pivot ratio larger than the QR pivot ratio of the design check. The test
            // therefore uses twice the cutoff of the design check.
            if first_evaluation
                && pivoted_cholesky_pivot_ratio(&information) <= 2.0 * cholesky_tolerance
            {
                design_check.run(design, sample_weight)?;
            }
            return Ok(IterationQuantities {
                log_likelihood,
                modified_score,
                step_matrix: information,
                information: None,
                gram_diagonal: None,
            });
        }

        let penalty_scale = 2.0 * penalty_weight;
        let mut augmentation_limit = AUGMENTATION_THRESHOLD / penalty_scale;
        // The first two arms fill
        // `residual = sample_weight * (y - p) + 2*penalty_weight*h * (0.5 - p)`. They then add
        // `X.T @ residual` to the modified score.
        let (log_determinant, augmented_count) = match cholesky_factor_logdet(&information) {
            Ok(cholesky) if cholesky.minimum_relative_pivot > cholesky_tolerance => {
                let mut augmented_count = 0;
                for rows in blocks {
                    let x_block = design.subrows(rows.start, rows.len());
                    let h = &mut h[rows.clone()];
                    // The hat diagonal is
                    // `h_i = w_i * x_i @ inv_fisher @ x_i = w_i * |L^-1 x_i|^2`.
                    cholesky.quadratic_form_rows(x_block, h, matrix_scratch);
                    // Zero-weight rows have `h_i = 0`. `J_S` therefore never augments them.
                    augmented_count += scale_hat(
                        h,
                        &w[rows.clone()],
                        &sample_weight[rows.clone()],
                        augmentation_limit,
                    );
                    fill_residual(
                        &mut residual[rows.clone()],
                        &sample_weight[rows.clone()],
                        &response[rows.clone()],
                        &y_minus_p[rows.clone()],
                        h,
                        penalty_scale,
                    );
                    add_transpose_mul_vec(x_block, &residual[rows], &mut modified_score);
                }
                (cholesky.log_determinant, augmented_count)
            },
            Ok(_) | Err(Error::NotPositiveDefinite) => {
                design_check.run(design, sample_weight)?;
                let weighted_design = scale_rows_sqrt(design, w);
                // With a zero limit, `J_S` augments every row with `h_i > 0`. The step matrix of
                // the QR fallback is therefore `J`.
                augmentation_limit = 0.0;
                let log_determinant = pivoted_qr_hat_logdet(&weighted_design, h)?;
                fill_residual(
                    residual,
                    sample_weight,
                    response,
                    y_minus_p,
                    h,
                    penalty_scale,
                );
                add_transpose_mul_vec(design.as_ref(), residual, &mut modified_score);
                (log_determinant, n)
            },
            Err(error) => return Err(error),
        };
        let log_likelihood = log_likelihood + penalty_weight * log_determinant;

        // `J_S` adds `penalty_scale * h[i]` to `sample_weight[i]` on rows where
        // `h[i] > augmentation_limit * sample_weight[i]`. Up to rounding, building `J_S` from all
        // rows gives the same result as adding only the augmented rows to `information`. The
        // all-rows path runs when more than half the rows are augmented. Of the cutoffs timed on an
        // M5 and an i5-12500, this one came closest to always taking the faster path.
        let all_rows = augmented_count * 2 > n;
        if all_rows {
            dispatch!(Level::new(), _ => {
                // Borrowing slices keeps this loop vectorized.
                let (variance, h, weights) =
                    (variance.as_slice(), h.as_slice(), weights.as_mut_slice());
                for i in 0..n {
                    let augmentation = if h[i] > augmentation_limit * sample_weight[i] {
                        penalty_scale * h[i]
                    } else {
                        0.0
                    };
                    weights[i] = (sample_weight[i] + augmentation) * variance[i];
                }
            });
        }

        let (step_matrix, information) = if augmented_count == 0 {
            (information, None)
        } else if all_rows {
            let step_matrix = weighted_gram(design.as_ref(), weights, matrix_scratch);
            (step_matrix, Some(information))
        } else {
            let step_matrix =
                self.gathered_step_matrix(&information, augmentation_limit, penalty_scale);
            (step_matrix, Some(information))
        };

        Ok(IterationQuantities {
            log_likelihood,
            modified_score,
            step_matrix,
            information,
            gram_diagonal: None,
        })
    }

    /// Returns `J_S`, which is `information` plus the sum of
    /// `penalty_scale * h_i * p_i*(1 - p_i) * x_i x_i'` over the rows with
    /// `h_i > augmentation_limit * sample_weight_i`. The sum uses `h` and `variance` from the
    /// current evaluation.
    fn gathered_step_matrix(
        &mut self,
        information: &Mat<f64>,
        augmentation_limit: f64,
        penalty_scale: f64,
    ) -> Mat<f64> {
        let (design, sample_weight) = (self.design, self.sample_weight);
        let (n, k) = design.shape();
        let LogisticWorkspace {
            matrix_scratch,
            variance,
            h,
            augmented_rows,
            gathered_design,
            augmentation_weights,
            ..
        } = &mut self.workspace;

        augmented_rows.clear();
        augmented_rows.extend((0..n).filter(|&i| h[i] > augmentation_limit * sample_weight[i]));
        augmentation_weights.clear();
        augmentation_weights.extend(
            augmented_rows
                .iter()
                .map(|&i| penalty_scale * h[i] * variance[i]),
        );
        gathered_design.clear();
        for j in 0..k {
            let column = design.col(j).try_as_col_major().unwrap().as_slice();
            gathered_design.extend(augmented_rows.iter().map(|&i| column[i]));
        }
        let rows =
            MatRef::from_column_major_slice(gathered_design.as_slice(), augmented_rows.len(), k);
        let mut step_matrix = weighted_gram(rows, augmentation_weights, matrix_scratch);
        step_matrix += information.as_ref();
        step_matrix
    }
}

/// Rows per block in `LogisticObjective::evaluate` outside Accelerate builds. A block of 1024 rows
/// and 50 features is 400 KB, which fits in a 512 KB L2 cache.
const EVALUATION_BLOCK: usize = 1024;

/// Rows per block in Accelerate builds. BLAS and vForce calls cost more per call than the faer
/// loops, so these blocks are larger. On an M5, blocks of 8192 or 32768 rows made large fits slower
/// than blocks of 16384 rows.
const ACCELERATE_EVALUATION_BLOCK: usize = 16_384;

/// Fills `residual = sample_weight * (y - p) + penalty_scale * h * (0.5 - p)`. The loop computes
/// `0.5 - p` as `(0.5 - y) + (y - p)`.
fn fill_residual(
    residual: &mut [f64],
    sample_weight: &[f64],
    response: &[f64],
    y_minus_p: &[f64],
    h: &[f64],
    penalty_scale: f64,
) {
    dispatch!(Level::new(), _ => {
        for ((((residual, &s), &y), &y_minus_p), &h) in
            residual.iter_mut().zip(sample_weight).zip(response).zip(y_minus_p).zip(h)
        {
            *residual = s * y_minus_p + penalty_scale * h * (0.5 - y + y_minus_p);
        }
    });
}

/// Multiplies `h` by `weights` and returns the number of rows with
/// `h_i > augmentation_limit * sample_weight_i`.
fn scale_hat(
    h: &mut [f64],
    weights: &[f64],
    sample_weight: &[f64],
    augmentation_limit: f64,
) -> usize {
    dispatch!(Level::new(), _ => {
        let mut augmented_count = 0;
        for ((h_i, &w_i), &s_i) in h.iter_mut().zip(weights).zip(sample_weight) {
            *h_i *= w_i;
            augmented_count += usize::from(*h_i > augmentation_limit * s_i);
        }
        augmented_count
    })
}

/// `J_S` augments a row when its augmentation `2*penalty_weight*h_i` exceeds this multiple of the
/// row's `sample_weight`.
const AUGMENTATION_THRESHOLD: f64 = 0.1;

/// `LOGLIK_BLOCK` must be a multiple of every SIMD lane count. With any other value, the last
/// vector load of a block also reads rows of the next block. The kernels then count those rows
/// twice. `LOGLIK_BLOCK` must also be at most 1023. The unit-weight kernel multiplies the factors
/// `1 + z` of all rows in a block. Each factor is at most 2. A product of 1024 factors can
/// therefore overflow.
const LOGLIK_BLOCK: usize = 64;
const _: () = assert!(LOGLIK_BLOCK <= 1023);

/// Fills `variance = p * (1 - p)` and `y_minus_p = y - p` for `p = expit(eta + offset)` and
/// returns the log-likelihood.
///
/// With `z = exp(-|eta + offset|)`, the smaller of `p` and `1 - p` is `z / (1 + z)`. The larger is
/// `1 / (1 + z)`. The kernels compute both outputs from these two values. The outputs then keep
/// their relative accuracy when `p` is near 0 or 1. A `1 - p` computed from a rounded `p` near 1
/// has few correct digits.
///
/// `margin = (2y - 1) * (eta + offset)` is the log-odds of the observed response. When
/// `margin >= 0`, `|y - p|` is the smaller value. Otherwise `|y - p|` is the larger value. The
/// log-likelihood term of a row is `min(margin, 0) - log1p(z)`. For a row with `margin >= 0`, the
/// term is exactly `-log1p(z)`.
fn expit_loglik(
    eta: &[f64],
    offset: &[f64],
    sample_weight: &[f64],
    unit_weights: bool,
    response: &[f64],
    variance: &mut [f64],
    y_minus_p: &mut [f64],
) -> f64 {
    if unit_weights {
        return dispatch!(Level::new(), simd => {
            expit_loglik_unit_lanes(simd, eta, offset, response, variance, y_minus_p)
        });
    }

    #[cfg(accelerate)]
    return expit_loglik_accelerate(eta, offset, sample_weight, response, variance, y_minus_p);

    #[cfg(not(accelerate))]
    dispatch!(Level::new(), simd => {
        expit_loglik_lanes(simd, eta, offset, sample_weight, response, variance, y_minus_p)
    })
}

#[cfg(not(accelerate))]
#[inline(always)]
fn expit_loglik_lanes<S: Simd>(
    simd: S,
    eta: &[f64],
    offset: &[f64],
    sample_weight: &[f64],
    response: &[f64],
    variance: &mut [f64],
    y_minus_p: &mut [f64],
) -> f64 {
    use crate::simd::{exp_nonpositive, load, log1p_unit, store};
    const { assert!(LOGLIK_BLOCK.is_multiple_of(S::f64s::LEN)) };
    let n = eta.len();
    let one = S::f64s::splat(simd, 1.0);
    let mut log_likelihood = CompensatedSum::default();
    for start in (0..n).step_by(LOGLIK_BLOCK) {
        let end = (start + LOGLIK_BLOCK).min(n);
        let mut block = S::f64s::splat(simd, 0.0);
        for i in (start..end).step_by(S::f64s::LEN) {
            let len = S::f64s::LEN.min(end - i);
            let eta_i = load(simd, &eta[i..]) + load(simd, &offset[i..]);
            let z = exp_nonpositive(simd, -eta_i.abs());
            let larger = one / (z + 1.0);
            let smaller = z * larger;
            let sign = load(simd, &response[i..]).mul_add(2.0, -1.0);
            let margin = sign * eta_i;
            store::<S>(smaller * larger, &mut variance[i..i + len]);
            store::<S>(
                sign * margin.simd_ge(0.0).select(smaller, larger),
                &mut y_minus_p[i..i + len],
            );
            block += load(simd, &sample_weight[i..]) * (margin.min(0.0) - log1p_unit(simd, z));
        }
        log_likelihood.add(block.reduce_sum());
    }
    log_likelihood.total()
}

/// Computes `expit_loglik` for unit sample weights. The `log1p(z)` terms of a block of rows sum to
/// `log1p(q)` with `q = prod(1 + z) - 1`. Each block therefore needs one logarithm instead of one
/// per row. Each lane updates `q` as `q * (1 + z) + z`. Because every term of that update is
/// nonnegative, `q` keeps its relative accuracy when every `z` is small. A product of rounded
/// `1 + z` factors would not keep that accuracy, because rounding `1 + z` discards the low bits of
/// a small `z`.
#[inline(always)]
fn expit_loglik_unit_lanes<S: Simd>(
    simd: S,
    eta: &[f64],
    offset: &[f64],
    response: &[f64],
    variance: &mut [f64],
    y_minus_p: &mut [f64],
) -> f64 {
    use crate::simd::{exp_nonpositive, load, store};
    const { assert!(LOGLIK_BLOCK.is_multiple_of(S::f64s::LEN)) };
    let n = eta.len();
    let one = S::f64s::splat(simd, 1.0);
    let mut log_likelihood = CompensatedSum::default();
    for start in (0..n).step_by(LOGLIK_BLOCK) {
        let end = (start + LOGLIK_BLOCK).min(n);
        let mut block = S::f64s::splat(simd, 0.0);
        let mut product_minus_one = S::f64s::splat(simd, 0.0);
        for i in (start..end).step_by(S::f64s::LEN) {
            let len = S::f64s::LEN.min(end - i);
            let eta_i = load(simd, &eta[i..]) + load(simd, &offset[i..]);
            let mut z = exp_nonpositive(simd, -eta_i.abs());
            if len < S::f64s::LEN {
                // `load` fills the lanes past the end of the slice with zeros. Setting `z = 0` in
                // those lanes leaves `product_minus_one` unchanged. `store` writes only the first
                // `len` lanes of `variance` and `y_minus_p`.
                z = S::f64s::from_fn(
                    simd,
                    |lane| if lane < len { z.as_slice()[lane] } else { 0.0 },
                );
            }
            let one_plus_z = z + 1.0;
            let larger = one / one_plus_z;
            let smaller = z * larger;
            let sign = load(simd, &response[i..]).mul_add(2.0, -1.0);
            let margin = sign * eta_i;
            store::<S>(smaller * larger, &mut variance[i..i + len]);
            store::<S>(
                sign * margin.simd_ge(0.0).select(smaller, larger),
                &mut y_minus_p[i..i + len],
            );
            block += margin.min(0.0);
            product_minus_one = product_minus_one.mul_add(one_plus_z, z);
        }
        let product_minus_one = product_minus_one
            .as_slice()
            .iter()
            .fold(0.0, |q, &lane| q + lane + q * lane);
        log_likelihood.add(block.reduce_sum() - product_minus_one.ln_1p());
    }
    log_likelihood.total()
}

#[cfg(accelerate)]
fn expit_loglik_accelerate(
    eta: &[f64],
    offset: &[f64],
    sample_weight: &[f64],
    response: &[f64],
    variance: &mut [f64],
    y_minus_p: &mut [f64],
) -> f64 {
    const CHUNK: usize = 8192; // selected after measuring on M5 and M2
    let n = eta.len();
    let mut log_likelihood = CompensatedSum::default();
    let mut start = 0;
    while start < n {
        let end = n.min(start + CHUNK);
        // `exp_in_place` writes `z = exp(-|eta + offset|)` to `y_minus_p`. `log1p` writes
        // `log1p(z)` to `variance`. The loop below reads both values before it overwrites them.
        for i in start..end {
            y_minus_p[i] = -(eta[i] + offset[i]).abs();
        }
        crate::accelerate::exp_in_place(&mut y_minus_p[start..end]);
        crate::accelerate::log1p(&y_minus_p[start..end], &mut variance[start..end]);
        for block_start in (start..end).step_by(LOGLIK_BLOCK) {
            let mut block = 0.0;
            for i in block_start..(block_start + LOGLIK_BLOCK).min(end) {
                let z = y_minus_p[i];
                let larger = 1.0 / (1.0 + z);
                let smaller = z * larger;
                let sign = 2.0 * response[i] - 1.0;
                let margin = sign * (eta[i] + offset[i]);
                let log1p_z = variance[i];
                variance[i] = smaller * larger;
                y_minus_p[i] = sign * if margin >= 0.0 { smaller } else { larger };
                block += sample_weight[i] * (margin.min(0.0) - log1p_z);
            }
            log_likelihood.add(block);
        }
        start = end;
    }
    log_likelihood.total()
}

/// The logistic function `1/(1 + e^(-x))`.
fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn scale_rows_sqrt(design: &Mat<f64>, weights: &[f64]) -> Mat<f64> {
    Mat::from_fn(design.nrows(), design.ncols(), |i, j| {
        weights[i].sqrt() * design[(i, j)]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    static FEATURES: [f64; 4] = [0.5, 1.5, 2.5, 3.5];
    static RESPONSE: [f64; 4] = [0.0, 0.0, 1.0, 1.0];
    static PREDICTION_FEATURES: [f64; 12] =
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    static PREDICTION_RESPONSE: [f64; 12] =
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    static PREDICTION_OFFSET: [f64; 12] = [
        -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0,
    ];

    fn prediction_features() -> MatrixView<'static> {
        MatrixView::from_row_major(&PREDICTION_FEATURES, 12, 1).unwrap()
    }

    fn prediction_fit() -> LogisticFit {
        LogisticData::new(prediction_features())
            .unwrap()
            .model(&PREDICTION_RESPONSE)
            .offset(&PREDICTION_OFFSET)
            .fit()
            .unwrap()
    }

    #[test]
    fn prediction_uses_provided_offset() {
        let fit = prediction_fit();
        let prediction_offset = [0.5; 12];
        let scores = fit
            .decision_function(prediction_features(), Some(&prediction_offset))
            .unwrap();
        let probabilities = fit
            .predict_probability(prediction_features(), Some(&prediction_offset))
            .unwrap();

        for i in 0..PREDICTION_FEATURES.len() {
            let expected_score = PREDICTION_FEATURES[i] * fit.coefficients()[0]
                + fit.intercept()
                + prediction_offset[i];
            assert_eq!(scores[i], expected_score);
            assert_eq!(probabilities[i], expit(expected_score));
        }
    }

    #[test]
    fn expit_robustness() {
        assert_eq!(expit(0.0), 0.5);
        assert_eq!(expit(800.0), 1.0);
        assert_eq!(expit(-800.0), 0.0);
        assert!(expit(-745.0) > 0.0);
    }

    #[test]
    #[expect(clippy::float_cmp)]
    fn expit_loglik_matches_scalar_reference() {
        // In Accelerate builds, the 10,001 rows span more than one chunk of 8192 rows in
        // `expit_loglik_accelerate`. The row count is also not a multiple of any SIMD lane count.
        let n = 10_001;
        let eta: Vec<f64> = (0..n)
            .map(|i| match i {
                0 => 700.0,
                1 => -700.0,
                _ => ((7 * i + 3) % 41) as f64 - 20.0,
            })
            .collect();
        let offset: Vec<f64> = (0..n).map(|i| (i % 5) as f64 * 0.25 - 0.5).collect();
        let response: Vec<f64> = (0..n).map(|i| f64::from(i % 3 == 0)).collect();
        // Every row of `tail` has a fitted probability below 1e-12 and a response of 0. Every row
        // of `mirrored_tail` has a fitted probability above 1 - 1e-12 and a response of 1. In both,
        // the log-likelihood is a sum of `log1p(z)` terms with `z < 1e-12`.
        let tail: Vec<f64> = (0..n).map(|i| -30.0 - (i % 7) as f64 * 0.5).collect();
        let mirrored_tail: Vec<f64> = tail.iter().map(|eta| -eta).collect();
        let (no_events, all_events) = (vec![0.0; n], vec![1.0; n]);
        let weighted: Vec<f64> = (0..n).map(|i| (i % 7) as f64 * 0.5).collect();
        let unit = vec![1.0; n];
        for (eta, response, sample_weight, unit_weights) in [
            (&eta, &response, &weighted, false),
            (&eta, &response, &unit, true),
            (&tail, &no_events, &weighted, false),
            (&tail, &no_events, &unit, true),
            (&mirrored_tail, &all_events, &weighted, false),
            (&mirrored_tail, &all_events, &unit, true),
        ] {
            let mut variance = vec![0.0; n];
            let mut y_minus_p = vec![0.0; n];
            let log_likelihood = expit_loglik(
                eta,
                &offset,
                sample_weight,
                unit_weights,
                response,
                &mut variance,
                &mut y_minus_p,
            );
            let mut expected_loglik = 0.0;
            for i in 0..n {
                let eta_i = eta[i] + offset[i];
                let z = (-eta_i.abs()).exp();
                let (p_i, one_minus_p) = if eta_i >= 0.0 {
                    (1.0 / (1.0 + z), z / (1.0 + z))
                } else {
                    (z / (1.0 + z), 1.0 / (1.0 + z))
                };
                let variance_i = p_i * one_minus_p;
                let y_minus_p_i = if response[i] == 1.0 {
                    one_minus_p
                } else {
                    -p_i
                };
                expected_loglik +=
                    sample_weight[i] * (response[i] * eta_i - eta_i.max(0.0) - z.ln_1p());
                assert!(
                    (variance[i] - variance_i).abs() <= 1e-14 * variance_i,
                    "variance {i}: {} vs {variance_i}",
                    variance[i]
                );
                assert!(
                    (y_minus_p[i] - y_minus_p_i).abs() <= 1e-14 * y_minus_p_i.abs(),
                    "y - p {i}: {} vs {y_minus_p_i}",
                    y_minus_p[i]
                );
            }
            assert!(
                (log_likelihood - expected_loglik).abs() <= 1e-12 * expected_loglik.abs(),
                "unit_weights={unit_weights}: {log_likelihood} vs {expected_loglik}"
            );
        }
    }

    fn design() -> Mat<f64> {
        Mat::from_fn(4, 2, |i, j| if j == 0 { FEATURES[i] } else { 1.0 })
    }

    fn quantities(offset: &[f64], penalty_weight: f64, beta: &[f64]) -> IterationQuantities {
        let design = design();
        LogisticObjective::new(
            &design,
            &RESPONSE,
            &[1.0; 4],
            offset,
            penalty_weight,
            &AtomicBool::new(false),
        )
        .evaluate(beta)
        .unwrap()
    }

    fn assert_mat_close(observed: &Mat<f64>, expected: &Mat<f64>) {
        for i in 0..expected.nrows() {
            for j in 0..expected.ncols() {
                let (obs, exp) = (observed[(i, j)], expected[(i, j)]);
                assert!(
                    (obs - exp).abs() < 1e-12,
                    "({i},{j}): got {obs}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn quantities_at_beta_zero_penalized() {
        let quantities = quantities(&[0.0; 4], 0.5, &[0.0, 0.0]);
        // At beta = 0, every fitted probability is 1/2. The penalty term `2 * tau * h * (0.5 - p)`
        // is then zero. The modified score therefore equals the unpenalized score `X.T @ (y - p)`.
        assert!((quantities.modified_score[0] - 2.0).abs() < 1e-12);
        assert!(quantities.modified_score[1].abs() < 1e-12);
        let expected_loglik = -4.0 * 2.0_f64.ln() + 0.5 * 1.25_f64.ln();
        assert!((quantities.log_likelihood - expected_loglik).abs() < 1e-12);
        let expected = faer::mat![[8.075, 3.0], [3.0, 1.5]];
        assert_mat_close(&quantities.step_matrix, &expected);
    }

    #[test]
    fn constant_offset_matches_intercept_shift() {
        // offset only affects eta, constant offset 0.7 == intercept beta of 0.7
        let shifted_beta = quantities(&[0.0; 4], 0.5, &[0.3, 0.7]);
        let shifted_offset = quantities(&[0.7; 4], 0.5, &[0.3, 0.0]);
        assert!((shifted_beta.log_likelihood - shifted_offset.log_likelihood).abs() < 1e-12);
        for j in 0..2 {
            assert!(
                (shifted_beta.modified_score[j] - shifted_offset.modified_score[j]).abs() < 1e-12
            );
        }
        assert_mat_close(&shifted_beta.step_matrix, &shifted_offset.step_matrix);
    }

    #[test]
    fn weighted_qr_fallback_with_reused_workspace() {
        // With nine columns, Accelerate builds use the Accelerate kernels. Row i is in group
        // i % 10. Group 0 has only column 0. Group 1 has only column 1. Group 2 has both columns.
        // Each group g from 3 to 9 has only column g - 1. Each group has five responses of 1 and
        // five of 0.
        let (n, k) = (100, 9);
        let design = Mat::from_fn(n, k, |i, j| {
            let group = i % 10;
            f64::from(match group {
                0 | 1 => j == group,
                2 => j < 2,
                _ => j == group - 1,
            })
        });
        let response: Vec<_> = (0..n).map(|i| f64::from(i / 10 % 2 == 0)).collect();
        let (sample_weight, offset) = (vec![1.0; n], vec![0.0; n]);
        let checked = AtomicBool::new(false);
        let mut reused =
            LogisticObjective::new(&design, &response, &sample_weight, &offset, 1.0, &checked);

        // At beta_0 = 36 and beta_1 = -36, the rows in groups 0 and 1 have weight 2.3e-16. The rows
        // in group 2 have weight 1/4. Columns 0 and 1 of the weighted design are then nearly equal.
        // The evaluation therefore falls back to QR. The last evaluation, at beta = 0, uses
        // Cholesky again.
        for saturation in [0.0, 36.0, 0.0] {
            let mut beta = vec![0.0; k];
            (beta[0], beta[1]) = (saturation, -saturation);
            let quantities = reused.evaluate(&beta).unwrap();
            let fresh_checked = AtomicBool::new(false);
            let expected = LogisticObjective::new(
                &design,
                &response,
                &sample_weight,
                &offset,
                1.0,
                &fresh_checked,
            )
            .evaluate(&beta)
            .unwrap();
            assert!(
                (quantities.log_likelihood - expected.log_likelihood).abs()
                    < 1e-12 * expected.log_likelihood.abs()
            );
            for j in 0..k {
                let score = (quantities.modified_score[j], expected.modified_score[j]);
                assert!((score.0 - score.1).abs() < 1e-12, "score {j}: {score:?}");
                for i in 0..k {
                    let entry = (quantities.step_matrix[(i, j)], expected.step_matrix[(i, j)]);
                    assert!((entry.0 - entry.1).abs() < 1e-12, "({i},{j}): {entry:?}");
                }
            }
            if saturation > 0.0 {
                assert!(
                    checked.load(Ordering::Relaxed),
                    "the evaluation at beta_0 = 36 did not use the QR fallback"
                );
                // The weight of groups 0 and 1 is `z / (1 + z)^2` with `z = exp(-36)`.
                let z = (-saturation).exp();
                let small = z / ((1.0 + z) * (1.0 + z));
                for (i, &h) in reused.workspace.h.iter().enumerate() {
                    let expected = match i % 10 {
                        0 | 1 => (small + 0.25) / (10.0 * (small + 0.5)),
                        2 => 0.5 / (10.0 * (small + 0.5)),
                        _ => 0.1,
                    };
                    assert!((h - expected).abs() < 1e-6, "row {i}: {h} vs {expected}");
                }
                let log_determinant = (100.0 * small * (small + 0.5)).ln() + 7.0 * 2.5_f64.ln();
                let expected = 10.0 * small.ln() + 80.0 * 0.5_f64.ln() + log_determinant;
                assert!((quantities.log_likelihood - expected).abs() < 1e-6);
            }
        }
    }
}
