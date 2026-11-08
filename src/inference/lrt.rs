use core::f64::consts::LN_10;

use faer::{Mat, MatRef};

use crate::inference::distributions::{chi_squared_log_sf, chi_squared_sf};
use crate::inference::{Estimate, InferenceStatus};
use crate::linalg::spd_solve;
use crate::solver::{IterationQuantities, NewtonOptions, newton_raphson};
use crate::validation::validate_indices;
use crate::{FitStatus, Result};

/// The result of a penalized likelihood ratio test that a set of coefficients are all zero.
///
/// When `status` is not `Valid`, `chi_squared` and `p_value` are NaN. Check `status`
/// instead of testing for NaNs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrtResult {
    chi_squared: f64,
    p_value: f64,
    degrees_of_freedom: usize,
    status: InferenceStatus,
    iterations: usize,
    // The status of the full or constrained fit when that fit did not converge.
    fit_status: Option<FitStatus>,
}

impl LrtResult {
    const fn invalid(
        degrees_of_freedom: usize,
        status: InferenceStatus,
        fit_status: Option<FitStatus>,
        iterations: usize,
    ) -> Self {
        Self {
            chi_squared: f64::NAN,
            p_value: f64::NAN,
            degrees_of_freedom,
            status,
            iterations,
            fit_status,
        }
    }

    fn from_log_likelihoods(
        degrees_of_freedom: usize,
        log_likelihood_full: f64,
        log_likelihood_constrained: f64,
        status: FitStatus,
        iterations: usize,
    ) -> Self {
        if !status.is_converged() {
            return Self::invalid(
                degrees_of_freedom,
                InferenceStatus::ConstrainedFitNonconvergence,
                Some(status),
                iterations,
            );
        }
        if !log_likelihood_full.is_finite() || !log_likelihood_constrained.is_finite() {
            return Self::invalid(
                degrees_of_freedom,
                InferenceStatus::NonFiniteLikelihood,
                None,
                iterations,
            );
        }
        let chi_squared = (2.0 * (log_likelihood_full - log_likelihood_constrained)).max(0.0);
        Self {
            chi_squared,
            p_value: chi_squared_sf(chi_squared, degrees_of_freedom),
            degrees_of_freedom,
            status: InferenceStatus::Valid,
            iterations,
            fit_status: None,
        }
    }

    /// `chi_squared = max(0, 2 * (loglik_full - loglik_constrained))`
    #[must_use]
    pub const fn chi_squared(&self) -> f64 {
        self.chi_squared
    }

    /// Probability that a chi-squared with `degrees_of_freedom` degrees of freedom exceeds the
    /// test statistic.
    #[must_use]
    pub const fn p_value(&self) -> f64 {
        self.p_value
    }

    /// Base-10 log of `p_value`. The log is finite even where `p_value` underflows to zero.
    #[must_use]
    pub fn log10_p_value(&self) -> f64 {
        chi_squared_log_sf(self.chi_squared, self.degrees_of_freedom) / LN_10
    }

    /// The number of coefficients the test fixes at zero.
    #[must_use]
    pub const fn degrees_of_freedom(&self) -> usize {
        self.degrees_of_freedom
    }

    #[must_use]
    pub const fn status(&self) -> InferenceStatus {
        self.status
    }

    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.status.is_valid()
    }

    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    #[must_use]
    pub const fn fit_status(&self) -> Option<FitStatus> {
        self.fit_status
    }
}

impl Estimate {
    /// Penalized likelihood ratio test that the coefficients at `indices` are all zero.
    ///
    /// The closure `compute_quantities` evaluates the model at a full-length coefficient vector
    /// that is zero at `indices`. When a fit does not converge, the method returns `Ok` with a
    /// nonconvergence status.
    pub(crate) fn likelihood_ratio_test(
        &self,
        indices: &[usize],
        warm_start: bool,
        solver: NewtonOptions,
        compute_quantities: impl FnMut(&[f64]) -> Result<IterationQuantities>,
    ) -> Result<LrtResult> {
        validate_indices(indices, self.beta.len())?;
        let degrees_of_freedom = indices.len();
        if !self.status.is_converged() {
            return Ok(LrtResult::invalid(
                degrees_of_freedom,
                InferenceStatus::FullFitNonconvergence,
                Some(self.status),
                0,
            ));
        }
        if !self.log_likelihood.is_finite() {
            return Ok(LrtResult::invalid(
                degrees_of_freedom,
                InferenceStatus::NonFiniteLikelihood,
                None,
                0,
            ));
        }
        let beta_start = if warm_start {
            constrained_warm_start(&self.beta, self.covariance.as_deref(), indices)
        } else {
            None
        };
        constrained_lrt(
            indices,
            self.beta.len(),
            self.log_likelihood,
            beta_start,
            solver,
            compute_quantities,
        )
    }
}

/// Computes warm-start values for the free coefficients of a constrained fit that fixes the
/// coefficients at `fixed` to zero. The values are `beta_F - C_FS C_SS^-1 beta_S`, the conditional
/// mean of the free coefficients under the full fit's covariance `C`. The values maximize the quadratic approximation
/// around `beta` with curvature `C^-1` when the coefficients at `fixed` are zero.
fn constrained_warm_start(
    beta: &[f64],
    covariance: Option<&[f64]>,
    fixed: &[usize],
) -> Option<Vec<f64>> {
    let covariance = MatRef::from_row_major_slice(covariance?, beta.len(), beta.len());
    let fixed_covariance = Mat::from_fn(fixed.len(), fixed.len(), |i, j| {
        covariance[(fixed[i], fixed[j])]
    });
    let fixed_beta: Vec<f64> = fixed.iter().map(|&j| beta[j]).collect();
    let weights = spd_solve(&fixed_covariance, &fixed_beta).ok()?;
    let beta_free: Vec<f64> = (0..beta.len())
        .filter(|j| !fixed.contains(j))
        .map(|j| {
            let shift: f64 = fixed
                .iter()
                .zip(&weights)
                .map(|(&s, weight)| covariance[(j, s)] * weight)
                .sum();
            beta[j] - shift
        })
        .collect();
    // A nearly singular `fixed_covariance` can make the weights overflow to infinity.
    beta_free
        .iter()
        .all(|value| value.is_finite())
        .then_some(beta_free)
}

fn constrained_lrt(
    fixed: &[usize],
    n_parameters: usize,
    log_likelihood_full: f64,
    beta_initial_free: Option<Vec<f64>>,
    options: NewtonOptions,
    mut compute_quantities: impl FnMut(&[f64]) -> Result<IterationQuantities>,
) -> Result<LrtResult> {
    let degrees_of_freedom = fixed.len();
    let free_indices: Vec<usize> = (0..n_parameters).filter(|j| !fixed.contains(j)).collect();
    let n_free = free_indices.len();
    let mut beta_full = vec![0.0; n_parameters];

    if n_free == 0 {
        let quantities = compute_quantities(&beta_full)?;
        return Ok(LrtResult::from_log_likelihoods(
            degrees_of_freedom,
            log_likelihood_full,
            quantities.log_likelihood,
            FitStatus::Converged,
            0,
        ));
    }

    let beta_start = beta_initial_free.unwrap_or_else(|| vec![0.0; n_free]);
    let constrained = newton_raphson(options, beta_start, |beta_free| {
        for (&idx, &value) in free_indices.iter().zip(beta_free) {
            beta_full[idx] = value;
        }
        let quantities = compute_quantities(&beta_full)?;
        Ok(IterationQuantities {
            log_likelihood: quantities.log_likelihood,
            modified_score: free_indices
                .iter()
                .map(|&j| quantities.modified_score[j])
                .collect(),
            step_matrix: Mat::from_fn(n_free, n_free, |i, j| {
                quantities.step_matrix[(free_indices[i], free_indices[j])]
            }),
            information: None,
            gram_diagonal: quantities
                .gram_diagonal
                .as_ref()
                .map(|diagonal| free_indices.iter().map(|&j| diagonal[j]).collect()),
        })
    })?;

    Ok(LrtResult::from_log_likelihoods(
        degrees_of_freedom,
        log_likelihood_full,
        constrained.quantities.log_likelihood,
        constrained.status,
        constrained.iterations,
    ))
}
