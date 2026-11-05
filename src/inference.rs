mod distributions;
mod lrt;
mod profile_ci;
mod wald;

use faer::Mat;

pub use lrt::LrtResult;
pub use profile_ci::{ProfileBound, ProfileOptions, Tail};

use crate::Result;
use crate::solver::{FitStatus, NewtonResult};
use wald::{WaldSummary, wald_confidence_intervals, wald_log10_p_values, wald_summary};

/// Status of a penalized likelihood ratio test or a profile likelihood bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferenceStatus {
    /// Converged with finite log-likelihoods.
    Valid,
    /// The full fit did not converge.
    FullFitNonconvergence,
    /// The constrained LRT fit did not converge.
    ConstrainedFitNonconvergence,
    /// A log-likelihood was non-finite. A profile bound also has this status when the score or the
    /// step matrix at an iterate is non-finite.
    NonFiniteLikelihood,
    /// The profile iteration did not converge within the iteration limit.
    ProfileBoundNonconvergence,
    /// The profile iteration ended on the other side of the estimate.
    ProfileBoundWrongSide,
}

impl InferenceStatus {
    #[must_use]
    pub const fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }
}

/// A penalized estimate and its Wald inference. `LogisticFit` and `CoxFit` each store one. The
/// methods for likelihood ratio tests and profile bounds call a closure to evaluate the model.
#[derive(Debug)]
pub(crate) struct Estimate {
    pub(crate) beta: Vec<f64>,
    pub(crate) log_likelihood: f64,
    /// The Fisher information at `beta`.
    information: Mat<f64>,
    /// The inverse of `information` in row-major order.
    pub(crate) covariance: Option<Vec<f64>>,
    pub(crate) standard_errors: Vec<f64>,
    pub(crate) p_values: Vec<f64>,
    pub(crate) iterations: usize,
    pub(crate) status: FitStatus,
}

impl Estimate {
    pub(crate) fn new(result: NewtonResult) -> Self {
        let log_likelihood = result.quantities.log_likelihood;
        // Wald inference uses the Fisher information, not the step matrix.
        let information = result.quantities.into_information();
        let WaldSummary {
            covariance,
            standard_errors,
            p_values,
        } = wald_summary(
            &result.beta,
            result.status.is_converged().then_some(&information),
        );
        Self {
            beta: result.beta,
            log_likelihood,
            information,
            covariance,
            standard_errors,
            p_values,
            iterations: result.iterations,
            status: result.status,
        }
    }

    pub(crate) fn log10_p_values(&self) -> Vec<f64> {
        wald_log10_p_values(&self.beta, &self.standard_errors)
    }

    pub(crate) fn wald_confidence_intervals(&self, alpha: f64) -> Result<Vec<[f64; 2]>> {
        wald_confidence_intervals(&self.beta, &self.standard_errors, alpha)
    }
}
