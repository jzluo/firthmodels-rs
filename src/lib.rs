//! Firth-penalized logistic and Cox proportional hazards regression.
//!
//! # Overview
//!
//! - **Penalized and unpenalized fits** for logistic regression and Cox proportional hazards
//!   regression. Both models default to
//!   Firth's Jeffreys-prior penalty (`penalty_weight = 0.5`) and support standard maximum
//!   likelihood (`penalty_weight = 0.0`) via [`LogisticOptions`] and [`CoxOptions`].
//! - **Wald, likelihood ratio, and profile-likelihood inference**: Wald standard errors,
//!   p-values, and intervals on [`LogisticFit`] and [`CoxFit`], joint penalized likelihood ratio
//!   tests ([`LrtResult`]), and Venzon-Moolgavkar profile-likelihood confidence bounds
//!   ([`ProfileBound`], [`ProfileOptions`]).
//! - **Reusable prepared data**: [`LogisticData`] and [`CoxData`] validate and store inputs once
//!   for reuse across many fits (such as weighted bootstrap resamples or multi-outcome logistic
//!   scans).
//! - **Single-threaded, `Send + Sync` execution**: callers can parallelize across fits or
//!   inference queries using their own thread pool.
//!
//! # Workflow
//!
//! Wrap a row-major or column-major slice in a [`MatrixView`], prepare [`LogisticData`] or
//! [`CoxData`], build a model with [`LogisticData::model`] or [`CoxData::model`], and call `fit`
//! to obtain a [`LogisticFit`] or [`CoxFit`].
//!
//! Solver non-convergence is reported through [`FitStatus`] on the returned fit rather than as an
//! [`Error`], and inference non-convergence is reported through [`InferenceStatus`].
//!
//! # Examples
//!
//! ## Logistic regression
//!
//! ```rust
//! use firthmodels::{LogisticData, MatrixView, ProfileOptions, Result};
//!
//! fn main() -> Result<()> {
//!     // Every row with x = 1 has y = 1, so the unpenalized estimate for x is infinite.
//!     let x = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//!     let y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//!     let features = MatrixView::from_row_major(&x, x.len(), 1)?;
//!
//!     let fit = LogisticData::new(features)?.model(&y).fit()?;
//!     assert!(fit.status().is_converged());
//!
//!     let lrt = fit.likelihood_ratio_test(&[0], true)?;
//!     assert!(lrt.is_valid());
//!
//!     let [lower, upper] = fit.profile_confidence_interval(0, 0.05, ProfileOptions::default())?;
//!     assert!(lower.is_valid() && upper.is_valid());
//!     Ok(())
//! }
//! ```
//!
//! ## Cox regression
//!
//! ```rust
//! use firthmodels::{CoxData, MatrixView, ProfileOptions, Result};
//!
//! fn main() -> Result<()> {
//!     // The only samples with x_0 = 1 also have the two earliest events, so the
//!     // unpenalized estimate for x_0 is infinite.
//!     let x = [
//!         1.0, 0.5,
//!         1.0, 1.2,
//!         0.0, 0.8,
//!         0.0, 1.5,
//!         0.0, 0.3,
//!         0.0, 1.1,
//!     ];
//!     let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//!     let event = [true, true, true, false, true, false];
//!     let features = MatrixView::from_row_major(&x, 6, 2)?;
//!
//!     let fit = CoxData::new(features, &time, &event)?.model().fit()?;
//!     assert!(fit.status().is_converged());
//!
//!     // Joint likelihood ratio test that both coefficients are zero (2 df).
//!     let lrt = fit.likelihood_ratio_test(&[0, 1], true)?;
//!     assert!(lrt.is_valid());
//!
//!     let [lower, upper] = fit.profile_confidence_interval(0, 0.05, ProfileOptions::default())?;
//!     assert!(lower.is_valid() && upper.is_valid());
//!
//!     let c_index = fit.concordance(features, &time, &event, None)?;
//!     assert!(c_index > 0.5);
//!     Ok(())
//! }
//! ```
//!
//! # Platform note (macOS)
//!
//! On Apple silicon, this crate calls Apple Accelerate functions that require macOS 13.3 or
//! later. Because `rustc` defaults to a macOS 11.0 deployment target, set
//! `MACOSX_DEPLOYMENT_TARGET=13.3` (or newer) when building binaries for distribution to other
//! Macs.
//!
//! # References
//!
//! - Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
//! - Heinze, G., & Schemper, M. (2001). A solution to the problem of monotone likelihood in Cox
//!   regression. *Biometrics*, 57(1), 114–119.
//! - Heinze, G., & Schemper, M. (2002). A solution to the problem of separation in logistic
//!   regression. *Statistics in Medicine*, 21(16), 2409–2419.
//! - Kosmidis, I., & Firth, D. (2021). Jeffreys-prior penalty, finiteness and shrinkage in
//!   binomial-response generalized linear models. *Biometrika*, 108(1), 71–82.
//! - Venzon, D. J., & Moolgavkar, S. H. (1988). A method for computing profile-likelihood-based
//!   confidence intervals. *Applied Statistics*, 37(1), 87–94.

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[cfg(accelerate)]
mod accelerate;
mod cox;
mod error;
mod inference;
mod linalg;
mod logistic;
mod matrix;
mod simd;
mod solver;
mod sum;
mod validation;

pub use cox::{CoxData, CoxFit, CoxModel, CoxOptions, concordance_index};
pub use error::{Error, Result};
pub use inference::{InferenceStatus, LrtResult, ProfileBound, ProfileOptions, Tail};
pub use logistic::{LogisticData, LogisticFit, LogisticModel, LogisticOptions};
pub use matrix::MatrixView;
pub use solver::{FitStatus, NewtonOptions};

// Callers run fits across threads, so prepared data and fits must stay `Send + Sync`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LogisticData>();
    assert_send_sync::<LogisticFit>();
    assert_send_sync::<CoxData>();
    assert_send_sync::<CoxFit>();
};
