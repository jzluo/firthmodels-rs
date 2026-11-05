//! Firth-penalized logistic and Cox regression.
//!
//! On Apple silicon, the crate calls Accelerate functions that are available only in macOS 13.3 and
//! later. By default, rustc records macOS 11.0 as the minimum version of a program. Set
//! `MACOSX_DEPLOYMENT_TARGET` to 13.3 or later when you build a program to run on other Macs.

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

// Callers run fits in parallel threads. Prepared data and fits must therefore stay `Send` and
// `Sync`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LogisticData>();
    assert_send_sync::<LogisticFit>();
    assert_send_sync::<CoxData>();
    assert_send_sync::<CoxFit>();
};
