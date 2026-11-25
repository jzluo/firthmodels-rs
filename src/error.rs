/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned on invalid input or numerical failure.
///
/// Solver non-convergence is not an error; it is reported through [`FitStatus`](crate::FitStatus)
/// and [`InferenceStatus`](crate::InferenceStatus).
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Input dimensions are zero, mismatched, or too small for the model.
    #[error("invalid input dimensions: {0}")]
    InvalidDimensions(String),

    /// An input value or option is out of range, non-finite, or invalid for the model.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Usually indicates collinear features.
    #[error("matrix is rank deficient: rank {rank}, expected {required}")]
    RankDeficient { rank: usize, required: usize },

    /// The design matrix is full rank but too ill-conditioned for the normal-equations solver.
    #[error("design matrix is too ill-conditioned for the normal-equations solver")]
    IllConditionedDesign,

    /// Comes from a failed Cholesky factorization.
    #[error("matrix is not positive definite")]
    NotPositiveDefinite,

    /// A linear system could not be solved because its matrix is singular.
    #[error("linear system is singular")]
    SingularSystem,

    /// A computation produced `NaN` or an infinite value.
    #[error("a computation produced a non-finite value")]
    NonFinite,
}
