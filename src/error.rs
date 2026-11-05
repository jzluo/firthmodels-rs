pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid input dimensions: {0}")]
    InvalidDimensions(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Usually indicates collinear features.
    #[error("matrix is rank deficient: rank {rank}, expected {required}")]
    RankDeficient { rank: usize, required: usize },

    #[error("design matrix is too ill-conditioned for the normal-equations solver")]
    IllConditionedDesign,

    /// Comes from a failed Cholesky factorization.
    #[error("matrix is not positive definite")]
    NotPositiveDefinite,

    #[error("linear system is singular")]
    SingularSystem,

    #[error("a computation produced a non-finite value")]
    NonFinite,
}
