//! Shared input validation.

use crate::{Error, MatrixView, Result};

/// Rejects a penalty weight that is negative or non-finite.
pub(crate) fn validate_penalty_weight(penalty_weight: f64) -> Result<()> {
    if !penalty_weight.is_finite() || penalty_weight < 0.0 {
        return Err(Error::InvalidInput(
            "penalty_weight must be finite and nonnegative".into(),
        ));
    }
    Ok(())
}

/// Rejects a feature matrix that contains NaN or infinite values.
pub(crate) fn validate_finite_features(features: MatrixView) -> Result<()> {
    if !features.all_finite() {
        return Err(Error::InvalidInput("features must be finite".into()));
    }
    Ok(())
}

/// Rejects a feature matrix whose column count differs from `n_features`.
pub(crate) fn validate_feature_count(features: MatrixView, n_features: usize) -> Result<()> {
    if features.ncols() != n_features {
        return Err(Error::InvalidDimensions(format!(
            "features have {} columns, expected {n_features}",
            features.ncols()
        )));
    }
    Ok(())
}

/// Rejects a significance level outside the open interval (0, 1).
pub(crate) fn validate_alpha(alpha: f64) -> Result<()> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidInput(format!(
            "alpha must be in the open interval (0, 1), got {alpha}"
        )));
    }
    Ok(())
}

/// Rejects a coefficient index past the last parameter.
pub(crate) fn validate_index(index: usize, n_parameters: usize) -> Result<()> {
    if index >= n_parameters {
        return Err(Error::InvalidInput(format!(
            "index {index} is out of bounds for {n_parameters} parameters"
        )));
    }
    Ok(())
}

/// Rejects an empty set of coefficient indices, an index past the last parameter, or a repeated
/// index.
pub(crate) fn validate_indices(indices: &[usize], n_parameters: usize) -> Result<()> {
    if indices.is_empty() {
        return Err(Error::InvalidInput("indices must not be empty".into()));
    }
    let mut seen = vec![false; n_parameters];
    for &index in indices {
        validate_index(index, n_parameters)?;
        if std::mem::replace(&mut seen[index], true) {
            return Err(Error::InvalidInput(format!(
                "index {index} appears more than once"
            )));
        }
    }
    Ok(())
}

/// Rejects survival times that are non-finite or negative.
pub(crate) fn validate_times(time: &[f64]) -> Result<()> {
    if time.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidInput("time must be finite".into()));
    }
    if time.iter().any(|&value| value < 0.0) {
        return Err(Error::InvalidInput("time must be nonnegative".into()));
    }
    Ok(())
}

/// Rejects values whose length differs from `expected` or that are not all finite.
pub(crate) fn validate_vector(values: &[f64], expected: usize, name: &str) -> Result<()> {
    if values.len() != expected {
        return Err(Error::InvalidDimensions(format!(
            "{name} has length {}, expected {expected}",
            values.len()
        )));
    }
    if values.iter().any(|&value| !value.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "{name} values must all be finite",
        )));
    }
    Ok(())
}

/// Returns `expected` copies of `default` if `values` is `None`. Otherwise, returns a copy of
/// `values` if it passes `validate_vector`.
pub(crate) fn validate_optional_vector(
    values: Option<&[f64]>,
    expected: usize,
    default: f64,
    name: &str,
) -> Result<Vec<f64>> {
    let Some(values) = values else {
        return Ok(vec![default; expected]);
    };
    validate_vector(values, expected, name)?;
    Ok(values.to_vec())
}

/// Rejects a negative case weight.
pub(crate) fn validate_nonnegative_weights(sample_weight: &[f64]) -> Result<()> {
    if sample_weight.iter().any(|&weight| weight < 0.0) {
        return Err(Error::InvalidInput(
            "sample_weight must be nonnegative".into(),
        ));
    }
    Ok(())
}
