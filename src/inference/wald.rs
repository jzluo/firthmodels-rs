use core::f64::consts::{LN_2, LN_10};

use faer::{Mat, MatMut};

use crate::Result;
use crate::inference::distributions::{normal_log_sf, normal_quantile, normal_sf};
use crate::linalg::cholesky_inverse_logdet;
use crate::validation::validate_alpha;

pub(super) struct WaldSummary {
    pub covariance: Option<Vec<f64>>,
    pub standard_errors: Vec<f64>,
    pub p_values: Vec<f64>,
}

/// Computes the covariance, standard errors, and Wald p-values from the Fisher information at
/// the fitted `beta`.
///
/// If `information` is `None` or not positive definite, the covariance is `None`. The standard
/// errors and p-values are then NaN.
pub(super) fn wald_summary(beta: &[f64], information: Option<&Mat<f64>>) -> WaldSummary {
    let n_parameters = beta.len();
    let covariance = information
        .and_then(|information| cholesky_inverse_logdet(information).ok())
        .map(|cholesky| {
            let mut covariance = vec![0.0; n_parameters * n_parameters];
            MatMut::from_row_major_slice_mut(&mut covariance, n_parameters, n_parameters)
                .copy_from(&cholesky.inverse);
            covariance
        });

    let standard_errors = covariance.as_ref().map_or_else(
        || vec![f64::NAN; n_parameters],
        |cov| {
            (0..n_parameters)
                .map(|i| cov[i * n_parameters + i].sqrt())
                .collect()
        },
    );

    let p_values = beta
        .iter()
        .zip(&standard_errors)
        .map(|(&beta, &se)| 2.0 * normal_sf((beta / se).abs()))
        .collect();

    WaldSummary {
        covariance,
        standard_errors,
        p_values,
    }
}

/// Base-10 logs of the two-sided Wald p-values. The logs are finite even where the p-values
/// underflow to zero.
pub(super) fn wald_log10_p_values(beta: &[f64], standard_errors: &[f64]) -> Vec<f64> {
    beta.iter()
        .zip(standard_errors)
        .map(|(&beta, &se)| (LN_2 + normal_log_sf((beta / se).abs())) / LN_10)
        .collect()
}

/// Computes `beta +- z * se` for each coefficient, where `z` is the standard normal
/// quantile for a two-sided confidence level of `1 - alpha`
///
/// # Errors
///
/// Returns `InvalidInput` if `alpha` is outside the open interval (0, 1).
pub(super) fn wald_confidence_intervals(
    beta: &[f64],
    standard_errors: &[f64],
    alpha: f64,
) -> Result<Vec<[f64; 2]>> {
    validate_alpha(alpha)?;
    let z = -normal_quantile(alpha / 2.0);
    Ok(beta
        .iter()
        .zip(standard_errors)
        .map(|(&beta, &se)| [beta - z * se, beta + z * se])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singular_information_gives_nan_errors_not_failure() {
        let fisher = Mat::from_fn(2, 2, |_, _| 1.0);
        let summary = wald_summary(&[0.5, 0.5], Some(&fisher));
        assert!(summary.covariance.is_none());
        assert!(summary.standard_errors.iter().all(|se| se.is_nan()));
        assert!(summary.p_values.iter().all(|p| p.is_nan()));
    }

    #[test]
    fn log10_p_values_reach_past_underflow() {
        // Two-sided p-values at z = 2 and z = 100, from 50-digit mpmath.
        let log10_p = wald_log10_p_values(&[-1.0, 200.0], &[0.5, 2.0]);
        assert!((log10_p[0] - -1.341_986_084_476_955_8).abs() < 1e-15);
        assert!((log10_p[1] - -2_173.570_512_873_370_3).abs() < 1e-12);
    }
}
