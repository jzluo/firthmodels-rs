//! Newton-Raphson solver.
use faer::{ColRef, Mat};

use crate::linalg::{has_cancelled_direction, least_squares, spd_solve};
use crate::{Error, Result};

/// Controls for the Newton-Raphson solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewtonOptions {
    /// Maximum number of Newton-Raphson iterations.
    pub max_iterations: usize,
    /// Maximum absolute change in any coefficient per step. Larger steps are scaled down
    /// uniformly.
    pub max_step: f64,
    /// Maximum number of step-halving attempts after a rejected full step. If `0`, every evaluable
    /// full step is accepted even if it decreases the penalized log-likelihood.
    pub max_half_steps: usize,
    /// Convergence tolerance on the largest absolute entry of the modified score (`gtol` in
    /// scipy).
    pub gradient_tolerance: f64,
    /// Convergence tolerance on the largest absolute coefficient change in a Newton step (`xtol`
    /// in scipy).
    pub step_tolerance: f64,
}

impl NewtonOptions {
    pub(crate) fn validate(self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(Error::InvalidInput(
                "max_iterations must be positive".into(),
            ));
        }
        if !self.max_step.is_finite() || self.max_step < 0.0 {
            return Err(Error::InvalidInput(
                "max_step must be finite and nonnegative".into(),
            ));
        }
        if !self.gradient_tolerance.is_finite() || self.gradient_tolerance < 0.0 {
            return Err(Error::InvalidInput(
                "gradient_tolerance must be finite and nonnegative".into(),
            ));
        }
        if !self.step_tolerance.is_finite() || self.step_tolerance < 0.0 {
            return Err(Error::InvalidInput(
                "step_tolerance must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}

/// Final status of a fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitStatus {
    Converged,
    /// Step halving could not find a step that improved the penalized log-likelihood (including
    /// trial steps where evaluation failed).
    StepHalvingFailed,
    /// Failed to converge within the iteration limit.
    MaximumIterations,
}

impl FitStatus {
    #[must_use]
    pub const fn is_converged(self) -> bool {
        matches!(self, Self::Converged)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IterationQuantities {
    pub(crate) log_likelihood: f64,
    pub(crate) modified_score: Vec<f64>,
    /// The matrix of the Newton system `step_matrix @ delta = modified_score`.
    pub(crate) step_matrix: Mat<f64>,
    /// The Fisher information, when it differs from `step_matrix`.
    pub(crate) information: Option<Mat<f64>>,
    /// Diagonal of `X.T @ diag(w) @ X`, the first term of the Cox information. The solvers use it
    /// to detect cancellation in `step_matrix`.
    pub(crate) gram_diagonal: Option<Vec<f64>>,
}

impl IterationQuantities {
    /// Returns true when `step_matrix` has no correct digits in some direction. The Newton step is
    /// rounding noise at such a point, so the solvers do not accept convergence there.
    pub(crate) fn step_matrix_cancelled(&self) -> bool {
        self.gram_diagonal
            .as_deref()
            .is_some_and(|diagonal| has_cancelled_direction(&self.step_matrix, diagonal))
    }

    pub(crate) fn into_information(self) -> Mat<f64> {
        self.information.unwrap_or(self.step_matrix)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NewtonResult {
    pub(crate) beta: Vec<f64>,
    pub(crate) quantities: IterationQuantities,
    pub(crate) iterations: usize,
    pub(crate) status: FitStatus,
}

/// Solves with `spd_solve`, or with `least_squares` when the step matrix is not positive definite.
fn solve_newton_step(step_matrix: &Mat<f64>, modified_score: &[f64]) -> Result<Vec<f64>> {
    match spd_solve(step_matrix, modified_score) {
        Ok(delta) => Ok(delta),
        Err(Error::NotPositiveDefinite) => {
            let delta = least_squares(step_matrix, ColRef::from_slice(modified_score).as_mat())?;
            Ok(delta.col(0).iter().copied().collect())
        },
        Err(error) => Err(error),
    }
}

/// Maximizes a log-likelihood by Newton-Raphson with step clamping and step halving.
///
/// The result always contains the last accepted beta and quantities.
pub(crate) fn newton_raphson(
    options: NewtonOptions,
    mut beta: Vec<f64>,
    mut compute_quantities: impl FnMut(&[f64]) -> Result<IterationQuantities>,
) -> Result<NewtonResult> {
    let mut quantities = compute_quantities(&beta)?;

    // inclusive so that the loop checks convergence after the last update
    for iteration in 0..=options.max_iterations {
        let mut delta = solve_newton_step(&quantities.step_matrix, &quantities.modified_score)?;
        let max_score = max_abs(&quantities.modified_score);
        let max_delta = max_abs(&delta);

        let predicted_gain: f64 = quantities
            .modified_score
            .iter()
            .zip(&delta)
            .map(|(s, d)| 0.5 * s * d)
            .sum();
        #[expect(clippy::float_cmp)]
        let unresolvable = predicted_gain > 0.0
            && quantities.log_likelihood + predicted_gain == quantities.log_likelihood;
        let skip_line_search = max_delta < options.step_tolerance && unresolvable;

        if max_score < options.gradient_tolerance
            && max_delta < options.step_tolerance
            && !quantities.step_matrix_cancelled()
        {
            return Ok(NewtonResult {
                beta,
                quantities,
                iterations: iteration,
                status: FitStatus::Converged,
            });
        }
        if iteration == options.max_iterations {
            break;
        }

        if max_delta > options.max_step {
            let scale = options.max_step / max_delta;
            for value in &mut delta {
                *value *= scale;
            }
        }

        // A trial step can fail to evaluate (for example, when fitted probabilities saturate near
        // 0 or 1) even when a shorter step succeeds, so treat evaluation errors like a decreased
        // log-likelihood.
        let mut beta_new: Vec<f64> = beta.iter().zip(&delta).map(|(b, d)| b + d).collect();
        if let Ok(quantities_new) = compute_quantities(&beta_new)
            && (skip_line_search
                || quantities_new.log_likelihood >= quantities.log_likelihood
                || options.max_half_steps == 0)
        {
            beta = beta_new;
            quantities = quantities_new;
            continue;
        }

        let mut accepted = false;
        let mut step_factor = 0.5;
        for _ in 0..options.max_half_steps {
            for ((new, &old), &step) in beta_new.iter_mut().zip(&beta).zip(&delta) {
                *new = old + step_factor * step;
            }
            if let Ok(quantities_new) = compute_quantities(&beta_new)
                && quantities_new.log_likelihood >= quantities.log_likelihood
            {
                beta = beta_new;
                quantities = quantities_new;
                accepted = true;
                break;
            }
            step_factor *= 0.5;
        }
        if !accepted {
            return Ok(NewtonResult {
                beta,
                quantities,
                iterations: iteration,
                status: FitStatus::StepHalvingFailed,
            });
        }
    }
    Ok(NewtonResult {
        beta,
        quantities,
        iterations: options.max_iterations,
        status: FitStatus::MaximumIterations,
    })
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |maximum, &value| {
        if value.is_nan() {
            f64::INFINITY
        } else {
            maximum.max(value.abs())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn newton_options_validation() {
        let good_options = NewtonOptions {
            max_iterations: 25,
            max_step: 5.0,
            max_half_steps: 5,
            gradient_tolerance: 1e-6,
            step_tolerance: 1e-6,
        };
        assert_eq!(good_options.validate(), Ok(()));

        let zero_max_iter = NewtonOptions {
            max_iterations: 0,
            ..good_options
        };
        assert!(matches!(
            zero_max_iter.validate(),
            Err(Error::InvalidInput(_))
        ));

        let nan_options = NewtonOptions {
            max_step: f64::NAN,
            ..good_options
        };
        assert!(matches!(
            nan_options.validate(),
            Err(Error::InvalidInput(_))
        ));

        let negative_tolerance = NewtonOptions {
            gradient_tolerance: -1e-6,
            ..good_options
        };
        assert!(matches!(
            negative_tolerance.validate(),
            Err(Error::InvalidInput(_))
        ));

        let infinite_step_tolerance = NewtonOptions {
            step_tolerance: f64::INFINITY,
            ..good_options
        };
        assert!(matches!(
            infinite_step_tolerance.validate(),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn max_abs_treats_nan_as_infinite() {
        assert_eq!(max_abs(&[1.0, f64::NAN, 2.0]), f64::INFINITY);
    }

    #[test]
    fn step_below_loglik_resolution_is_taken_without_line_search() {
        // The gain is below f64 resolution. Every trial evaluates lower than the current point.
        let result = newton_raphson(
            NewtonOptions {
                max_iterations: 5,
                max_step: 5.0,
                max_half_steps: 5,
                gradient_tolerance: 1e-8,
                step_tolerance: 1e-3,
            },
            vec![0.0],
            |beta| {
                Ok(IterationQuantities {
                    log_likelihood: if beta[0] == 0.0 { -1e6 } else { -1e6 - 1e-9 },
                    modified_score: vec![1e-6 - beta[0]],
                    step_matrix: Mat::identity(1, 1),
                    information: None,
                    gram_diagonal: None,
                })
            },
        )
        .unwrap();
        assert_eq!(result.status, FitStatus::Converged);
        assert_eq!(result.beta, vec![1e-6]);
    }

    #[test]
    fn singular_step_matrix_falls_back_to_least_squares_step() {
        let result = newton_raphson(
            NewtonOptions {
                max_iterations: 1,
                max_step: 5.0,
                max_half_steps: 0,
                gradient_tolerance: 0.0,
                step_tolerance: 0.0,
            },
            vec![0.0; 2],
            |_| {
                Ok(IterationQuantities {
                    log_likelihood: 0.0,
                    modified_score: vec![1.0, 1.0],
                    step_matrix: mat![[1.0, 1.0], [1.0, 1.0]],
                    information: None,
                    gram_diagonal: None,
                })
            },
        )
        .unwrap();

        assert_eq!(result.status, FitStatus::MaximumIterations);
        for beta in result.beta {
            assert!((beta - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn evaluation_error_at_trial_point_rejects_the_step() {
        // The step matrix is half the curvature, so the full step moves beta to 2 where evaluation
        // fails, while the half step lands on the maximum at beta = 1. With `max_half_steps = 0`,
        // the fit returns `StepHalvingFailed` at beta = 0.
        for (max_half_steps, expected_status, expected_beta) in [
            (2, FitStatus::Converged, 1.0),
            (0, FitStatus::StepHalvingFailed, 0.0),
        ] {
            let result = newton_raphson(
                NewtonOptions {
                    max_iterations: 5,
                    max_step: 5.0,
                    max_half_steps,
                    gradient_tolerance: 1e-8,
                    step_tolerance: 1e-8,
                },
                vec![0.0],
                |beta| {
                    if beta[0] > 1.5 {
                        return Err(Error::RankDeficient {
                            rank: 0,
                            required: 1,
                        });
                    }
                    Ok(IterationQuantities {
                        log_likelihood: -(beta[0] - 1.0).powi(2),
                        modified_score: vec![2.0 * (1.0 - beta[0])],
                        step_matrix: Mat::identity(1, 1),
                        information: None,
                        gram_diagonal: None,
                    })
                },
            )
            .unwrap();
            assert_eq!(result.status, expected_status);
            assert_eq!(result.beta, vec![expected_beta]);
        }
    }

    #[test]
    fn step_halving_failure_returns_last_accepted_state() {
        let result = newton_raphson(
            NewtonOptions {
                max_iterations: 1,
                max_step: 5.0,
                max_half_steps: 2,
                gradient_tolerance: 0.0,
                step_tolerance: 0.0,
            },
            vec![0.0],
            |beta| {
                Ok(IterationQuantities {
                    log_likelihood: if beta[0] == 0.0 { 0.0 } else { -1.0 },
                    modified_score: vec![1.0],
                    step_matrix: Mat::identity(1, 1),
                    information: None,
                    gram_diagonal: None,
                })
            },
        )
        .unwrap();
        assert_eq!(result.status, FitStatus::StepHalvingFailed);
        assert_eq!(result.beta, vec![0.0]);
        assert_eq!(result.quantities.log_likelihood, 0.0_f64);
    }
}
