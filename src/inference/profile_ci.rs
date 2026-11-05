use faer::{Col, ColRef, Mat, MatRef};

use crate::inference::distributions::normal_quantile;
use crate::inference::{Estimate, InferenceStatus};
use crate::linalg::{general_solve, least_squares};
use crate::solver::IterationQuantities;
use crate::validation::{validate_alpha, validate_index};
use crate::{Error, FitStatus, Result};

/// The lower or upper bound of a profile-likelihood confidence interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    Lower,
    Upper,
}

impl Tail {
    pub(crate) const fn sign(self) -> f64 {
        match self {
            Self::Lower => -1.0,
            Self::Upper => 1.0,
        }
    }
}

/// The result of a profile-likelihood confidence bound computation.
///
/// When `status` is not `Valid`, `bound` is NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileBound {
    bound: f64,
    status: InferenceStatus,
    iterations: usize,
    // The status of the full fit when that fit did not converge.
    fit_status: Option<FitStatus>,
}

impl ProfileBound {
    const fn invalid(
        status: InferenceStatus,
        fit_status: Option<FitStatus>,
        iterations: usize,
    ) -> Self {
        Self {
            bound: f64::NAN,
            status,
            iterations,
            fit_status,
        }
    }

    #[must_use]
    pub const fn bound(&self) -> f64 {
        self.bound
    }

    #[must_use]
    pub const fn status(&self) -> InferenceStatus {
        self.status
    }

    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.status.is_valid()
    }

    /// Returns the number of iterations in the last run (see [`ProfileOptions`]).
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    #[must_use]
    pub const fn fit_status(&self) -> Option<FitStatus> {
        self.fit_status
    }
}

/// Controls for the Venzon-Moolgavkar iteration that locates a profile likelihood bound.
///
/// The first run starts at the bound of the Wald interval with the same `alpha`. If the first
/// run does not reach a valid bound, a second run starts halfway between the estimate and that
/// Wald bound. The halfway point is the starting point in Venzon and Moolgavkar (1988, Eq. 6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileOptions {
    /// The maximum number of iterations per run.
    pub max_iterations: usize,
    /// The iteration converges only when the largest of `|l - l*|` and the absolute scores of the
    /// nuisance parameters is at most this value. Here `l*` is the penalized log-likelihood at the
    /// bound.
    pub tolerance: f64,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-5,
        }
    }
}

impl ProfileOptions {
    fn validate(self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(Error::InvalidInput(
                "max_iterations must be positive".into(),
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(Error::InvalidInput(format!(
                "tolerance must be finite and nonnegative, got {}",
                self.tolerance
            )));
        }
        Ok(())
    }
}

/// Solves with `general_solve`, or with `least_squares` when the system is singular.
fn solve_with_fallback(matrix: &Mat<f64>, rhs: MatRef<'_, f64>) -> Result<Mat<f64>> {
    match general_solve(matrix, rhs) {
        Ok(solution) => Ok(solution),
        Err(Error::SingularSystem) => least_squares(matrix, rhs),
        Err(error) => Err(error),
    }
}

impl Estimate {
    /// Returns both profile likelihood bounds for coefficient `index` as `[lower, upper]`.
    pub(crate) fn profile_interval(
        &self,
        index: usize,
        alpha: f64,
        options: ProfileOptions,
        mut compute_quantities: impl FnMut(&[f64]) -> Result<IterationQuantities>,
    ) -> Result<[ProfileBound; 2]> {
        Ok([
            self.profile_bound(index, Tail::Lower, alpha, options, &mut compute_quantities)?,
            self.profile_bound(index, Tail::Upper, alpha, options, &mut compute_quantities)?,
        ])
    }

    /// Computes one profile likelihood bound for coefficient `index` with the Venzon-Moolgavkar
    /// (1988) algorithm.
    ///
    /// Solves `F(theta) = [l(theta) - l*, dl/dw]' = 0` where `beta = theta[index]`
    /// is the parameter of interest and `w` (omega) are the nuisance parameters.
    ///
    /// # References
    ///
    /// Venzon, D.J. and Moolgavkar, S.H. (1988). "A Method for Computing
    /// Profile-Likelihood-Based Confidence Intervals." Applied Statistics,
    /// 37(1), 87-94.
    #[expect(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_lines
    )]
    pub(crate) fn profile_bound(
        &self,
        index: usize,
        tail: Tail,
        alpha: f64,
        options: ProfileOptions,
        mut compute_quantities: impl FnMut(&[f64]) -> Result<IterationQuantities>,
    ) -> Result<ProfileBound> {
        let k = self.beta.len();
        validate_index(index, k)?;
        validate_alpha(alpha)?;
        options.validate()?;
        if !self.status.is_converged() {
            return Ok(ProfileBound::invalid(
                InferenceStatus::FullFitNonconvergence,
                Some(self.status),
                0,
            ));
        }
        if !self.log_likelihood.is_finite() {
            return Ok(ProfileBound::invalid(
                InferenceStatus::NonFiniteLikelihood,
                None,
                0,
            ));
        }
        let ProfileOptions {
            max_iterations,
            tolerance,
        } = options;
        let theta_hat: &[f64] = &self.beta;
        let (log_likelihood_full, information) = (self.log_likelihood, &self.information);

        let z = normal_quantile(alpha / 2.0);
        let chi2_crit = z * z;
        let l_star = log_likelihood_full - chi2_crit / 2.0;

        // Appendix step 1: D0 = d2l/dtheta2 at the estimate, approximated by -I.
        let d0 = Mat::from_fn(k, k, |i, j| -information[(i, j)]);

        let other_indices: Vec<usize> = (0..k).filter(|&j| j != index).collect();

        // Appendix step 2: compute dw/dbeta and h
        let (dw_db, d2l_db2) = if other_indices.is_empty() {
            // Single parameter case
            (Vec::new(), d0[(index, index)])
        } else {
            // d2l/dw2
            let d0_ww = Mat::from_fn(k - 1, k - 1, |i, j| {
                d0[(other_indices[i], other_indices[j])]
            });
            // d2l/dbeta*dw
            let d0_bw = Col::from_fn(k - 1, |i| d0[(index, other_indices[i])]);

            // dw/dbeta = -(d2l/dw2)^-1 (d2l/dbeta*dw)  (Eq. 5)
            let solution = solve_with_fallback(&d0_ww, d0_bw.as_mat())?;
            let dw_db: Vec<f64> = solution.col(0).iter().map(|value| -value).collect();

            // d2l_bar/dbeta2 = D_bb - D_bw D_ww^-1 D_wb (Eq. 6 denominator)
            let d2l_db2 = d0[(index, index)] + d0_bw.transpose() * ColRef::from_slice(&dw_db);
            (dw_db, d2l_db2)
        };

        let finite = |quantities: &IterationQuantities| {
            quantities.log_likelihood.is_finite()
                && quantities
                    .modified_score
                    .iter()
                    .all(|value| value.is_finite())
                && (0..k).all(|i| (0..k).all(|j| quantities.step_matrix[(i, j)].is_finite()))
        };

        // step size h (Eq. 6)
        let h = if d2l_db2 >= 0.0 {
            // A nonnegative `d2l_db2` means that D0 is not negative definite. The fallback step
            // size is then 0.5 toward the requested bound.
            tail.sign() * 0.5
        } else {
            tail.sign() * (chi2_crit / d2l_db2.abs()).sqrt() / 2.0
        };

        // Appendix step 3: theta(1) = theta_hat + h * [1, dw/dbeta]' (Eq. 4)
        let start = |distance: f64| {
            let mut theta = theta_hat.to_vec();
            theta[index] += distance;
            for (&j, &slope) in other_indices.iter().zip(&dw_db) {
                theta[j] += distance * slope;
            }
            theta
        };

        let quadratic_form = |left: ColRef<f64>, matrix: &Mat<f64>, right: ColRef<f64>| -> f64 {
            left.transpose() * (matrix * right)
        };
        // The steps can cross the estimate. The iteration can then end at or near the bound of the
        // other tail.
        let on_requested_side = |value: f64| tail.sign() * (value - theta_hat[index]) > 0.0;

        let nonconvergence = |iterations: usize| {
            ProfileBound::invalid(
                InferenceStatus::ProfileBoundNonconvergence,
                None,
                iterations,
            )
        };

        // `iterate` runs the modified Newton-Raphson iteration of appendix steps 4-9 from `theta`.
        // When `abandon` is true, `iterate` stops at the first failed evaluation or damped step. It
        // then returns a nonconvergence result.
        let mut iterate = |mut theta: Vec<f64>, abandon: bool| -> Result<ProfileBound> {
            for iteration in 0..=max_iterations {
                // Appendix step 4: compute score and Hessian at theta(i)
                let quantities = match compute_quantities(&theta) {
                    Ok(quantities) if finite(&quantities) => quantities,
                    _ if abandon => return Ok(nonconvergence(iteration)),
                    // The iteration cannot compute a valid bound or a next step from non-finite
                    // quantities.
                    Ok(_) | Err(Error::NonFinite) => {
                        return Ok(ProfileBound::invalid(
                            InferenceStatus::NonFiniteLikelihood,
                            None,
                            iteration,
                        ));
                    },
                    Err(error) => return Err(error),
                };

                // Appendix step 5: F = [l - l*, dl/dw]' (Eq. 2)
                let mut equations = quantities.modified_score.clone();
                equations[index] = quantities.log_likelihood - l_star;

                // Appendix step 9: check convergence
                // TODO: The convergence test of the paper uses the relative changes in the
                // log-likelihood and in the coefficients between iterations. This code tests
                // `|log_likelihood - l_star|` and the absolute scores of the nuisance parameters.
                let max_abs = equations
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs()));
                if max_abs <= tolerance && !quantities.step_matrix_cancelled() {
                    if !on_requested_side(theta[index]) {
                        return Ok(ProfileBound::invalid(
                            InferenceStatus::ProfileBoundWrongSide,
                            None,
                            iteration,
                        ));
                    }
                    return Ok(ProfileBound {
                        bound: theta[index],
                        status: InferenceStatus::Valid,
                        iterations: iteration,
                        fit_status: None,
                    });
                }
                if iteration == max_iterations {
                    break;
                }

                // Appendix step 4 sets D = d2l/dtheta2 at theta. Here we set it to -step_matrix.
                let d = Mat::from_fn(k, k, |i, j| -quantities.step_matrix[(i, j)]);
                // The Jacobian G (Eq. 3) is D with row `index` replaced by the modified score.
                let jacobian = Mat::from_fn(k, k, |i, j| {
                    if i == index {
                        quantities.modified_score[j]
                    } else {
                        d[(i, j)]
                    }
                });

                // Appendix step 6: v = G^-1 F (direction to subtract).
                // Solve G [v, g_j] = [F, e_index].
                let mut rhs = Mat::zeros(k, 2);
                rhs.col_mut(0).copy_from(ColRef::from_slice(&equations));
                rhs[(index, 1)] = 1.0;
                let solution = solve_with_fallback(&jacobian, rhs.as_ref())?;
                let (v, g_j) = (solution.col(0), solution.col(1));

                // Appendix step 7: quadratic correction
                // g'Dg*s^2 + (2v'Dg - 2)*s + v'Dv = 0 (Eq. 8)
                let a = quadratic_form(g_j, &d, g_j);
                let b = 2.0 * quadratic_form(v, &d, g_j) - 2.0;
                let c = quadratic_form(v, &d, v);

                let discriminant = b * b - 4.0 * a * c;

                let delta = if discriminant >= 0.0 && a.abs() > 1e-10 {
                    let sqrt_disc = discriminant.sqrt();
                    let s1 = (-b + sqrt_disc) / (2.0 * a);
                    let s2 = (-b - sqrt_disc) / (2.0 * a);

                    // The iteration selects the root whose step is shorter in the norm of -D0
                    // (pg 90). The matrix -D0 is `information`.
                    let step1 = Col::from_fn(k, |i| -v[i] - s1 * g_j[i]);
                    let step2 = Col::from_fn(k, |i| -v[i] - s2 * g_j[i]);
                    let norm1 = quadratic_form(step1.as_ref(), information, step1.as_ref());
                    let norm2 = quadratic_form(step2.as_ref(), information, step2.as_ref());
                    if norm1 < norm2 { step1 } else { step2 }
                } else if abandon {
                    return Ok(nonconvergence(iteration));
                } else {
                    // When Eq. 8 has no real root or `a` is near zero, the damped step is
                    // `-0.1 * v` (pg 92).
                    Col::from_fn(k, |i| -0.1 * v[i])
                };

                // Appendix step 8: theta(i+1) = theta(i) + delta
                for (value, change) in theta.iter_mut().zip(delta.iter()) {
                    *value += change;
                }
            }

            // The iteration did not converge within `max_iterations` iterations.
            if !on_requested_side(theta[index]) {
                return Ok(ProfileBound::invalid(
                    InferenceStatus::ProfileBoundWrongSide,
                    None,
                    max_iterations,
                ));
            }
            Ok(nonconvergence(max_iterations))
        };

        // The step size h from Eq. 6 is half the distance from the estimate to the point where the
        // quadratic approximation of the profile log-likelihood equals l_star. That distance is |z|
        // times the Wald standard error, sqrt([I^-1]_bb), because -d2l_db2 is 1 / [I^-1]_bb. At
        // start(2h), beta equals the Wald bound. The first run starts at start(2h) unless h is the
        // fallback above. If the first run reaches a valid bound without a failed evaluation or a
        // damped step, `profile_bound` returns that bound. Otherwise a second run starts at
        // start(h). The iteration count of the second run starts at zero.
        if d2l_db2 < 0.0 {
            let bound = iterate(start(2.0 * h), true)?;
            if bound.is_valid() {
                return Ok(bound);
            }
        }
        iterate(start(h), false)
    }
}
