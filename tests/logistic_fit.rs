//! The logistf values for single coefficients are copied from
//! `<https://github.com/jzluo/firthmodels/blob/60af10d/tests/test_logistic.py>`. The statsmodels
//! coefficients, intercept, log-likelihood, and standard errors are copied from the same file. The
//! brglm2 values come from the brglm2 1.0.1 call `glm(y ~ ., binomial, method = "brglmFit",
//! type = "AS_mean", control = brglmControl(epsilon = 1e-12))`. The joint test values come from
//! the logistf 1.26.1 call `logistftest(fit, test = ~ x2 + x3 - 1)`. The coefficient in
//! `matches_logistf_with_one_centered_carrier` comes from the logistf 1.26.1 call
//! `logistf(y ~ x - 1 + offset(o), control = logistf.control(maxit = 1000, xconv = 1e-12,
//! gconv = 1e-12))`.

use firthmodels::{
    Error, FitStatus, InferenceStatus, LogisticData, LogisticOptions, MatrixView, NewtonOptions,
    ProfileOptions, Tail, fit_logistic,
};

#[path = "fixtures/logistic_separation.rs"]
mod separation;

#[path = "fixtures/logistic_unpenalized.rs"]
mod unpenalized;

const PROFILE: ProfileOptions = ProfileOptions {
    max_iterations: 25,
    tolerance: 1e-4,
};

fn features() -> MatrixView<'static> {
    MatrixView::from_row_major(separation::X, separation::N, separation::P).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], rtol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= rtol * e.abs(),
            "index {i}: got {a}, expected {e} (rtol {rtol})"
        );
    }
}

#[test]
fn matches_brglm2_with_separation() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();

    assert_eq!(fit.status(), FitStatus::Converged);

    let expected_coef = [
        3.657_714_286_43,
        0.675_978_190_17,
        -0.863_311_969_61,
        0.338_578_882_66,
    ];
    let expected_intercept = -0.443_456_328_97;
    assert_close(fit.coefficients(), &expected_coef, 1e-4);
    assert_close(&[fit.intercept()], &[expected_intercept], 1e-4);

    let expected_wald_se = [
        1.590_765_808_61,
        0.276_374_005_45,
        0.397_078_349_38,
        0.141_649_218_62,
        0.353_777_137_86,
    ];
    assert_close(fit.standard_errors(), &expected_wald_se, 1e-4);

    #[rustfmt::skip]
    let expected_covariance = [
        2.530_535_857_85, 0.066_167_204_35, -0.060_998_184_57, 0.012_309_414_28, -0.079_217_479_90,
        0.066_167_204_35, 0.076_382_590_89, -0.005_941_605_88, 0.006_409_743_46, -0.004_821_510_70,
        -0.060_998_184_57, -0.005_941_605_88, 0.157_671_215_55, -0.006_100_436_73, 0.023_132_496_30,
        0.012_309_414_28, 0.006_409_743_46, -0.006_100_436_73, 0.020_064_501_14, -0.036_172_009_89,
        -0.079_217_479_90, -0.004_821_510_70, 0.023_132_496_30, -0.036_172_009_89, 0.125_158_263_28,
    ];
    assert_close(fit.covariance().unwrap(), &expected_covariance, 1e-4);
}

fn fit_two_column_design(x: &[f64], response: &[f64], penalty_weight: f64) -> Result<(), Error> {
    let options = LogisticOptions {
        penalty_weight,
        ..LogisticOptions::default()
    };
    fit_logistic(
        MatrixView::from_row_major(x, 4, 2).unwrap(),
        false,
        response,
        None,
        None,
        options,
    )
    .map(|_| ())
}

fn near_collinear_design(delta: f64) -> [f64; 8] {
    [
        1.0,
        1.0 + delta,
        1.0,
        1.0 - delta,
        1.0,
        1.0 + delta,
        1.0,
        1.0 - delta,
    ]
}

#[test]
fn near_collinear_design_is_rejected() {
    let d = 5.0e-9;
    let x = [1.0, 1.0, 1.0, 1.0 + d, 1.0, 1.0 - d, 1.0, 1.0 + 2.0 * d];
    let response = [0.0, 0.0, 0.0, 1.0];
    for penalty_weight in [0.0, 0.5] {
        assert_eq!(
            fit_two_column_design(&x, &response, penalty_weight),
            Err(Error::IllConditionedDesign),
            "penalty_weight={penalty_weight}"
        );
    }
}

#[test]
fn ill_conditioned_design_found_at_a_trial_step_is_rejected() {
    // Both features of row 0 are 1e9. The scaled design is therefore ill-conditioned. At the start,
    // the offset of -40 gives row 0 a weight of 4e-18. With that weight, the information passes the
    // Cholesky check. The design check first runs at a trial step where the linear predictor of
    // row 0 is near 0. The weight of row 0 dominates the information at that step.
    let x = [
        1e9, 1e9, //
        -2.0, 0.0, //
        1.0, 1.0, //
        2.0, 1.0, //
        0.0, -1.0, //
        -3.0, 2.0,
    ];
    let offset = [-40.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let fit = |options| {
        fit_logistic(
            MatrixView::from_row_major(&x, 6, 2).unwrap(),
            true,
            &[1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            None,
            Some(&offset),
            options,
        )
    };
    // With a step limit of zero, every trial point is the starting point. This fit therefore fails
    // only if the evaluation at the starting point fails.
    let mut start_only = LogisticOptions::default();
    start_only.solver.max_step = 0.0;
    start_only.solver.max_iterations = 1;
    assert!(fit(start_only).is_ok());
    assert_eq!(
        fit(LogisticOptions::default()).unwrap_err(),
        Error::IllConditionedDesign
    );
}

#[test]
fn rejects_pivots_below_normal_equations_cutoff() {
    let response = [0.0, 1.0, 0.0, 1.0];
    let cutoff = (4.0 * f64::EPSILON).sqrt();

    for penalty_weight in [0.0, 0.5] {
        // With `factor = 0.98`, rounding in the information raises its Cholesky pivot ratio above
        // the cutoff.
        for factor in [0.98, 0.999] {
            assert_eq!(
                fit_two_column_design(
                    &near_collinear_design(cutoff * factor),
                    &response,
                    penalty_weight
                ),
                Err(Error::IllConditionedDesign),
                "penalty_weight={penalty_weight}, factor={factor}"
            );
        }

        let result = fit_two_column_design(
            &near_collinear_design(cutoff * 1.001),
            &response,
            penalty_weight,
        );
        assert!(
            result.is_ok(),
            "penalty_weight={penalty_weight}: got {result:?}"
        );
    }
}

#[test]
fn duplicate_columns_are_rank_deficient() {
    let x = [1.0; 8];
    let response = [0.0, 0.0, 1.0, 1.0];
    assert!(matches!(
        fit_logistic(
            MatrixView::from_row_major(&x, 4, 2).unwrap(),
            false,
            &response,
            None,
            None,
            LogisticOptions::default(),
        ),
        Err(Error::RankDeficient {
            rank: 1,
            required: 2
        })
    ));
}

#[test]
fn scaled_duplicate_column_is_rank_deficient() {
    // The second column is 1e4 times the first. Without the unit-diagonal scaling of the
    // information, the second pivot of its unpivoted Cholesky factor would be rounding noise from
    // the large column. That pivot would be above the cutoff.
    let x = [1.0, 1e4, 2.0, 2e4, 3.0, 3e4, 4.0, 4e4];
    for penalty_weight in [0.0, 0.5] {
        assert_eq!(
            fit_two_column_design(&x, &[0.0, 1.0, 1.0, 0.0], penalty_weight),
            Err(Error::RankDeficient {
                rank: 1,
                required: 2
            }),
            "penalty_weight={penalty_weight}"
        );
    }
}

#[test]
fn covariate_in_large_units_matches_the_rescaled_fit() {
    // The columns of the design are platelet counts per microliter, a 0/1 exposure, and an
    // intercept. Without column scaling, the smallest QR pivot of this design is about 3e-7 times
    // the largest. That ratio is below the cutoff sqrt(n eps) = 6.7e-7.
    let n = 2000;
    let platelets: Vec<f64> = (0..n)
        .map(|i| 250_000.0 + 1000.0 * ((i * 37) % 101) as f64)
        .collect();
    let exposure: Vec<f64> = (0..n).map(|i| f64::from((i * 13 + 5) % 7 < 2)).collect();
    let response: Vec<f64> = (0..n)
        .map(|i| {
            let eta = -1.0 + 1e-5 * (platelets[i] - 300_000.0) + 0.7 * exposure[i];
            let uniform = ((i * 7919) % 1000) as f64 / 1000.0;
            f64::from(uniform < 1.0 / (1.0 + (-eta).exp()))
        })
        .collect();
    let fit = |scale: f64, penalty_weight| {
        let x: Vec<f64> = (0..n)
            .flat_map(|i| [platelets[i] * scale, exposure[i]])
            .collect();
        let options = LogisticOptions {
            penalty_weight,
            ..LogisticOptions::default()
        };
        let fit = fit_logistic(
            MatrixView::from_row_major(&x, n, 2).unwrap(),
            true,
            &response,
            None,
            None,
            options,
        )
        .unwrap();
        assert_eq!(fit.status(), FitStatus::Converged);
        let mut parameters = fit.parameters().to_vec();
        parameters[0] *= scale;
        parameters
    };
    for penalty_weight in [0.0, 0.5] {
        assert_close(&fit(1.0, penalty_weight), &fit(1e-3, penalty_weight), 1e-6);
    }
}

#[test]
fn zero_column_is_rank_deficient() {
    for penalty_weight in [0.0, 0.5] {
        let options = LogisticOptions {
            penalty_weight,
            ..LogisticOptions::default()
        };
        let result = fit_logistic(
            MatrixView::from_row_major(&[0.0; 4], 4, 1).unwrap(),
            false,
            &[0.0, 1.0, 0.0, 1.0],
            None,
            None,
            options,
        );
        assert!(
            matches!(
                result,
                Err(Error::RankDeficient {
                    rank: 0,
                    required: 1
                })
            ),
            "penalty_weight={penalty_weight}: got {result:?}"
        );
    }
}

#[test]
fn zero_weight_rows_do_not_affect_design_conditioning() {
    let features = [0.0, 1.0, 2.0, 3.0, 1.0e8];
    let response = [0.0, 1.0, 0.0, 1.0, 0.0];
    let sample_weight = [1.0, 1.0, 1.0, 1.0, 0.0];
    let options = LogisticOptions {
        penalty_weight: 0.0,
        ..LogisticOptions::default()
    };

    let result = fit_logistic(
        MatrixView::from_row_major(&features, 5, 1).unwrap(),
        true,
        &response,
        Some(&sample_weight),
        None,
        options,
    );
    assert!(result.is_ok(), "got {result:?}");
}

#[test]
fn zero_weight_rows_do_not_supply_rank() {
    let features = [0.0, 0.0, 0.0, 0.0, 1.0];
    let response = [0.0, 1.0, 0.0, 1.0, 0.0];
    let sample_weight = [1.0, 1.0, 1.0, 1.0, 0.0];
    let options = LogisticOptions {
        penalty_weight: 0.0,
        ..LogisticOptions::default()
    };
    let data =
        LogisticData::new(MatrixView::from_row_major(&features, 5, 1).unwrap(), true).unwrap();

    // Each fit checks the conditioning of the design with its own weights. The check in the
    // unweighted fit does not affect the check in the weighted fit.
    assert!(data.fit(&response, None, None, options).is_ok());
    assert!(matches!(
        data.fit(&response, Some(&sample_weight), None, options),
        Err(Error::RankDeficient {
            rank: 1,
            required: 2
        })
    ));
}

#[test]
fn penalty_weight_zero_matches_statsmodels_logit() {
    let x = MatrixView::from_row_major(unpenalized::X, unpenalized::N, unpenalized::P).unwrap();
    let options = LogisticOptions {
        penalty_weight: 0.0,
        ..LogisticOptions::default()
    };
    let fit = fit_logistic(x, true, unpenalized::Y, None, None, options).unwrap();

    assert_eq!(fit.status(), FitStatus::Converged);

    let expected_coef = [1.280_948_872_082_475, 0.835_549_211_355_678_7];
    let expected_intercept = -0.239_469_361_920_743_75;
    let expected_loglik = -53.050_566_397_112_07;
    let expected_se = [
        0.323_451_822_685_022_6,
        0.303_908_354_953_502_66,
        0.239_021_202_810_620_53,
    ];
    // these Wald p-values come from the same statsmodels Logit fit.
    let expected_p_values = [
        7.487_251_106_992_76e-5,
        0.005_971_432_073_167_891,
        0.316_403_979_731_982_75,
    ];
    // The 95% Wald confidence intervals also come from the same statsmodels fit.
    let expected_intervals = [
        [0.646_994_948_885_995_4, 1.914_902_795_278_955_2],
        [0.239_899_781_045_998_74, 1.431_198_641_665_358_8],
        [-0.707_942_310_971_004, 0.229_003_587_129_516_43],
    ];
    assert_close(fit.coefficients(), &expected_coef, 1e-5);
    assert_close(&[fit.intercept()], &[expected_intercept], 1e-5);
    assert_close(&[fit.log_likelihood()], &[expected_loglik], 1e-5);
    assert_close(fit.standard_errors(), &expected_se, 1e-5);
    assert_close(fit.p_values(), &expected_p_values, 1e-5);
    let intervals = fit.wald_confidence_intervals(0.05).unwrap();
    assert_eq!(intervals.len(), expected_intervals.len());
    for (interval, expected) in intervals.iter().zip(&expected_intervals) {
        assert_close(interval, expected, 1e-5);
    }
}

#[test]
fn wald_confidence_intervals_reject_bad_alpha() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();
    for alpha in [f64::NAN, -0.05, 0.0, 1.0, 1.5] {
        assert!(matches!(
            fit.wald_confidence_intervals(alpha),
            Err(Error::InvalidInput(_))
        ));
    }
}

fn options_with_max_iterations(max_iterations: usize) -> LogisticOptions {
    let defaults = LogisticOptions::default();
    LogisticOptions {
        solver: NewtonOptions {
            max_iterations,
            ..defaults.solver
        },
        ..defaults
    }
}

#[test]
fn converges_on_final_permitted_iteration() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();
    assert_eq!(fit.status(), FitStatus::Converged);
    assert!(fit.iterations() >= 2);

    let exact = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        options_with_max_iterations(fit.iterations()),
    )
    .unwrap();
    assert_eq!(exact.status(), FitStatus::Converged);
    assert_eq!(exact.iterations(), fit.iterations());

    let short = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        options_with_max_iterations(fit.iterations() - 1),
    )
    .unwrap();
    assert_eq!(short.status(), FitStatus::MaximumIterations);
}

#[test]
fn matches_logistf_with_one_centered_carrier() {
    // Row 0 is the only carrier of a binary column. With the column centered and no intercept, the
    // leverage of row 0 is about 0.95. A step with the information I moves the linear predictor of
    // row 0 almost twice as far as a Newton step. Steps with I therefore oscillate around the
    // estimate. A fit that steps with I needs 50 iterations on these data.
    let x: Vec<f64> = (0..20).map(|i| f64::from(i == 0) - 1.0 / 20.0).collect();
    let response: Vec<f64> = (0..20).map(|i| f64::from(i % 2 == 0)).collect();
    let offset: Vec<f64> = (0..20).map(|i| 0.2 * f64::from(i) - 2.0).collect();
    let fit = fit_logistic(
        MatrixView::from_row_major(&x, x.len(), 1).unwrap(),
        false,
        &response,
        None,
        Some(&offset),
        LogisticOptions::default(),
    )
    .unwrap();

    assert_eq!(fit.status(), FitStatus::Converged);
    assert_close(fit.coefficients(), &[3.283_061_328_98], 1e-4);
}

#[test]
fn lrt_matches_logistf_with_separation() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();

    let expected_p_values = [
        0.000_208_414_714_9,
        0.009_317_314_895_9,
        0.023_638_571_320_6,
        0.005_588_796_916_4,
        0.199_714_719_4,
    ];
    let p_values: Vec<f64> = (0..fit.parameters().len())
        .map(|index| {
            let lrt = fit.likelihood_ratio_test(&[index], true).unwrap();
            assert!(lrt.is_valid(), "index {index}: status {:?}", lrt.status());
            lrt.p_value()
        })
        .collect();
    assert_close(&p_values, &expected_p_values, 1e-4);
}

#[test]
fn lrt_cold_start_matches_warm_start() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();

    for index in 0..fit.parameters().len() {
        let warm = fit.likelihood_ratio_test(&[index], true).unwrap();
        let cold = fit.likelihood_ratio_test(&[index], false).unwrap();
        assert!(warm.is_valid() && cold.is_valid(), "index {index}");
        assert_close(&[cold.chi_squared()], &[warm.chi_squared()], 1e-6);
        assert_close(&[cold.p_value()], &[warm.p_value()], 1e-6);
    }
}

#[test]
fn lrt_rejects_bad_indices() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();
    for indices in [&[fit.parameters().len()][..], &[], &[1, 2, 1]] {
        assert!(
            matches!(
                fit.likelihood_ratio_test(indices, true),
                Err(Error::InvalidInput(_))
            ),
            "indices {indices:?}"
        );
    }
}

#[test]
fn joint_lrt_matches_logistf_with_separation() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();
    let lrt = fit.likelihood_ratio_test(&[1, 2], true).unwrap();
    assert!(lrt.is_valid(), "status {:?}", lrt.status());
    assert_eq!(lrt.degrees_of_freedom(), 2);
    assert_close(
        &[lrt.chi_squared(), lrt.p_value()],
        &[12.389_452_680_696_7, 0.002_040_161_433_173_54],
        1e-8,
    );
}

#[test]
fn lrt_on_a_nonconverged_fit_is_invalid_without_error() {
    let options = LogisticOptions {
        penalty_weight: 0.0,
        ..LogisticOptions::default()
    };
    let fit = fit_logistic(features(), true, separation::Y, None, None, options).unwrap();
    assert_eq!(fit.status(), FitStatus::MaximumIterations);

    let lrt = fit.likelihood_ratio_test(&[0], true).unwrap();
    assert!(!lrt.is_valid());
    assert_eq!(lrt.status(), InferenceStatus::FullFitNonconvergence);
    assert_eq!(lrt.fit_status(), Some(FitStatus::MaximumIterations));
    assert_eq!(lrt.iterations(), 0);
    assert!(lrt.chi_squared().is_nan());
    assert!(lrt.p_value().is_nan());
}

#[test]
fn lrt_design_check_failure_affects_only_that_test() {
    // Both features of row 0 are 1e9. The scaled design is therefore ill-conditioned. In the full
    // fit, the first step moves the linear predictor of row 0 from -50 to above 1e8. Because row 0
    // has zero weight at that iterate and at every later one, the design check never runs in the
    // full fit. In the refit with the second coefficient at zero, the linear predictor of row 0 is
    // positive only when the first coefficient is above 5e-8. Without row 0, the estimate of the
    // first coefficient is negative. That refit reaches a point where the linear predictor of row 0
    // is near 3. Because the weight of row 0 dominates the information there, the design check runs
    // at that point. The check returns `IllConditionedDesign`.
    let x = [
        1e9, 1e9, //
        -2.0, 2.0, //
        -1.0, 1.0, //
        -1.0, 0.0, //
        1.0, -1.0, //
        1.0, 0.0, //
        2.0, -2.0,
    ];
    let fit = fit_logistic(
        MatrixView::from_row_major(&x, 7, 2).unwrap(),
        true,
        &[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        None,
        Some(&[-50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        LogisticOptions::default(),
    )
    .unwrap();
    assert!(fit.status().is_converged());
    assert_eq!(
        fit.likelihood_ratio_test(&[1], false).unwrap_err(),
        Error::IllConditionedDesign
    );
    assert!(fit.likelihood_ratio_test(&[0], false).unwrap().is_valid());
}

#[test]
fn profile_ci_matches_logistf_with_separation() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();

    let expected_lower = [
        1.390_104_289_9,
        0.160_256_803_6,
        -1.667_788_375_8,
        0.089_305_401_2,
        -1.160_893_825_06,
    ];
    let expected_upper = [
        8.587_606_211_4,
        1.256_899_921_1,
        -0.113_518_694_8,
        0.657_268_131_3,
        0.229_513_920_9,
    ];

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for index in 0..fit.parameters().len() {
        let [low, high] = fit
            .profile_confidence_interval(index, 0.05, PROFILE)
            .unwrap();
        assert!(low.is_valid(), "index {index}: status {:?}", low.status());
        assert!(high.is_valid(), "index {index}: status {:?}", high.status());
        lower.push(low.bound());
        upper.push(high.bound());
    }
    assert_close(&lower, &expected_lower, 1e-4);
    assert_close(&upper, &expected_upper, 1e-4);
}

#[test]
fn profile_ci_rejects_bad_inputs() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();

    for alpha in [f64::NAN, -0.05, 0.0, 1.0, 1.5] {
        assert!(matches!(
            fit.profile_confidence_interval(0, alpha, PROFILE),
            Err(Error::InvalidInput(_))
        ));
    }
    assert!(matches!(
        fit.profile_confidence_interval(fit.parameters().len(), 0.05, PROFILE),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        fit.profile_confidence_interval(
            0,
            0.05,
            ProfileOptions {
                max_iterations: 0,
                ..PROFILE
            }
        ),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        fit.profile_confidence_interval(
            0,
            0.05,
            ProfileOptions {
                tolerance: -1e-4,
                ..PROFILE
            }
        ),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn profile_ci_nonconvergence_returns_a_nan_bound() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();

    let bound = fit
        .profile_confidence_bound(
            0,
            Tail::Lower,
            0.05,
            ProfileOptions {
                max_iterations: 1,
                ..PROFILE
            },
        )
        .unwrap();
    assert!(!bound.is_valid());
    assert_eq!(bound.status(), InferenceStatus::ProfileBoundNonconvergence);
    assert_eq!(bound.iterations(), 1);
    assert!(bound.bound().is_nan());
}

#[test]
fn profile_bound_on_the_wrong_side_is_invalid() {
    // On these data the lower-bound iteration for the slope crosses the estimate from both starting
    // points. From the second starting point it converges to the upper bound, 0.9173, after 11
    // iterations. With a limit of 10, the run from the second starting point ends one iteration
    // before it converges. The interval from logistf 1.26.1 is [-8.9864, 0.9173].
    let fit = fit_logistic(
        MatrixView::from_row_major(&[-2.0, -2.0, -4.0, -2.0, -3.0], 5, 1).unwrap(),
        true,
        &[1.0, 0.0, 1.0, 0.0, 0.0],
        None,
        Some(&[0.0, 3.0, 0.0, -5.0, -5.0]),
        LogisticOptions::default(),
    )
    .unwrap();
    for max_iterations in [10, 50] {
        let options = ProfileOptions {
            max_iterations,
            ..ProfileOptions::default()
        };
        let lower = fit
            .profile_confidence_bound(0, Tail::Lower, 0.05, options)
            .unwrap();
        assert_eq!(lower.status(), InferenceStatus::ProfileBoundWrongSide);
        assert!(lower.bound().is_nan());
    }
}

#[test]
fn profile_ci_on_a_nonconverged_fit_is_invalid_without_error() {
    let options = LogisticOptions {
        penalty_weight: 0.0,
        ..LogisticOptions::default()
    };
    let fit = fit_logistic(features(), true, separation::Y, None, None, options).unwrap();
    assert_eq!(fit.status(), FitStatus::MaximumIterations);

    let bound = fit
        .profile_confidence_bound(0, Tail::Lower, 0.05, PROFILE)
        .unwrap();
    assert!(!bound.is_valid());
    assert_eq!(bound.status(), InferenceStatus::FullFitNonconvergence);
    assert_eq!(bound.fit_status(), Some(FitStatus::MaximumIterations));
    assert_eq!(bound.iterations(), 0);
    assert!(bound.bound().is_nan());
}

#[test]
fn wald_inference_on_a_nonconverged_fit_is_nan() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        options_with_max_iterations(1),
    )
    .unwrap();
    assert_eq!(fit.status(), FitStatus::MaximumIterations);

    assert!(fit.parameters().iter().all(|value| value.is_finite()));
    assert!(fit.standard_errors().iter().all(|value| value.is_nan()));
    assert!(fit.covariance().is_none());
    assert!(fit.p_values().iter().all(|value| value.is_nan()));
    assert!(fit.log10_p_values().iter().all(|value| value.is_nan()));
    let intervals = fit.wald_confidence_intervals(0.05).unwrap();
    assert!(intervals.iter().flatten().all(|value| value.is_nan()));
}

#[test]
fn profile_ci_with_a_single_parameter() {
    let x = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let y = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let fit = fit_logistic(
        MatrixView::from_row_major(&x, 6, 1).unwrap(),
        false,
        &y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();

    let [low, high] = fit.profile_confidence_interval(0, 0.05, PROFILE).unwrap();
    assert!(low.is_valid(), "status {:?}", low.status());
    assert!(high.is_valid(), "status {:?}", high.status());
    assert!(low.bound() < fit.parameters()[0]);
    assert!(fit.parameters()[0] < high.bound());
}

#[test]
fn integer_weights_match_duplicated_rows_across_evaluation_blocks() {
    // Outside Accelerate builds, the evaluation processes rows in blocks of 1024. Accelerate builds
    // use blocks of 16384 rows. The 18,000 duplicated rows therefore span more than one block in
    // both builds. With nine columns, Accelerate builds also use the Accelerate kernels.
    let (n, p, copies) = (6000, 8, 3);
    let x: Vec<f64> = (0..n * p)
        .map(|i| ((i * 37) % 101) as f64 / 50.0 - 1.0)
        .collect();
    let y: Vec<f64> = (0..n).map(|i| f64::from((i * 13 + 5) % 7 < 3)).collect();
    let mut duplicated_x = Vec::new();
    let mut duplicated_y = Vec::new();
    for i in 0..n {
        for _ in 0..copies {
            duplicated_x.extend_from_slice(&x[i * p..(i + 1) * p]);
            duplicated_y.push(y[i]);
        }
    }
    for penalty_weight in [0.0, 0.5] {
        let options = LogisticOptions {
            penalty_weight,
            ..LogisticOptions::default()
        };
        let weighted = fit_logistic(
            MatrixView::from_row_major(&x, n, p).unwrap(),
            true,
            &y,
            Some(&vec![copies as f64; n]),
            None,
            options,
        )
        .unwrap();
        let duplicated = fit_logistic(
            MatrixView::from_row_major(&duplicated_x, n * copies, p).unwrap(),
            true,
            &duplicated_y,
            None,
            None,
            options,
        )
        .unwrap();
        assert_eq!(weighted.status(), FitStatus::Converged);
        assert_eq!(duplicated.status(), FitStatus::Converged);
        assert_close(weighted.parameters(), duplicated.parameters(), 1e-9);
        assert_close(
            &[weighted.log_likelihood()],
            &[duplicated.log_likelihood()],
            1e-12,
        );
    }
}

#[test]
fn integer_weights_match_duplicated_rows() {
    // Row i has weight `copies[i % 4]` in the weighted fit. It appears `copies[i % 4]` times in the
    // duplicated fit.
    let copies = [1_u32, 2, 0, 3];
    let p = separation::P;
    let weight: Vec<f64> = (0..separation::N)
        .map(|i| f64::from(copies[i % 4]))
        .collect();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..separation::N {
        for _ in 0..copies[i % 4] {
            x.extend_from_slice(&separation::X[i * p..(i + 1) * p]);
            y.push(separation::Y[i]);
        }
    }
    // At penalty weight 2, every evaluation of the duplicated fit builds the step matrix from all
    // rows. Most evaluations of the weighted fit use `gathered_step_matrix` instead.
    for penalty_weight in [0.5, 2.0] {
        let options = LogisticOptions {
            penalty_weight,
            ..LogisticOptions::default()
        };
        let weighted = fit_logistic(
            features(),
            true,
            separation::Y,
            Some(&weight),
            None,
            options,
        )
        .unwrap();
        let duplicated_features = MatrixView::from_row_major(&x, y.len(), p).unwrap();
        let duplicated = fit_logistic(duplicated_features, true, &y, None, None, options).unwrap();

        assert_eq!(weighted.status(), FitStatus::Converged);
        assert_eq!(duplicated.status(), FitStatus::Converged);
        assert_close(weighted.parameters(), duplicated.parameters(), 1e-9);
        assert_close(
            weighted.standard_errors(),
            duplicated.standard_errors(),
            1e-9,
        );
        assert_close(
            &[weighted.log_likelihood()],
            &[duplicated.log_likelihood()],
            1e-12,
        );
        for index in 0..weighted.parameters().len() {
            let weighted_lrt = weighted.likelihood_ratio_test(&[index], true).unwrap();
            let duplicated_lrt = duplicated.likelihood_ratio_test(&[index], true).unwrap();
            assert!(weighted_lrt.is_valid() && duplicated_lrt.is_valid());
            assert_close(
                &[weighted_lrt.chi_squared()],
                &[duplicated_lrt.chi_squared()],
                1e-8,
            );
            let weighted_ci = weighted
                .profile_confidence_interval(index, 0.05, PROFILE)
                .unwrap();
            let duplicated_ci = duplicated
                .profile_confidence_interval(index, 0.05, PROFILE)
                .unwrap();
            for (weighted, duplicated) in weighted_ci.iter().zip(&duplicated_ci) {
                assert!(weighted.is_valid() && duplicated.is_valid());
                assert_close(&[weighted.bound()], &[duplicated.bound()], 1e-8);
            }
        }
    }
}

#[test]
fn prediction_rejects_invalid_inputs() {
    let fit = fit_logistic(
        features(),
        true,
        separation::Y,
        None,
        None,
        LogisticOptions::default(),
    )
    .unwrap();

    assert!(matches!(
        fit.predict_probability(features(), Some(&[0.0; separation::N - 1])),
        Err(Error::InvalidDimensions(_))
    ));

    let wrong_columns = [0.0; 24];
    assert!(matches!(
        fit.decision_function(
            MatrixView::from_row_major(&wrong_columns, 12, 2).unwrap(),
            None,
        ),
        Err(Error::InvalidDimensions(_))
    ));

    let mut nonfinite_features = separation::X.to_vec();
    nonfinite_features[4] = f64::INFINITY;
    assert!(matches!(
        fit.decision_function(
            MatrixView::from_row_major(&nonfinite_features, separation::N, separation::P).unwrap(),
            None,
        ),
        Err(Error::InvalidInput(_))
    ));
}

const FOUR_ROW_RESPONSE: [f64; 4] = [0.0, 0.0, 1.0, 1.0];

fn four_rows() -> MatrixView<'static> {
    MatrixView::from_row_major(&[0.5, 1.5, 2.5, 3.5], 4, 1).unwrap()
}

#[test]
fn rejects_bad_shapes() {
    let single = [1.0];
    let one_sample = MatrixView::from_row_major(&single, 1, 1).unwrap();
    assert!(matches!(
        fit_logistic(
            one_sample,
            true,
            &[0.0],
            None,
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidDimensions(_))
    ));
    assert!(matches!(
        fit_logistic(
            four_rows(),
            true,
            &[0.0, 1.0],
            None,
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidDimensions(_))
    ));
}

#[test]
fn rejects_bad_values() {
    assert!(matches!(
        fit_logistic(
            four_rows(),
            true,
            &[0.0, 0.5, 1.0, 1.0],
            None,
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidInput(_))
    ));
    let negative_penalty = LogisticOptions {
        penalty_weight: -0.5,
        ..LogisticOptions::default()
    };
    assert!(matches!(
        fit_logistic(
            four_rows(),
            true,
            &FOUR_ROW_RESPONSE,
            None,
            None,
            negative_penalty
        ),
        Err(Error::InvalidInput(_))
    ));
    let negative_weight = [1.0, 1.0, 1.0, -1.0];
    assert!(matches!(
        fit_logistic(
            four_rows(),
            true,
            &FOUR_ROW_RESPONSE,
            Some(negative_weight.as_slice()),
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidInput(_))
    ));
    let with_nan = [0.5, 1.5, f64::NAN, 3.5];
    let nan_features = MatrixView::from_row_major(&with_nan, 4, 1).unwrap();
    assert!(matches!(
        fit_logistic(
            nan_features,
            true,
            &FOUR_ROW_RESPONSE,
            None,
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn requires_positive_weight_in_each_class() {
    assert!(matches!(
        fit_logistic(
            four_rows(),
            true,
            &[1.0, 1.0, 1.0, 1.0],
            None,
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidInput(_))
    ));
    // Both classes are present. The rows with response 0 have zero weight.
    let unweighted_class = [0.0, 0.0, 1.0, 1.0];
    assert!(matches!(
        fit_logistic(
            four_rows(),
            true,
            &FOUR_ROW_RESPONSE,
            Some(unweighted_class.as_slice()),
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn requires_enough_effective_samples() {
    // With two features and an intercept, k = 3 is larger than n = 2.
    let square = [1.0, 2.0, 3.0, 4.0];
    let two_by_two = MatrixView::from_row_major(&square, 2, 2).unwrap();
    assert!(matches!(
        fit_logistic(
            two_by_two,
            true,
            &[0.0, 1.0],
            None,
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidDimensions(_))
    ));
    // With three features and an intercept, k = 4 is larger than the number of positive-weight
    // samples, 3.
    let wide = [
        1.0, 0.0, 2.0, //
        0.0, 1.0, 1.0, //
        1.0, 1.0, 0.0, //
        2.0, 0.5, 1.0,
    ];
    let four_by_three = MatrixView::from_row_major(&wide, 4, 3).unwrap();
    let one_zero_weight = [1.0, 1.0, 1.0, 0.0];
    assert!(matches!(
        fit_logistic(
            four_by_three,
            true,
            &FOUR_ROW_RESPONSE,
            Some(one_zero_weight.as_slice()),
            None,
            LogisticOptions::default()
        ),
        Err(Error::InvalidDimensions(_))
    ));
}

#[test]
fn fitted_probabilities_near_one_converge() {
    // The offsets put the fitted probabilities of the first two rows within about e^-t of
    // their responses of 1. At t = 40, these probabilities round to 1. The third row has a
    // zero feature. It supplies the response of 0 that every fit requires. The second fit uses the
    // responses 1 - y and the negated offsets. Its estimate is the negative of the first estimate.
    let features = MatrixView::from_row_major(&[1.0, -0.5, 0.0], 3, 1).unwrap();
    for t in [30.0, 40.0] {
        for sample_weight in [None, Some([2.0; 3].as_slice())] {
            for penalty_weight in [0.0, 0.5] {
                let options = LogisticOptions {
                    penalty_weight,
                    ..LogisticOptions::default()
                };
                let fit = |response: &[f64], offset: &[f64]| {
                    let fit = fit_logistic(
                        features,
                        false,
                        response,
                        sample_weight,
                        Some(offset),
                        options,
                    )
                    .unwrap();
                    assert_eq!(fit.status(), FitStatus::Converged);
                    fit.parameters()[0]
                };
                let near_one = fit(&[1.0, 1.0, 0.0], &[t, t, -800.0]);
                let near_zero = fit(&[0.0, 0.0, 1.0], &[-t, -t, 800.0]);
                assert!(
                    (near_one + near_zero).abs() <= 1e-6,
                    "t={t}, {near_one} vs {near_zero}"
                );
            }
        }
    }
}
