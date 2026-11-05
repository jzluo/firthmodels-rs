//! The coxphf and lifelines values for datasets from firthmodels-py's test suite are copied from
//! `<https://github.com/jzluo/firthmodels/blob/60af10d/tests/test_cox.py>`.

use firthmodels::{
    CoxData, CoxFit, CoxOptions, Error, FitStatus, MatrixView, ProfileOptions, Tail,
    concordance_index,
};
use proptest::prelude::*;

#[path = "fixtures/cox_separation.rs"]
mod separation;

#[path = "fixtures/cox_unpenalized.rs"]
mod unpenalized;

const PROFILE: ProfileOptions = ProfileOptions {
    max_iterations: 50,
    tolerance: 1e-4,
};

fn separation_fit() -> CoxFit {
    let x = MatrixView::from_row_major(separation::X, separation::N, separation::P).unwrap();
    CoxData::new(x, separation::TIME, separation::EVENT)
        .unwrap()
        .model()
        .fit()
        .unwrap()
}

fn two_individual_fit(features: &[f64; 2]) -> CoxFit {
    let x = MatrixView::from_row_major(features, 2, 1).unwrap();
    CoxData::new(x, &[1.0, 2.0], &[true, false])
        .unwrap()
        .model()
        .fit()
        .unwrap()
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
fn matches_coxphf_with_monotone_likelihood() {
    let fit = separation_fit();

    assert_eq!(fit.status(), FitStatus::Converged);

    let expected_coef = [
        3.881_580_058_3,
        0.604_206_642_7,
        -0.420_213_920_1,
        1.041_506_308_0,
    ];
    assert_close(fit.coefficients(), &expected_coef, 1e-6);

    let expected_bse = [
        1.443_019_026_9,
        0.159_960_748_3,
        0.254_180_010_7,
        0.312_878_441_0,
    ];
    assert_close(fit.standard_errors(), &expected_bse, 1e-6);

    let expected_log_likelihood = -139.852_348_142_850_93;
    assert_close(&[fit.log_likelihood()], &[expected_log_likelihood], 1e-6);
}

#[test]
fn lrt_matches_coxphf_with_monotone_likelihood() {
    let fit = separation_fit();

    let expected_p_values = [
        7.318_984_974e-9,
        1.686_334_002e-4,
        9.295_083_722e-2,
        8.463_066_114e-4,
    ];
    let p_values: Vec<f64> = (0..fit.coefficients().len())
        .map(|index| {
            let lrt = fit.likelihood_ratio_test(&[index], true).unwrap();
            assert!(lrt.is_valid(), "index {index}: status {:?}", lrt.status());
            lrt.p_value()
        })
        .collect();
    assert_close(&p_values, &expected_p_values, 1e-6);
}

#[test]
fn profile_ci_matches_coxphf_with_monotone_likelihood() {
    let fit = separation_fit();

    let expected_lower = [
        1.934_346_927_2,
        0.292_118_618_8,
        -0.921_720_064_7,
        0.434_843_127_9,
    ];
    let expected_upper = [
        8.721_567_601_5,
        0.916_427_097_8,
        0.069_941_533_6,
        1.655_046_148_2,
    ];

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for index in 0..fit.coefficients().len() {
        let [low, high] = fit
            .profile_confidence_interval(index, 0.05, PROFILE)
            .unwrap();
        assert!(low.is_valid(), "index {index}: status {:?}", low.status());
        assert!(high.is_valid(), "index {index}: status {:?}", high.status());
        lower.push(low.bound());
        upper.push(high.bound());
    }
    assert_close(&lower, &expected_lower, 1e-6);
    assert_close(&upper, &expected_upper, 1e-6);
}

#[test]
fn profile_bounds_match_coxphf_with_features_on_different_scales() {
    // Because the second feature is 1e7 times the scale of the first, the information is
    // numerically rank deficient at the first starting point of the penalized lower bound. The
    // evaluation at that point fails. The iteration then starts again from the second starting
    // point. The expected bounds come from coxphf 1.13.4.
    let x = [0.0, -1e7, 0.0, 2e7, 1.0, 1e7, 0.0, -1e7, 1.0, 2e7, 0.0, 2e7];
    let features = MatrixView::from_row_major(&x, 6, 2).unwrap();
    let time = [6.0, 5.0, 3.0, 1.0, 2.0, 4.0];
    let event = [true; 6];
    for (penalty_weight, tail, expected) in [
        (0.5, Tail::Lower, -0.915_347_406_8),
        (0.0, Tail::Upper, 4.477_125_663_9),
    ] {
        let options = CoxOptions {
            penalty_weight,
            ..CoxOptions::default()
        };
        let fit = CoxData::new(features, &time, &event)
            .unwrap()
            .model()
            .options(options)
            .fit()
            .unwrap();
        let bound = fit
            .profile_confidence_bound(0, tail, 0.05, ProfileOptions::default())
            .unwrap();
        assert!(bound.is_valid(), "status {:?}", bound.status());
        assert_close(&[bound.bound()], &[expected], 1e-6);
    }
}

#[test]
fn penalty_weight_zero_matches_lifelines_coxph() {
    let x = MatrixView::from_row_major(unpenalized::X, unpenalized::N, unpenalized::P).unwrap();
    let options = CoxOptions {
        penalty_weight: 0.0,
        ..CoxOptions::default()
    };
    let fit = CoxData::new(x, unpenalized::TIME, unpenalized::EVENT)
        .unwrap()
        .model()
        .options(options)
        .fit()
        .unwrap();

    assert_eq!(fit.status(), FitStatus::Converged);

    let expected_coef = [0.524_213_441_484_833, -0.027_371_010_108_854_982];
    assert_close(fit.coefficients(), &expected_coef, 1e-5);

    let expected_log_likelihood = -200.142_702_030_497_7;
    assert_close(&[fit.log_likelihood()], &[expected_log_likelihood], 1e-5);

    let expected_bse = [0.174_399_183_642_268_92, 0.249_246_199_003_714_28];
    assert_close(fit.standard_errors(), &expected_bse, 1e-5);
}

#[test]
fn predictions_are_invariant_to_feature_translation() {
    let features = [1.0, 0.0];
    let translated_features = [10_000_001.0, 10_000_000.0];
    let fit = two_individual_fit(&features);
    let translated_fit = two_individual_fit(&translated_features);

    assert_close(translated_fit.coefficients(), fit.coefficients(), 1e-6);

    let x = MatrixView::from_row_major(&features, 2, 1).unwrap();
    let translated_x = MatrixView::from_row_major(&translated_features, 2, 1).unwrap();
    let hazard = fit.predict_cumulative_hazard(x, None).unwrap();
    let translated_hazard = translated_fit
        .predict_cumulative_hazard(translated_x, None)
        .unwrap();
    assert_close(&translated_hazard, &hazard, 1e-6);
}

#[test]
fn zero_linear_predictor_matches_baseline_curves() {
    let fit = two_individual_fit(&[1.0, 0.0]);
    let zero = [0.0];
    let features = MatrixView::from_row_major(&zero, 1, 1).unwrap();

    assert_eq!(fit.linear_predictor(features, None).unwrap(), vec![0.0]);
    assert_eq!(
        fit.predict_cumulative_hazard(features, None).unwrap(),
        fit.cumulative_baseline_hazard()
    );
    assert_eq!(
        fit.predict_survival(features, None).unwrap(),
        fit.baseline_survival()
    );
}

#[test]
fn constant_offset_shift_leaves_predictions_unchanged() {
    // Adding a constant c to every offset leaves the coefficients unchanged. The shift divides the
    // baseline hazard by exp(c). Adding c to the new rows' offsets too gives the same predictions.
    let features = MatrixView::from_row_major(separation::X, separation::N, separation::P).unwrap();
    let fit = |offset: &[f64]| {
        CoxData::new(features, separation::TIME, separation::EVENT)
            .unwrap()
            .model()
            .offset(offset)
            .fit()
            .unwrap()
    };
    let offset: Vec<f64> = (0..separation::N)
        .map(|i| [-0.3, -0.1, 0.0, 0.2, 0.4][i % 5])
        .collect();
    let shifted: Vec<f64> = offset.iter().map(|value| value + 2.5).collect();
    let new_rows = [1.0, 0.5, -0.2, 0.0, 0.0, -1.0, 0.8, 1.0];
    let new_rows = MatrixView::from_row_major(&new_rows, 2, separation::P).unwrap();
    let new_offset = [0.4, -0.2];
    let new_shifted = new_offset.map(|value| value + 2.5);

    assert_close(
        &fit(&shifted)
            .predict_survival(new_rows, Some(&new_shifted))
            .unwrap(),
        &fit(&offset)
            .predict_survival(new_rows, Some(&new_offset))
            .unwrap(),
        1e-10,
    );
}

#[test]
fn prediction_rejects_invalid_inputs() {
    let fit = two_individual_fit(&[1.0, 0.0]);

    let wrong_columns = [0.0; 2];
    let wrong_columns = MatrixView::from_row_major(&wrong_columns, 1, 2).unwrap();
    assert!(matches!(
        fit.predict_cumulative_hazard(wrong_columns, None),
        Err(Error::InvalidDimensions(_))
    ));

    let nonfinite = [f64::NAN];
    let nonfinite = MatrixView::from_row_major(&nonfinite, 1, 1).unwrap();
    assert!(matches!(
        fit.predict_survival(nonfinite, None),
        Err(Error::InvalidInput(_))
    ));

    let one_row = [0.0];
    let one_row = MatrixView::from_row_major(&one_row, 1, 1).unwrap();
    assert!(matches!(
        fit.predict_cumulative_hazard(one_row, Some(&[0.0, 0.0])),
        Err(Error::InvalidDimensions(_))
    ));
}

#[test]
fn concordance_matches_python_on_separation_data() {
    let fit = separation_fit();
    let features = MatrixView::from_row_major(separation::X, separation::N, separation::P).unwrap();

    assert_eq!(
        fit.concordance(features, separation::TIME, separation::EVENT, None)
            .unwrap(),
        0.808_899_483_512_117_6
    );
}

#[test]
fn censoring_before_every_event_preserves_fit_and_inference() {
    let mut x = Vec::new();
    let mut time = Vec::new();
    let mut event = Vec::new();
    for i in 0..64 {
        let value = f64::from(i);
        x.extend([value.sin(), (value * 0.7).cos(), f64::from(i % 7) - 3.0]);
        time.push(value + 1.0);
        event.push(i % 3 != 1);
    }
    // This sample is censored at the time of the earliest event. It is therefore in the risk set of
    // that event.
    x.extend([6.0, -2.0, 3.0]);
    time.push(1.0);
    event.push(false);
    let view = MatrixView::from_row_major(&x, time.len(), 3).unwrap();
    let expected = CoxData::new(view, &time, &event)
        .unwrap()
        .model()
        .fit()
        .unwrap();
    assert_eq!(expected.status(), FitStatus::Converged);
    let close = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{actual} vs {expected}"
            );
        }
    };
    for i in 0..129 {
        let value = 1e6 + f64::from(i);
        x.extend([value, -value, value * 0.5]);
        time.push(0.5);
        event.push(false);
    }
    let extended = MatrixView::from_row_major(&x, time.len(), 3).unwrap();
    let fit = CoxData::new(extended, &time, &event)
        .unwrap()
        .model()
        .fit()
        .unwrap();
    assert_eq!(fit.status(), expected.status());
    assert_eq!(fit.iterations(), expected.iterations());
    close(fit.coefficients(), expected.coefficients());
    close(&[fit.log_likelihood()], &[expected.log_likelihood()]);
    close(fit.standard_errors(), expected.standard_errors());
    close(fit.unique_times(), expected.unique_times());
    close(
        fit.cumulative_baseline_hazard(),
        expected.cumulative_baseline_hazard(),
    );

    let actual_lrt = fit.likelihood_ratio_test(&[0], true).unwrap();
    let expected_lrt = expected.likelihood_ratio_test(&[0], true).unwrap();
    assert_eq!(actual_lrt.status(), expected_lrt.status());
    assert!(actual_lrt.is_valid());
    close(&[actual_lrt.p_value()], &[expected_lrt.p_value()]);
    let actual_ci = fit.profile_confidence_interval(0, 0.05, PROFILE).unwrap();
    let expected_ci = expected
        .profile_confidence_interval(0, 0.05, PROFILE)
        .unwrap();
    for (actual, expected) in actual_ci.iter().zip(expected_ci) {
        assert_eq!(actual.status(), expected.status());
        assert!(actual.is_valid());
        close(&[actual.bound()], &[expected.bound()]);
    }
}

/// Fits the separation data twice. In the first fit, row i has weight `copies[i % 4]`. In the
/// second fit, row i appears `copies[i % 4]` times.
fn weighted_and_duplicated_fits(copies: [u32; 4]) -> (CoxFit, CoxFit) {
    let p = separation::P;
    let weight: Vec<f64> = (0..separation::N)
        .map(|i| f64::from(copies[i % 4]))
        .collect();
    let mut x = Vec::new();
    let mut time = Vec::new();
    let mut event = Vec::new();
    for i in 0..separation::N {
        for _ in 0..copies[i % 4] {
            x.extend_from_slice(&separation::X[i * p..(i + 1) * p]);
            time.push(separation::TIME[i]);
            event.push(separation::EVENT[i]);
        }
    }
    let features = MatrixView::from_row_major(separation::X, separation::N, p).unwrap();
    let weighted = CoxData::new(features, separation::TIME, separation::EVENT)
        .unwrap()
        .model()
        .sample_weight(&weight)
        .fit()
        .unwrap();
    let duplicated_features = MatrixView::from_row_major(&x, time.len(), p).unwrap();
    let duplicated = CoxData::new(duplicated_features, &time, &event)
        .unwrap()
        .model()
        .fit()
        .unwrap();
    (weighted, duplicated)
}

#[test]
fn integer_weights_match_duplicated_rows() {
    // Rows with a copy count of 0 have weight 0 in the weighted fit. They do not appear in the
    // duplicated fit.
    let (weighted, duplicated) = weighted_and_duplicated_fits([1, 2, 0, 3]);
    assert_eq!(weighted.status(), FitStatus::Converged);
    assert_eq!(duplicated.status(), FitStatus::Converged);
    assert_close(weighted.coefficients(), duplicated.coefficients(), 1e-9);
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
    assert_eq!(weighted.unique_times(), duplicated.unique_times());
    assert_close(
        weighted.cumulative_baseline_hazard(),
        duplicated.cumulative_baseline_hazard(),
        1e-9,
    );

    for index in 0..separation::P {
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

#[test]
fn offset_matches_coefficient_shift() {
    let (column, shift) = (1, 0.75);
    let offset: Vec<f64> = (0..separation::N)
        .map(|i| shift * separation::X[i * separation::P + column])
        .collect();
    let features = MatrixView::from_row_major(separation::X, separation::N, separation::P).unwrap();
    let fit = separation_fit();
    let offset_fit = CoxData::new(features, separation::TIME, separation::EVENT)
        .unwrap()
        .model()
        .offset(&offset)
        .fit()
        .unwrap();

    let mut expected = fit.coefficients().to_vec();
    expected[column] -= shift;
    assert_eq!(offset_fit.status(), FitStatus::Converged);
    assert_close(offset_fit.coefficients(), &expected, 1e-8);
    assert_close(offset_fit.standard_errors(), fit.standard_errors(), 1e-8);
    assert_close(
        &[offset_fit.log_likelihood()],
        &[fit.log_likelihood()],
        1e-12,
    );
    assert_close(
        offset_fit.cumulative_baseline_hazard(),
        fit.cumulative_baseline_hazard(),
        1e-8,
    );
}

#[test]
fn unpenalized_monotone_likelihood_does_not_converge() {
    // Each event has the smallest feature value in its risk set. The partial likelihood therefore
    // increases as the coefficient decreases. It has no maximum.
    let x: Vec<f64> = (1..=8).map(f64::from).collect();
    let event: Vec<bool> = (1..=8).map(|t| t % 2 == 1).collect();
    let options = CoxOptions {
        penalty_weight: 0.0,
        ..CoxOptions::default()
    };
    let features = MatrixView::from_row_major(&x, 8, 1).unwrap();
    let fit = CoxData::new(features, &x, &event)
        .unwrap()
        .model()
        .options(options)
        .fit()
        .unwrap();
    assert!(!fit.status().is_converged(), "status {:?}", fit.status());
}

#[test]
fn duplicate_columns_are_rank_deficient() {
    let x_data = [
        1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0,
    ];
    let x = MatrixView::from_row_major(&x_data, 8, 2).unwrap();
    let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let event = [true, false, true, false, true, false, true, false];
    // The binary column is one only in the zero-weight rows.
    let binary_data = [
        1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 0.0, 5.0, 1.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
    ];
    let binary = MatrixView::from_row_major(&binary_data, 8, 2).unwrap();
    let sample_weight = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    // With the event at time 1 at zero weight, the sample censored at time 2 is in no risk set
    // of a positive-weight event. The binary column is one only in that sample.
    let unweighted_first_event = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let late_binary_data = [
        1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
    ];
    let late_binary = MatrixView::from_row_major(&late_binary_data, 8, 2).unwrap();

    for penalty_weight in [0.5, 0.0] {
        let options = CoxOptions {
            penalty_weight,
            ..CoxOptions::default()
        };
        for result in [
            CoxData::new(x, &time, &event)
                .unwrap()
                .model()
                .options(options)
                .fit(),
            CoxData::new(binary, &time, &event)
                .unwrap()
                .model()
                .sample_weight(&sample_weight)
                .options(options)
                .fit(),
            CoxData::new(late_binary, &time, &event)
                .unwrap()
                .model()
                .sample_weight(&unweighted_first_event)
                .options(options)
                .fit(),
        ] {
            assert!(
                matches!(
                    result,
                    Err(Error::RankDeficient {
                        rank: 1,
                        required: 2
                    })
                ),
                "penalty_weight={penalty_weight}"
            );
        }
    }
}

#[test]
fn indicator_columns_that_sum_to_one_are_rank_deficient() {
    // Every row has exactly one indicator equal to one. Adding the same value to all three
    // coefficients therefore leaves the partial likelihood unchanged. Rounding in the information
    // keeps its smallest pivot above the cutoff of `factor_information`.
    let (n, k) = (60, 3);
    let x: Vec<f64> = (0..n * k).map(|a| f64::from(a / k % k == a % k)).collect();
    // Each of the times 1 to 20 has one event from each category.
    let time: Vec<f64> = (0..20).flat_map(|t| [f64::from(t + 1); 3]).collect();
    for penalty_weight in [0.5, 0.0] {
        let options = CoxOptions {
            penalty_weight,
            ..CoxOptions::default()
        };
        let result = CoxData::new(
            MatrixView::from_row_major(&x, n, k).unwrap(),
            &time,
            &[true; 60],
        )
        .unwrap()
        .model()
        .options(options)
        .fit();
        assert!(
            matches!(
                result,
                Err(Error::RankDeficient {
                    rank: 2,
                    required: 3
                })
            ),
            "penalty_weight={penalty_weight}: {result:?}"
        );
    }
}

fn pairwise_concordance_reference(time: &[f64], event: &[bool], risk: &[f64]) -> f64 {
    let n = time.len();
    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut tied = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let (shorter, longer) = match (event[i], event[j]) {
                (true, true) if time[i] < time[j] => (i, j),
                (true, true) if time[j] < time[i] => (j, i),
                (true, false) if time[i] <= time[j] => (i, j),
                (false, true) if time[j] <= time[i] => (j, i),
                _ => continue,
            };

            if risk[shorter] > risk[longer] {
                concordant += 1;
            } else if risk[shorter] < risk[longer] {
                discordant += 1;
            } else {
                tied += 1;
            }
        }
    }
    let total = concordant + discordant + tied;
    if total == 0 {
        0.5
    } else {
        (concordant as f64 + 0.5 * tied as f64) / total as f64
    }
}

proptest! {
    #[test]
    fn concordance_matches_pairwise_reference(
        rows in proptest::collection::vec((any::<bool>(), 0u8..8, -40i8..41), 0..65),
    ) {
        let event: Vec<bool> = rows.iter().map(|row| row.0).collect();
        let time: Vec<f64> = rows.iter().map(|row| f64::from(row.1)).collect();
        let risk: Vec<f64> = rows.iter().map(|row| f64::from(row.2)).collect();
        let expected = pairwise_concordance_reference(&time, &event, &risk);
        prop_assert_eq!(concordance_index(&time, &event, &risk).unwrap().to_bits(), expected.to_bits());
    }
}

#[test]
fn concordance_ties_negative_and_positive_zero() {
    assert_eq!(
        concordance_index(&[1.0, 2.0], &[true, false], &[-0.0, 0.0]).unwrap(),
        0.5
    );
}

#[test]
fn concordance_index_rejects_invalid_inputs() {
    assert!(matches!(
        concordance_index(&[1.0, 2.0], &[true], &[2.0, 1.0]),
        Err(Error::InvalidDimensions(_))
    ));
    assert!(matches!(
        concordance_index(&[1.0, 2.0], &[true, false], &[1.0]),
        Err(Error::InvalidDimensions(_))
    ));
    assert!(matches!(
        concordance_index(&[f64::NAN], &[true], &[1.0]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        concordance_index(&[-1.0], &[true], &[1.0]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        concordance_index(&[1.0], &[true], &[f64::INFINITY]),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn fit_rejects_bad_inputs() {
    let x_data = [1.0, 0.0];
    let x = MatrixView::from_row_major(&x_data, 2, 1).unwrap();
    let time = [1.0, 2.0];
    let event = [true, false];
    let data = CoxData::new(x, &time, &event).unwrap();

    let bad_penalty = CoxOptions {
        penalty_weight: f64::NAN,
        ..CoxOptions::default()
    };
    assert!(matches!(
        data.model().options(bad_penalty).fit(),
        Err(Error::InvalidInput(_))
    ));

    let nan_x_data = [f64::NAN, 0.0];
    let nan_x = MatrixView::from_row_major(&nan_x_data, 2, 1).unwrap();
    assert!(matches!(
        CoxData::new(nan_x, &time, &event),
        Err(Error::InvalidInput(_))
    ));

    // The survival-data validation rejects data in which every sample is censored.
    assert!(matches!(
        CoxData::new(x, &time, &[false, false]),
        Err(Error::InvalidInput(_))
    ));

    assert!(matches!(
        data.model().sample_weight(&[1.0]).fit(),
        Err(Error::InvalidDimensions(_))
    ));
    assert!(matches!(
        data.model().offset(&[0.0]).fit(),
        Err(Error::InvalidDimensions(_))
    ));
    for sample_weight in [[1.0, -1.0], [f64::NAN, 1.0], [f64::INFINITY, 1.0]] {
        assert!(matches!(
            data.model().sample_weight(&sample_weight).fit(),
            Err(Error::InvalidInput(_))
        ));
    }
    assert!(matches!(
        data.model().offset(&[f64::NAN, 0.0]).fit(),
        Err(Error::InvalidInput(_))
    ));
    // The only event has zero weight.
    assert!(matches!(
        data.model().sample_weight(&[0.0, 1.0]).fit(),
        Err(Error::InvalidInput(_))
    ));
}
