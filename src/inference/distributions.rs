//! Upper tails and their logarithms for the standard normal and chi-squared distributions, and
//! the standard normal quantile.
//!
//! To preserve relative accuracy, `normal_sf` and `chi_squared_1df_sf` correct the erfc argument
//! for its rounding error.

use core::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI, LN_2, PI};

use coefficients::{A, B, C, D, E, F};

// 1 / sqrt(2) - FRAC_1_SQRT_2.
const FRAC_1_SQRT_2_LOW: f64 = -4.833_646_656_726_457e-17;

// At t² = 676, erfc(t) is still above the smallest normal double. The remainder of the 12-term
// series in `log_erfc_series` is below 9e-27 there.
const SERIES_START: f64 = 676.0;

/// Probability that a standard normal exceeds `z`.
pub(crate) fn normal_sf(z: f64) -> f64 {
    let t = z * FRAC_1_SQRT_2;
    let dt = z.mul_add(FRAC_1_SQRT_2, -t) + z * FRAC_1_SQRT_2_LOW;
    0.5 * erfc_shifted(t, dt)
}

/// Probability that a 1-df chi-squared exceeds `x`.
pub(crate) fn chi_squared_1df_sf(x: f64) -> f64 {
    let half = x / 2.0;
    let t = half.sqrt();
    let dt = (-t).mul_add(t, half) / (2.0 * t);
    erfc_shifted(t, dt)
}

/// Natural log of `normal_sf(z)`. The result is finite even where `normal_sf(z)` underflows to
/// zero.
pub(crate) fn normal_log_sf(z: f64) -> f64 {
    let t2 = (z / 2.0) * z;
    if z < 0.0 {
        (-normal_sf(-z)).ln_1p()
    } else if t2 < SERIES_START {
        normal_sf(z).ln()
    } else {
        log_erfc_series(t2) - LN_2
    }
}

/// Natural log of `chi_squared_1df_sf(x)`. The result is finite even where `chi_squared_1df_sf(x)`
/// underflows to zero.
pub(crate) fn chi_squared_1df_log_sf(x: f64) -> f64 {
    if x < 0.5 {
        (-libm::erf(x.sqrt() * FRAC_1_SQRT_2)).ln_1p()
    } else if x / 2.0 < SERIES_START {
        chi_squared_1df_sf(x).ln()
    } else {
        log_erfc_series(x / 2.0)
    }
}

/// Probability that a chi-squared with `df` degrees of freedom exceeds `x`.
pub(crate) fn chi_squared_sf(x: f64, df: usize) -> f64 {
    if df == 1 {
        chi_squared_1df_sf(x)
    } else if x < df as f64 {
        1.0 - chi_squared_cdf_series(x, df)
    } else {
        chi_squared_upper_log_sf(x, df).exp()
    }
}

/// Natural log of `chi_squared_sf(x, df)`. The result is finite even where `chi_squared_sf(x, df)`
/// underflows to zero.
pub(crate) fn chi_squared_log_sf(x: f64, df: usize) -> f64 {
    if df == 1 {
        chi_squared_1df_log_sf(x)
    } else if x < df as f64 {
        (-chi_squared_cdf_series(x, df)).ln_1p()
    } else {
        chi_squared_upper_log_sf(x, df)
    }
}

/// Computes the chi-squared CDF for `x < df` from the power series of the lower incomplete gamma
/// function. In that range, each term of the series is smaller than the term before it.
fn chi_squared_cdf_series(x: f64, df: usize) -> f64 {
    let (y, a) = (x / 2.0, df as f64 / 2.0);
    let (mut term, mut sum, mut k) = (1.0, 1.0, 1.0);
    while term > f64::EPSILON * sum {
        term *= y / (a + k);
        sum += term;
        k += 1.0;
    }
    (a * y.ln() - y - libm::lgamma(a + 1.0)).exp() * sum
}

/// `ln chi_squared_sf(x, df)` for `x >= df` and `df >= 2`.
///
/// With `y = x / 2` and `m = floor(df / 2)`, the tail is `exp(-y)` times the sum of `m` terms
/// `y^(j + c) / Gamma(j + c + 1)`, where `c` is 1/2 for odd `df` and 0 for even `df` (plus the
/// 1-df tail when `df` is odd). Because `y >= j + c`, the terms grow with `j`, so this factors
/// out `exp(-y)` times the largest (last) term and returns the sum of the two logs.
fn chi_squared_upper_log_sf(x: f64, df: usize) -> f64 {
    if x == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    let y = x / 2.0;
    let odd = df % 2 == 1;
    let c = if odd { 0.5 } else { 0.0 };
    let m = df / 2;
    let last = m as f64 - 1.0 + c;
    let log_scaled_last = last * y.ln() - libm::lgamma(last + 1.0) - y;
    let (mut ratio, mut sum) = (1.0, 1.0);
    for j in (1..m).rev() {
        ratio *= (j as f64 + c) / y;
        sum += ratio;
    }
    if odd {
        let relative_1df = (chi_squared_1df_log_sf(x) - log_scaled_last).exp();
        log_scaled_last + (sum + relative_1df).ln()
    } else {
        log_scaled_last + sum.ln()
    }
}

/// `ln erfc(sqrt(t2))` from the asymptotic expansion for `t2 >= SERIES_START`.
fn log_erfc_series(t2: f64) -> f64 {
    let u = -0.5 / t2;
    let (mut series, mut term) = (1.0, 1.0);
    for k in 1..12 {
        term *= f64::from(2 * k - 1) * u;
        series += term;
    }
    -t2 - f64::midpoint(PI.ln(), t2.ln()) + series.ln()
}

/// Approximates `erfc(t + dt)` using a first-order correction.
fn erfc_shifted(t: f64, dt: f64) -> f64 {
    let erfc = libm::erfc(t);
    // The residual `dt` is non-finite only when `t` is zero, infinite, or NaN. In those cases,
    // `erfc(t)` is exact or NaN.
    if dt.is_finite() {
        erfc - FRAC_2_SQRT_PI * (-t * t).exp() * dt
    } else {
        erfc
    }
}

/// Standard normal quantile: the `z` with `P(Z <= z) = p`.
///
/// Uses Wichura's AS241 algorithm. For an upper-tail probability `p`, use
/// `-normal_quantile(p)` to avoid cancellation in `1 - p`.
///
/// Returns `-inf` at 0, `inf` at 1, and NaN outside `[0, 1]`.
pub(crate) fn normal_quantile(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * poly(&A, r) / poly(&B, r);
    }
    let tail = if q < 0.0 { p } else { 1.0 - p };
    if tail == 0.0 {
        return if q < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    let r = (-tail.ln()).sqrt();
    let magnitude = if r <= 5.0 {
        let r = r - 1.6;
        poly(&C, r) / poly(&D, r)
    } else {
        let r = r - 5.0;
        poly(&E, r) / poly(&F, r)
    };
    if q < 0.0 { -magnitude } else { magnitude }
}

/// Returns `coefficients[0] + coefficients[1] * x + ... + coefficients[7] * x^7`.
fn poly(coefficients: &[f64; 8], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

// The literals below have all of the digits published in AS 241. An `f64` represents only about 16
// significant digits.
#[expect(clippy::excessive_precision)]
mod coefficients {
    pub const A: [f64; 8] = [
        3.387_132_872_796_366_608,
        133.141_667_891_784_377_45,
        1_971.590_950_306_551_442_7,
        13_731.693_765_509_461_125,
        45_921.953_931_549_871_457,
        67_265.770_927_008_700_853,
        33_430.575_583_588_128_105,
        2_509.080_928_730_122_672_7,
    ];
    pub const B: [f64; 8] = [
        1.0,
        42.313_330_701_600_911_252,
        687.187_007_492_057_908_3,
        5_394.196_021_424_751_107_7,
        21_213.794_301_586_595_867,
        39_307.895_800_092_710_61,
        28_729.085_735_721_942_674,
        5_226.495_278_852_854_561,
    ];
    pub const C: [f64; 8] = [
        1.423_437_110_749_683_577_34,
        4.630_337_846_156_545_295_9,
        5.769_497_221_460_691_405_5,
        3.647_848_324_763_204_605_04,
        1.270_458_252_452_368_382_58,
        0.241_780_725_177_450_611_77,
        0.022_723_844_989_269_184_583_3,
        7.745_450_142_783_414_076_4e-4,
    ];
    pub const D: [f64; 8] = [
        1.0,
        2.053_191_626_637_758_821_87,
        1.676_384_830_183_803_849_4,
        0.689_767_334_985_100_004_55,
        0.148_103_976_427_480_074_59,
        0.015_198_666_563_616_457_196_6,
        5.475_938_084_995_344_946e-4,
        1.050_750_071_644_416_843_24e-9,
    ];
    pub const E: [f64; 8] = [
        6.657_904_643_501_103_777_2,
        5.463_784_911_164_114_369_9,
        1.784_826_539_917_291_335_8,
        0.296_560_571_828_504_891_23,
        0.026_532_189_526_576_123_093,
        1.242_660_947_388_078_438_6e-3,
        2.711_555_568_743_487_578_15e-5,
        2.010_334_399_292_288_132_65e-7,
    ];
    pub const F: [f64; 8] = [
        1.0,
        0.599_832_206_555_887_937_69,
        0.136_929_880_922_735_805_31,
        0.014_875_361_290_850_614_852_5,
        7.868_691_311_456_132_591e-4,
        1.846_318_317_510_054_681_8e-5,
        1.421_511_758_316_445_888_7e-7,
        2.044_263_103_389_939_785_64e-15,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the distance in ulps between `actual` and `expected`. The two values must have the
    /// same sign. Unlike a relative bound, the distance stays meaningful for subnormal values.
    #[expect(clippy::cast_possible_wrap)]
    fn ulp_distance(actual: f64, expected: f64) -> i64 {
        (actual.to_bits() as i64 - expected.to_bits() as i64).abs()
    }

    /// `expected` is the double nearest a high-precision reference evaluated at the exact input.
    fn assert_close(actual: f64, expected: f64, max_ulps: i64) {
        assert!(
            ulp_distance(actual, expected) <= max_ulps,
            "got {actual:e}, expected {expected:e} ({} ulps)",
            ulp_distance(actual, expected)
        );
    }

    /// The reference values include points in all three AS241 regions and at the boundary `r = 5`
    /// of the tail region.
    #[test]
    fn quantile_matches_references() {
        let references = [
            (0.1, -1.281_551_565_544_600_5),
            (0.25, -0.674_489_750_196_081_7),
            (0.75, 0.674_489_750_196_081_7),
            (0.9, 1.281_551_565_544_600_5),
            (0.075, -1.439_531_470_938_456),
            (0.925, 1.439_531_470_938_456),
            (0.05, -1.644_853_626_951_472_7),
            (0.025, -1.959_963_984_540_054_2),
            (0.001, -3.090_232_306_167_813_5),
            (1e-6, -4.753_424_308_822_899),
            (1e-10, -6.361_340_902_404_056),
            (1.388_794_386_496_402_1e-11, -6.657_904_643_501_103),
            (1.388_794_386_496_394_7e-11, -6.657_904_643_501_104),
            (1e-12, -7.034_483_825_301_132),
            (1e-20, -9.262_340_089_798_408),
            (1e-50, -14.933_337_534_788_489),
            (1e-100, -21.273_453_560_965_324),
            (1e-300, -37.047_096_299_361_2),
            (5e-324, -38.467_405_617_144_346),
        ];
        for (p, expected) in references {
            assert_close(normal_quantile(p), expected, 2);
        }
    }

    #[test]
    fn quantile_edges() {
        assert_eq!(normal_quantile(0.5), 0.0);
        assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_quantile(1.0), f64::INFINITY);
        for p in [f64::NAN, -0.1, 1.1, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(normal_quantile(p).is_nan(), "p = {p}");
        }
    }

    #[test]
    fn normal_sf_matches_references() {
        let references = [
            (0.5, 0.308_537_538_725_986_9),
            (1.0, 0.158_655_253_931_457_05),
            (1.959_963_984_540_054, 0.025_000_000_000_000_012),
            (3.0, 0.001_349_898_031_630_094_6),
            (5.0, 2.866_515_718_791_939e-7),
            (8.0, 6.220_960_574_271_784e-16),
            (10.0, 7.619_853_024_160_525e-24),
            (20.0, 2.753_624_118_606_233_7e-89),
            (30.0, 4.906_713_927_148_187e-198),
            (37.0, 5.725_571_222_524_577e-300),
            (37.5, 4.605_353_009_581_955e-308),
            (38.0, 2.885_428_35e-316),
            (38.4, 6.4e-323),
        ];
        for (z, expected) in references {
            assert_close(normal_sf(z), expected, 2);
        }
        assert_eq!(normal_sf(0.0), 0.5);
        assert_eq!(normal_sf(38.5), 0.0);
        assert_eq!(normal_sf(1e200), 0.0);
        assert_eq!(normal_sf(f64::INFINITY), 0.0);
        assert!(normal_sf(f64::NAN).is_nan());
    }

    #[test]
    fn chi_squared_sf_matches_references() {
        let references = [
            (0.1, 0.751_829_634_045_849_2),
            (1.0, 0.317_310_507_862_914_1),
            (3.841_458_820_694_124, 0.050_000_000_000_000_06),
            (10.0, 0.001_565_402_258_002_549_7),
            (32.0, 1.541_725_790_028_002e-8),
            (100.0, 1.523_970_604_832_105e-23),
            (300.0, 3.294_362_383_314_041e-67),
            (1000.0, 1.795_832_784_800_726_2e-219),
            (1400.0, 2.101_014_516_264_217_6e-306),
            (1407.0, 6.328_734_450_614_197e-308),
            (1450.0, 2.867_198e-317),
            (1479.0, 1.5e-323),
        ];
        for (x, expected) in references {
            assert_close(chi_squared_1df_sf(x), expected, 2);
        }
        assert_eq!(chi_squared_1df_sf(0.0), 1.0);
        assert_eq!(chi_squared_1df_sf(1500.0), 0.0);
        assert_eq!(chi_squared_1df_sf(1e100), 0.0);
        assert_eq!(chi_squared_1df_sf(f64::INFINITY), 0.0);
        assert!(chi_squared_1df_sf(f64::NAN).is_nan());
    }

    /// Allows four ulps for rounding in the `ln_1p` branches.
    #[test]
    fn normal_log_sf_matches_references() {
        let references = [
            (-13.730_036_413_643_795, -3.354_989_225_428_348_5e-43),
            (-3.0, -0.001_350_809_964_748_193_8),
            (-0.5, -0.368_946_415_288_656_4),
            (0.0, -LN_2),
            (1.0, -1.841_021_645_009_263_6),
            (5.0, -15.064_998_393_988_725),
            (20.0, -203.917_155_371_097_27),
            (37.5, -707.668_989_317_507_2),
            (36.769_552_621_700_46, -680.524_346_943_753_8),
            (36.769_552_621_700_47, -680.524_346_943_754_2),
            (38.5, -745.695_270_290_411),
            (1e5, -5_000_000_012.431_864),
            (1.8e154, -1.62e308),
        ];
        for (z, expected) in references {
            assert_close(normal_log_sf(z), expected, 4);
        }
        assert_eq!(normal_log_sf(f64::NEG_INFINITY), 0.0);
        assert_eq!(normal_log_sf(1.9e154), f64::NEG_INFINITY);
        assert_eq!(normal_log_sf(f64::INFINITY), f64::NEG_INFINITY);
        assert!(normal_log_sf(f64::NAN).is_nan());
    }

    #[test]
    fn chi_squared_log_sf_matches_references() {
        let references = [
            (5e-324, -1.773_504_888_603_627_4e-162),
            (1e-10, -7.978_877_439_053_607e-6),
            (0.257_154_197_984_314_77, -0.490_888_962_112_702_7),
            (0.499_999_999_999_999_94, -0.735_011_129_837_084_4),
            (0.5, -0.735_011_129_837_084_4),
            (1.0, -1.147_874_464_449_318_2),
            (3.841_458_820_694_124, -2.995_732_273_553_99),
            (10.0, -6.459_612_454_150_122),
            (100.0, -52.538_137_969_952_52),
            (1000.0, -503.680_666_504_381_7),
            (1_351.999_999_999_999_8, -679.831_199_763_194_1),
            (1352.0, -679.831_199_763_194_3),
            (1500.0, -753.883_067_105_382_5),
            (1e10, -5_000_000_011.738_717),
            (f64::MAX, -8.988_465_674_311_579e307),
        ];
        for (x, expected) in references {
            assert_close(chi_squared_1df_log_sf(x), expected, 4);
        }
        assert_eq!(chi_squared_1df_log_sf(0.0), 0.0);
        assert_eq!(chi_squared_1df_log_sf(f64::INFINITY), f64::NEG_INFINITY);
        assert!(chi_squared_1df_log_sf(f64::NAN).is_nan());
    }

    /// Against 50-digit references on 3,500 points, the largest relative error was 2e-13. The error
    /// occurs at a small tail probability `Q`, where `chi_squared_sf` returns `exp(ln Q)`. The
    /// relative error of `Q` is then `|ln Q|` times the relative error of `ln Q`.
    #[test]
    fn chi_squared_sf_matches_references_for_integer_df() {
        let references = [
            (2, 0.5, 0.778_800_783_071_404_9, -0.25),
            (2, 10.0, 0.006_737_946_999_085_467, -5.0),
            (2, 1_500.0, 0.0, -750.0),
            (3, 1.0, 0.801_251_956_901_200_8, -0.221_579_828_439_848_66),
            (3, 3.0, 0.391_625_176_271_088_95, -0.937_450_079_608_155_3),
            (3, 100.0, 1.554_159_431_389_605e-21, -47.913_352_111_990_12),
            (10, 5.0, 0.891_178_018_914_151_3, -0.115_211_074_706_929_97),
            (10, 10.0, 0.440_493_285_065_212_4, -0.819_860_077_614_913),
            (
                10,
                30.0,
                0.000_856_641_210_775_300_4,
                -7.062_491_384_232_906_5,
            ),
            (
                10,
                1_450.0,
                1.585_042_144_501_453e-305,
                -701.827_842_366_617_3,
            ),
            (25, 20.0, 0.746_825_306_016_537, -0.291_923_981_953_311_3),
            (25, 25.0, 0.462_373_662_926_613_7, -0.771_381_920_584_738_4),
            (
                25,
                100.0,
                6.274_266_201_376_257e-11,
                -23.491_979_484_774_394,
            ),
            (25, 10_000.0, 0.0, -4_920.784_323_623_525),
        ];
        for (df, x, sf, log_sf) in references {
            for (actual, expected) in [
                (chi_squared_sf(x, df), sf),
                (chi_squared_log_sf(x, df), log_sf),
            ] {
                assert!(
                    (actual - expected).abs() <= 1e-12 * expected.abs(),
                    "df {df}, x {x}: got {actual:e}, expected {expected:e}"
                );
            }
        }
    }
}
