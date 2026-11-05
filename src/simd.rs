//! SIMD arithmetic and elementary functions.

use std::f64::consts::{LN_2, LOG2_E, SQRT_2};

use fearless_simd::prelude::*;
#[cfg(not(accelerate))]
use fearless_simd::{Level, dispatch};

/// Loads a vector from `values`, zero-padded past the end of the slice.
#[inline(always)]
pub(crate) fn load<S: Simd>(simd: S, values: &[f64]) -> S::f64s {
    if values.len() >= S::f64s::LEN {
        S::f64s::from_slice(simd, &values[..S::f64s::LEN])
    } else {
        S::f64s::from_fn(simd, |lane| values.get(lane).copied().unwrap_or(0.0))
    }
}

/// Stores the leading lanes of `value` into `out`.
#[inline(always)]
pub(crate) fn store<S: Simd>(value: S::f64s, out: &mut [f64]) {
    if out.len() >= S::f64s::LEN {
        value.store_slice(&mut out[..S::f64s::LEN]);
    } else {
        // LLVM compiles a copy of `out.len()` elements to a memcpy call, even when the copy is a
        // loop over `out`. A call inside a kernel loop slows the loop, because a call preserves at
        // most the low 64 bits of each vector register. LLVM unrolls this loop instead, because
        // the loop has one iteration per lane and a conditional store.
        for (i, &lane) in value.as_slice().iter().enumerate() {
            if let Some(element) = out.get_mut(i) {
                *element = lane;
            }
        }
    }
}

/// `sum(a * b)`.
#[inline(always)]
pub(crate) fn dot<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    let mut a = a.chunks_exact(S::f64s::LEN);
    let mut b = b.chunks_exact(S::f64s::LEN);
    let mut sum = S::f64s::splat(simd, 0.0);
    for (a, b) in a.by_ref().zip(b.by_ref()) {
        sum = S::f64s::from_slice(simd, a).mul_add(S::f64s::from_slice(simd, b), sum);
    }
    let tail: f64 = a
        .remainder()
        .iter()
        .zip(b.remainder())
        .map(|(a, b)| a * b)
        .sum();
    sum.reduce_sum() + tail
}

/// `y += a * x`.
#[inline(always)]
pub(crate) fn axpy<S: Simd>(simd: S, y: &mut [f64], a: f64, x: &[f64]) {
    let mut y = y.chunks_exact_mut(S::f64s::LEN);
    let mut x = x.chunks_exact(S::f64s::LEN);
    for (y, x) in y.by_ref().zip(x.by_ref()) {
        S::f64s::from_slice(simd, x)
            .mul_add(a, S::f64s::from_slice(simd, y))
            .store_slice(y);
    }
    for (y, x) in y.into_remainder().iter_mut().zip(x.remainder()) {
        *y += a * x;
    }
}

/// `exp(a)` for `a <= 0`.
#[inline(always)]
pub(crate) fn exp_nonpositive<S: Simd>(simd: S, a: S::f64s) -> S::f64s {
    // Adding 1.5 * 2^52 encodes the rounded integer in the mantissa.
    // Subtracting STEP_BASE gives the step count plus 4096.
    const ROUND_SHIFT: f64 = 6_755_399_441_055_744.0;
    const STEP_BASE: u64 = ROUND_SHIFT.to_bits() - 4096;
    const HALF_LN2_HI: f64 = 6.931_471_803_691_238e-1 / 2.0;
    const HALF_LN2_LO: f64 = 1.908_214_929_270_587_7e-10 / 2.0;
    // Taylor coefficients 1/10! .. 1/1!, evaluated by Horner's rule.
    const TAYLOR: [f64; 10] = [
        1.0 / 3_628_800.0,
        1.0 / 362_880.0,
        1.0 / 40_320.0,
        1.0 / 5_040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        1.0 / 2.0,
        1.0,
    ];
    // Below -746 the result is zero.
    let a = a.clamp(-746.0, 0.0);
    let shifted = a.mul_add(2.0 * LOG2_E, ROUND_SHIFT);
    let steps = shifted - ROUND_SHIFT;
    let r = steps.mul_add(-HALF_LN2_HI, a);
    let r = steps.mul_add(-HALF_LN2_LO, r);
    let mut p = S::f64s::splat(simd, TAYLOR[0]);
    for &coefficient in &TAYLOR[1..] {
        p = p.mul_add(r, coefficient);
    }
    let p = p.mul_add(r, 1.0);

    // Each half exponent includes an offset of 1024.
    // Subtract 1 to get f64's exponent bias of 1023.
    let steps = shifted.bitcast::<S::u64s>() - STEP_BASE;
    let odd = (steps & 1).simd_eq(1);
    let k = steps >> 1;
    let half = k >> 1;
    let scale = |exponent: S::u64s| ((exponent - 1) << 52).bitcast::<S::f64s>();
    let p = odd.select(p * SQRT_2, p);
    // Split the scaling so only the final multiplication can underflow.
    p * scale(half) * scale(k - half)
}

/// `ln(1 + z)` for `z` in `[0, 1]`.
#[inline(always)]
#[cfg_attr(
    all(accelerate, not(test)),
    expect(dead_code, reason = "Accelerate builds call vvlog1p instead")
)]
pub(crate) fn log1p_unit<S: Simd>(simd: S, z: S::f64s) -> S::f64s {
    // The atanh series coefficients 1/21 .. 1/3, evaluated by Horner's rule in s^2.
    const SERIES: [f64; 10] = [
        1.0 / 21.0,
        1.0 / 19.0,
        1.0 / 17.0,
        1.0 / 15.0,
        1.0 / 13.0,
        1.0 / 11.0,
        1.0 / 9.0,
        1.0 / 7.0,
        1.0 / 5.0,
        1.0 / 3.0,
    ];
    // When 1 + z is above sqrt(2), halve it to keep the atanh argument small. Then add ln(2) to the
    // result.
    let fold = z.simd_gt(SQRT_2 - 1.0);
    let s = fold.select(z - 1.0, z) / fold.select(z + 3.0, z + 2.0);
    let offset = fold.select(S::f64s::splat(simd, LN_2), S::f64s::splat(simd, 0.0));
    let s2 = s * s;
    let mut series = S::f64s::splat(simd, SERIES[0]);
    for &coefficient in &SERIES[1..] {
        series = series.mul_add(s2, coefficient);
    }
    s * 2.0 * series.mul_add(s2, 1.0) + offset
}

/// Computes `exp(value - shift)` in place. Requires `value <= shift`.
pub(crate) fn exp_shifted(values: &mut [f64], shifts: &[f64]) {
    #[cfg(accelerate)]
    {
        for (value, shift) in values.iter_mut().zip(shifts) {
            *value -= shift;
        }
        crate::accelerate::exp_in_place(values);
    }
    #[cfg(not(accelerate))]
    dispatch!(Level::new(), simd => exp_shifted_lanes(simd, values, shifts));
}

#[cfg(not(accelerate))]
#[inline(always)]
fn exp_shifted_lanes<S: Simd>(simd: S, values: &mut [f64], shifts: &[f64]) {
    for i in (0..values.len()).step_by(S::f64s::LEN) {
        let len = S::f64s::LEN.min(values.len() - i);
        let value = exp_nonpositive(simd, load(simd, &values[i..]) - load(simd, &shifts[i..]));
        store::<S>(value, &mut values[i..i + len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline(always)]
    fn exp_lanes<S: Simd>(simd: S, a: f64) -> f64 {
        exp_nonpositive(simd, S::f64s::splat(simd, a)).as_slice()[0]
    }

    #[inline(always)]
    fn log1p_lanes<S: Simd>(simd: S, z: f64) -> f64 {
        log1p_unit(simd, S::f64s::splat(simd, z)).as_slice()[0]
    }

    fn exp_lane(a: f64) -> f64 {
        fearless_simd::dispatch!(fearless_simd::Level::new(), simd => exp_lanes(simd, a))
    }

    fn log1p_lane(z: f64) -> f64 {
        fearless_simd::dispatch!(fearless_simd::Level::new(), simd => log1p_lanes(simd, z))
    }

    fn ulps(observed: f64, expected: f64) -> f64 {
        (observed - expected).abs() / (expected.next_up() - expected)
    }

    #[test]
    fn exp_nonpositive_within_two_ulp_of_libm() {
        let check = |a: f64| {
            let error = ulps(exp_lane(a), a.exp());
            assert!(error <= 2.0, "a = {a}: error {error} ulp");
        };
        for step in 0..=2_000_000 {
            let a = -746.0 * f64::from(step) / 2_000_000.0;
            check(a);
        }
        // Sample more densely near underflow and zero.
        for step in 0..=200_000 {
            let a = -708.0 - 38.0 * f64::from(step) / 200_000.0;
            check(a);
        }
        for step in 0..=200_000 {
            let a = -2.0 * LN_2 * f64::from(step) / 200_000.0;
            check(a);
        }
        for a in [
            0.0,
            -0.0,
            -1e-300,
            -1e-17,
            -1e-8,
            -LN_2 / 4.0,
            -LN_2 / 2.0,
            -LN_2,
            -1.0,
            -20.0,
            -700.0,
            -708.396_418_532_264_1,
            -709.0,
            -745.0,
            -745.133_219_101_941_1,
            -746.0,
            -800.0,
            -1e300,
            f64::MIN,
        ] {
            check(a);
        }
        assert_eq!(exp_lane(f64::NEG_INFINITY), 0.0);
        assert!(exp_lane(f64::NAN).is_nan());
        assert_eq!(exp_lane(0.0), 1.0);
    }

    #[test]
    fn log1p_unit_within_four_ulp_of_libm() {
        let check = |z: f64| {
            let error = ulps(log1p_lane(z), z.ln_1p());
            assert!(error <= 4.0, "z = {z}: error {error} ulp");
        };
        for step in 0..=2_000_000 {
            let z = f64::from(step) / 2_000_000.0;
            check(z);
        }
        for z in [
            0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE.next_down(),
            f64::MIN_POSITIVE,
            1e-300,
            1e-16,
            1e-8,
            0.001,
            SQRT_2 - 1.0,
            0.5,
            0.999_999,
            1.0,
        ] {
            check(z);
        }
        assert_eq!(log1p_lane(0.0), 0.0);
        assert!(log1p_lane(f64::NAN).is_nan());
    }
}
