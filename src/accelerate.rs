//! Thin bindings to Apple Accelerate.
#![expect(unsafe_code)]

use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{MatMut, MatRef};
use fearless_simd::{Level, f64x2, prelude::*};

// CBLAS enum values.
const ROW_MAJOR: i32 = 101;
const COL_MAJOR: i32 = 102;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;
const LOWER: i32 = 122;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[link_name = "cblas_dsyrk$NEWLAPACK"]
    fn dsyrk(
        order: i32,
        uplo: i32,
        trans: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );

    #[link_name = "cblas_dgemm$NEWLAPACK"]
    fn dgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );

    #[link_name = "cblas_dgemv$NEWLAPACK"]
    fn dgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );

    #[link_name = "dpotrf$NEWLAPACK"]
    fn dpotrf(
        uplo: *const std::ffi::c_char,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        info: *mut i32,
    );

    fn vvexp(out: *mut f64, input: *const f64, count: *const i32);

    fn vvlog1p(out: *mut f64, input: *const f64, count: *const i32);
}

fn to_blas_int(value: usize) -> i32 {
    i32::try_from(value).expect("BLAS dimension or stride exceeds i32::MAX")
}

/// Returns the column stride to pass as a leading dimension. The function uses `col_stride()`
/// rather than `nrows()` because faer pads column strides for alignment.
fn leading_dimension(matrix: MatRef<'_, f64>) -> i32 {
    assert_eq!(matrix.row_stride(), 1, "matrix must be column-major");
    let stride = usize::try_from(matrix.col_stride()).unwrap();
    assert!(
        stride >= matrix.nrows(),
        "column stride must be at least the row count"
    );
    // BLAS requires lda >= max(1, rows) even for empty matrices.
    to_blas_int(stride.max(1))
}

/// Adds the lower triangle of `a @ a.T` to `out` for a `(k, n)` matrix `a`.
/// Only the lower triangle of `out` is written.
#[expect(clippy::needless_pass_by_value)]
pub(crate) fn a_mul_a_transpose_lower(a: MatRef<'_, f64>, out: MatMut<'_, f64>) {
    let (k, n) = a.shape();
    assert_eq!(out.shape(), (k, k));
    let (lda, ldc) = (leading_dimension(a), leading_dimension(out.rb()));
    // SAFETY: The assert above checks that `out` is `k` by `k`. `leading_dimension` checks that `a`
    // and `out` are column-major and that each leading dimension is at least the row count. The
    // dsyrk call reads only the `k` by `n` matrix `a`. It writes only the lower triangle of `out`.
    // The mutable borrow `out` cannot overlap `a`.
    unsafe {
        dsyrk(
            COL_MAJOR,
            LOWER,
            NO_TRANS,
            to_blas_int(k),
            to_blas_int(n),
            1.0,
            a.as_ptr(),
            lda,
            1.0,
            out.as_ptr_mut(),
            ldc,
        );
    }
}

/// Writes `sqrt(weights[i]) * x.row(i)` to column `i` of `packed`.
///
/// When `x` is column-major, samples are packed in pairs by transposing 2×2
/// blocks. Other layouts are copied one row at a time.
pub(crate) fn pack_scaled_rows(x: MatRef<'_, f64>, weights: &[f64], packed: MatMut<'_, f64>) {
    let (samples, k) = x.shape();
    assert_eq!(packed.shape(), (k, samples));
    let mut packed = packed.try_as_col_major_mut().unwrap();
    let mut i = 0;
    if let Some(x) = x.try_as_col_major() {
        let simd = Level::new().as_neon().unwrap();
        while i + 1 < samples {
            let (root0, root1) = (weights[i].sqrt(), weights[i + 1].sqrt());
            let (out0, out1) = packed.rb_mut().two_cols_mut(i, i + 1);
            let (out0, out1) = (out0.as_slice_mut(), out1.as_slice_mut());
            let mut j = 0;
            while j + 1 < k {
                let col_j = f64x2::from_slice(simd, &x.col(j).as_slice()[i..i + 2]);
                let col_j1 = f64x2::from_slice(simd, &x.col(j + 1).as_slice()[i..i + 2]);
                (col_j.zip_low(col_j1) * root0).store_slice(&mut out0[j..j + 2]);
                (col_j.zip_high(col_j1) * root1).store_slice(&mut out1[j..j + 2]);
                j += 2;
            }
            if j < k {
                out0[j] = root0 * x[(i, j)];
                out1[j] = root1 * x[(i + 1, j)];
            }
            i += 2;
        }
    }
    for (i, &weight) in weights.iter().enumerate().skip(i) {
        let root = weight.sqrt();
        for (value, &x) in packed.rb_mut().col_mut(i).iter_mut().zip(x.row(i).iter()) {
            *value = root * x;
        }
    }
}

/// Writes `a @ b` to `out` for a column-major `b`. The matrix `a` can be row-major or column-major.
/// The matrix `out` must have the same layout as `a`.
#[expect(clippy::needless_pass_by_value)]
pub(crate) fn a_mul_b(a: MatRef<'_, f64>, b: MatRef<'_, f64>, out: MatMut<'_, f64>) {
    let (m, k) = a.shape();
    assert_eq!(b.shape(), (k, k));
    assert_eq!(out.shape(), (m, k));
    // For row-major `a`, call dgemm in `ROW_MAJOR` order; dgemm then views column-major `b` as
    // `b.T`, so pass `TRANS` for `b`.
    let (order, trans_b, lda, ldc) = if a.row_stride() == 1 {
        (
            COL_MAJOR,
            NO_TRANS,
            leading_dimension(a),
            leading_dimension(out.rb()),
        )
    } else {
        (
            ROW_MAJOR,
            TRANS,
            leading_dimension(a.transpose()),
            leading_dimension(out.rb().transpose()),
        )
    };
    let ldb = leading_dimension(b);
    // SAFETY: The asserts above check the shapes of `b` and `out`. `leading_dimension` checks that
    // `b` is column-major. It also checks that `a` and `out` are column-major when `order` is
    // `COL_MAJOR` and row-major when `order` is `ROW_MAJOR`. Each leading dimension is at least the
    // number of elements in one contiguous column or row of its matrix. The mutable borrow `out`
    // cannot overlap `a` or `b`.
    unsafe {
        dgemm(
            order,
            NO_TRANS,
            trans_b,
            to_blas_int(m),
            to_blas_int(k),
            to_blas_int(k),
            1.0,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            0.0,
            out.as_ptr_mut(),
            ldc,
        );
    }
}

/// Writes `a @ x` to `out`.
pub(crate) fn a_mul_x(a: MatRef<'_, f64>, x: &[f64], out: &mut [f64]) {
    assert_eq!((out.len(), x.len()), a.shape());
    gemv(NO_TRANS, a, x, out);
}

/// Writes `a.T @ x` to `out`.
pub(crate) fn a_transpose_mul_x(a: MatRef<'_, f64>, x: &[f64], out: &mut [f64]) {
    assert_eq!((x.len(), out.len()), a.shape());
    gemv(TRANS, a, x, out);
}

fn gemv(trans: i32, a: MatRef<'_, f64>, x: &[f64], out: &mut [f64]) {
    let (m, n) = a.shape();
    let lda = leading_dimension(a);
    // SAFETY: `leading_dimension` checks that `a` is column-major with a leading dimension of at
    // least `m`. Both callers assert the lengths of `x` and `out` for their value of `trans`. The
    // mutable borrow `out` cannot overlap `a` or `x`.
    unsafe {
        dgemv(
            COL_MAJOR,
            trans,
            to_blas_int(m),
            to_blas_int(n),
            1.0,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            1,
            0.0,
            out.as_mut_ptr(),
            1,
        );
    }
}

/// Factors the lower triangle in place. LAPACK `dpotrf` neither reads nor writes the upper
/// triangle.
#[expect(clippy::needless_pass_by_value)]
pub(crate) fn cholesky_lower_in_place(lower: MatMut<'_, f64>) -> crate::Result<()> {
    let n = lower.nrows();
    assert_eq!(lower.ncols(), n);
    let count = to_blas_int(n);
    let lda = leading_dimension(lower.rb());
    let mut info = 0;
    // SAFETY: The assert above checks that `lower` is square. `leading_dimension` checks that
    // `lower` is column-major and that its leading dimension is at least `n`. LAPACK `dpotrf` then
    // accesses only the lower triangle of the `n` by `n` matrix. The integer arguments are `i32`
    // because `dpotrf$NEWLAPACK` is the LP64 interface.
    unsafe {
        dpotrf(
            b"L".as_ptr().cast(),
            &raw const count,
            lower.as_ptr_mut(),
            &raw const lda,
            &raw mut info,
        );
    }
    if info == 0 {
        Ok(())
    } else {
        Err(crate::Error::NotPositiveDefinite)
    }
}

pub(crate) fn exp_in_place(values: &mut [f64]) {
    let count = to_blas_int(values.len());
    let values = values.as_mut_ptr();
    // SAFETY: `values` is valid for reads and writes of `count` elements. The C declaration of
    // vvexp doesn't have a `restrict` qualifier, so the input and output can alias the same array.
    unsafe { vvexp(values, values, &raw const count) }
}

/// Writes `log(1 + input)` elementwise to `out`.
pub(crate) fn log1p(input: &[f64], out: &mut [f64]) {
    assert_eq!(input.len(), out.len());
    let count = to_blas_int(input.len());
    // SAFETY: The assert above checks that `input` and `out` both have `count` elements. The
    // mutable borrow `out` cannot overlap `input`.
    unsafe { vvlog1p(out.as_mut_ptr(), input.as_ptr(), &raw const count) }
}

#[cfg(test)]
mod tests {
    use faer::linalg::solvers::Llt;
    use faer::{Mat, Side, mat};

    use super::*;

    #[test]
    fn cholesky_lower_matches_faer() {
        let matrix = Mat::from_fn(12, 12, |i, j| {
            if i == j {
                16.0
            } else {
                1.0 / (i + j + 1) as f64
            }
        });
        let expected = Llt::new(matrix.as_ref(), Side::Lower).unwrap();
        let mut factor = matrix.clone();
        cholesky_lower_in_place(factor.as_mut()).unwrap();
        let tolerance = 1e-12 * expected.L().norm_max();
        for i in 0..12 {
            for j in 0..=i {
                assert!((factor[(i, j)] - expected.L()[(i, j)]).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn cholesky_lower_rejects_indefinite() {
        let mut matrix = mat![[1.0, 2.0], [2.0, 1.0]];
        assert!(matches!(
            cholesky_lower_in_place(matrix.as_mut()),
            Err(crate::Error::NotPositiveDefinite)
        ));
    }

    #[test]
    fn a_mul_a_transpose_lower_streams_blocks() {
        let a = mat![
            [1.0, 1.0, 0.0, 2.0, 1.0],
            [0.5, -0.3, 1.7, 0.2, -1.1],
            [-1.0, 2.0, 0.4, -0.6, 1.3]
        ];
        // dsyrk adds to the lower triangle of `out` and never writes the upper triangle, so
        // initialize with 99s and verify both triangles.
        let mut out = Mat::from_fn(3, 3, |_, _| 99.0);
        a_mul_a_transpose_lower(a.as_ref().subcols(0, 2), out.as_mut());
        a_mul_a_transpose_lower(a.as_ref().subcols(2, 3), out.as_mut());
        for r in 0..3 {
            for s in 0..3 {
                if s > r {
                    assert_eq!(out[(r, s)], 99.0, "({r},{s})");
                } else {
                    let expected: f64 = 99.0 + (0..5).map(|i| a[(r, i)] * a[(s, i)]).sum::<f64>();
                    assert!(
                        (out[(r, s)] - expected).abs() < 1e-12,
                        "({r},{s}): got {}, expected {expected}",
                        out[(r, s)]
                    );
                }
            }
        }
    }
}
