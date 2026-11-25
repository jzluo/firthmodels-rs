use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::{llt, llt_pivoting};
use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{self, BlockStructure};
use faer::linalg::solvers::{ColPivQr, PartialPivLu, Solve, SolveLstsq, Svd};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{Accum, ColMut, Mat, MatMut, MatRef, Par, Spec};
use fearless_simd::{Level, dispatch};

use crate::{Error, Result};

/// Solves a symmetric positive-definite linear system using a Cholesky factorization.
pub(crate) fn spd_solve(matrix: &Mat<f64>, rhs: &[f64]) -> Result<Vec<f64>> {
    let lower = cholesky_factor_logdet(matrix)?.lower;
    let mut solution = rhs.to_vec();
    let mut column = ColMut::from_slice_mut(&mut solution).as_mat_mut();
    solve_lower_triangular_in_place(lower.as_ref(), column.rb_mut(), Par::Seq);
    solve_upper_triangular_in_place(lower.transpose(), column, Par::Seq);
    Ok(solution)
}

/// Result of `cholesky_inverse_logdet` and `pivoted_cholesky_inverse_logdet`.
pub(crate) struct CholeskySummary {
    pub(crate) inverse: Mat<f64>,
    pub(crate) log_determinant: f64,
    pub(crate) minimum_pivot: f64,
}

/// Result of `cholesky_factor_logdet`.
pub(crate) struct CholeskyFactor {
    // Only the lower triangle contains the factor.
    lower: Mat<f64>,
    /// The inverse of the factored matrix. `quadratic_form_rows` computes it on its first call.
    #[cfg(accelerate)]
    cached_inverse: std::cell::OnceCell<Mat<f64>>,
    pub(crate) log_determinant: f64,
    pub(crate) minimum_pivot: f64,
    /// The smallest pivot of the factor of `D A D`, where `D = diag(A)^(-1/2)`.
    pub(crate) minimum_relative_pivot: f64,
}

/// Reusable matrix buffers. Kernels overwrite the parts they use.
pub(crate) struct MatrixScratch {
    rows: Mat<f64>,
    /// Packed samples, one per column. The column stride is a multiple of 8. Allocating a zeroed
    /// `Vec` is faster than `Mat::zeros`, which writes every zero.
    #[cfg(accelerate)]
    packed: Vec<f64>,
    #[cfg(accelerate)]
    product: Vec<f64>,
    /// Scaled chunk of `x` in `add_weighted_gram_faer` when `x` is row-major.
    row_major: Vec<f64>,
}

impl Default for MatrixScratch {
    fn default() -> Self {
        Self {
            rows: Mat::new(),
            #[cfg(accelerate)]
            packed: Vec::new(),
            #[cfg(accelerate)]
            product: Vec::new(),
            row_major: Vec::new(),
        }
    }
}

impl MatrixScratch {
    fn rows(&mut self, n: usize, k: usize) -> MatMut<'_, f64> {
        if self.rows.ncols() != k || self.rows.nrows() < n {
            self.rows = Mat::zeros(n, k);
        }
        self.rows.as_mut().subrows_mut(0, n)
    }

    fn row_major_rows(&mut self, n: usize, k: usize) -> MatMut<'_, f64> {
        if self.row_major.len() < n * k {
            self.row_major = vec![0.0; n * k];
        }
        MatMut::from_row_major_slice_mut(&mut self.row_major[..n * k], n, k)
    }

    #[cfg(accelerate)]
    fn packed(&mut self, k: usize, n: usize) -> MatMut<'_, f64> {
        // On an M5, fits took about 1% longer when samples were not padded to a multiple of 8.
        let stride = k.next_multiple_of(8);
        let n = n.min((MATRIX_SCRATCH_BYTES / (stride * 8)).max(1));
        if self.packed.len() < stride * n {
            self.packed = vec![0.0; stride * n];
        }
        MatMut::from_column_major_slice_with_stride_mut(
            &mut self.packed[..stride * n],
            k,
            n,
            stride,
        )
    }
}

impl CholeskyFactor {
    fn inverse(&self) -> Mat<f64> {
        let n = self.lower.nrows();
        let mut out = Mat::zeros(n, n);
        let mut scratch = MemBuffer::new(llt::inverse::inverse_scratch::<f64>(n, Par::Seq));
        llt::inverse::inverse(
            out.as_mut(),
            self.lower.as_ref(),
            Par::Seq,
            MemStack::new(&mut scratch),
        );
        copy_lower_to_upper(&mut out);
        out
    }

    /// Computes the squared norm of `L^-1 @ x[i, :].T` for each row of `x`.
    pub(crate) fn quadratic_form_rows(
        &self,
        x: MatRef<f64>,
        values: &mut [f64],
        scratch: &mut MatrixScratch,
    ) {
        debug_assert_eq!(values.len(), x.nrows());

        #[cfg(accelerate)]
        if x.ncols() >= ACCELERATE_MIN_COLS {
            let inverse = self.cached_inverse.get_or_init(|| self.inverse());
            return quadratic_form_rows(x, inverse, values, scratch);
        }

        let lower = self.lower.as_ref();
        dispatch!(Level::new(), _ => solved_square_norms(lower, x, values, scratch));
    }
}

/// Computes the squared norm of `lower^-1 @ x[i, :].T` for each row of `x` with faer.
///
/// A portable build compiles the loops of the crate for the baseline target, which is SSE2 on
/// x86-64. This function and `add_weighted_gram_faer` run inside `dispatch!`, which compiles their
/// loops for the SIMD level of the CPU.
#[inline(always)]
fn solved_square_norms(
    lower: MatRef<f64>,
    x: MatRef<f64>,
    values: &mut [f64],
    scratch: &mut MatrixScratch,
) {
    let (n, k) = x.shape();
    let chunk_rows = ROW_CHUNK.min(n);
    let mut solved = scratch.rows(chunk_rows, k);
    let mut start = 0;
    while start < n {
        let rows = chunk_rows.min(n - start);
        let mut solved_chunk = solved.rb_mut().subrows_mut(0, rows);
        let x_chunk = x.subrows(start, rows);
        // Copying each column with `copy_from_slice` is faster than faer's `copy_from` on these
        // chunks.
        if let Some(x_chunk) = x_chunk.try_as_col_major() {
            let mut solved_chunk = solved_chunk.rb_mut().try_as_col_major_mut().unwrap();
            for j in 0..k {
                solved_chunk
                    .rb_mut()
                    .col_mut(j)
                    .as_slice_mut()
                    .copy_from_slice(x_chunk.col(j).as_slice());
            }
        } else {
            solved_chunk.copy_from(x_chunk);
        }
        // Solve `L Z' = X'`. Then accumulate the squared row norms of `Z` one column at a time.
        solve_lower_triangular_in_place(lower, solved_chunk.rb_mut().transpose_mut(), Par::Seq);
        let solved_chunk = solved_chunk.rb();
        column_dot_products(solved_chunk, solved_chunk, &mut values[start..start + rows]);
        start += rows;
    }
}

/// Writes the dot product of row `i` of `left` and row `i` of `right` to `values[i]` for
/// column-major matrices. The products of each row are added in column order.
#[inline(always)]
fn column_dot_products(left: MatRef<f64>, right: MatRef<f64>, values: &mut [f64]) {
    fn column(matrix: MatRef<'_, f64>, j: usize, rows: usize) -> &[f64] {
        &matrix.col(j).try_as_col_major().unwrap().as_slice()[..rows]
    }
    let rows = values.len();
    values.fill(0.0);
    let mut j = 0;
    while j + 4 <= left.ncols() {
        let [a0, a1, a2, a3] = std::array::from_fn(|offset| column(left, j + offset, rows));
        let [b0, b1, b2, b3] = std::array::from_fn(|offset| column(right, j + offset, rows));
        for i in 0..rows {
            values[i] = values[i] + a0[i] * b0[i] + a1[i] * b1[i] + a2[i] * b2[i] + a3[i] * b3[i];
        }
        j += 4;
    }
    for j in j..left.ncols() {
        for ((value, a), b) in values
            .iter_mut()
            .zip(column(left, j, rows))
            .zip(column(right, j, rows))
        {
            *value += a * b;
        }
    }
}

fn cholesky_lower_in_place(lower: MatMut<'_, f64>) -> Result<()> {
    #[cfg(accelerate)]
    return crate::accelerate::cholesky_lower_in_place(lower);

    #[cfg(not(accelerate))]
    {
        let mut scratch = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
            lower.nrows(),
            Par::Seq,
            Spec::default(),
        ));
        llt::factor::cholesky_in_place(
            lower,
            llt::factor::LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut scratch),
            Spec::default(),
        )
        .map_err(|_| Error::NotPositiveDefinite)?;
        Ok(())
    }
}

pub(crate) fn cholesky_factor_logdet(matrix: &Mat<f64>) -> Result<CholeskyFactor> {
    let n = check_square_finite(matrix)?;
    let mut lower = matrix.clone();
    cholesky_lower_in_place(lower.as_mut())?;
    let mut log_determinant = 0.0;
    let mut minimum_pivot = f64::INFINITY;
    let mut minimum_relative_pivot = f64::INFINITY;
    for i in 0..n {
        let pivot = lower[(i, i)];
        if !pivot.is_finite() || pivot <= 0.0 {
            return Err(Error::NotPositiveDefinite);
        }
        log_determinant += 2.0 * pivot.ln();
        minimum_pivot = minimum_pivot.min(pivot);
        minimum_relative_pivot = minimum_relative_pivot.min(pivot / matrix[(i, i)].sqrt());
    }
    Ok(CholeskyFactor {
        lower,
        #[cfg(accelerate)]
        cached_inverse: std::cell::OnceCell::new(),
        log_determinant,
        minimum_pivot,
        minimum_relative_pivot,
    })
}

/// Returns true when `matrix` has no correct digits in some direction.
///
/// `matrix` is a Gram matrix minus another term, and `gram_diagonal` is the diagonal of that Gram
/// matrix. Rounding in the subtraction is about `eps` times the Gram matrix, so the check scales
/// `matrix` by `gram_diagonal` before it tests the Cholesky pivots.
pub(crate) fn has_cancelled_direction(matrix: &Mat<f64>, gram_diagonal: &[f64]) -> bool {
    if gram_diagonal.iter().any(|&value| value <= 0.0) {
        return true;
    }
    let scale: Vec<f64> = gram_diagonal
        .iter()
        .map(|value| value.sqrt().recip())
        .collect();
    let k = scale.len();
    let scaled = Mat::from_fn(k, k, |r, s| matrix[(r, s)] * scale[r] * scale[s]);
    let pivot_cutoff = k as f64 * f64::EPSILON;
    cholesky_factor_logdet(&scaled).map_or(true, |factor| {
        factor.minimum_pivot * factor.minimum_pivot <= pivot_cutoff
    })
}

pub(crate) fn cholesky_inverse_logdet(matrix: &Mat<f64>) -> Result<CholeskySummary> {
    let factor = cholesky_factor_logdet(matrix)?;
    Ok(CholeskySummary {
        inverse: factor.inverse(),
        log_determinant: factor.log_determinant,
        minimum_pivot: factor.minimum_pivot,
    })
}

/// Returns the ratio of the smallest to the largest pivot of the pivoted Cholesky factor of
/// `matrix` after `matrix` is scaled to a unit diagonal. Returns 0 when an entry of `matrix` is not
/// finite or when `matrix` is not numerically positive definite.
pub(crate) fn pivoted_cholesky_pivot_ratio(matrix: &Mat<f64>) -> f64 {
    let Ok(n) = check_square_finite(matrix) else {
        return 0.0;
    };
    if (0..n).any(|i| matrix[(i, i)] <= 0.0) {
        return 0.0;
    }
    let scale: Vec<f64> = (0..n).map(|i| matrix[(i, i)].sqrt().recip()).collect();
    let mut lower = Mat::from_fn(n, n, |i, j| matrix[(i, j)] * scale[i] * scale[j]);
    let mut permutation = vec![0usize; n];
    let mut inverse_permutation = vec![0usize; n];
    let mut scratch = MemBuffer::new(
        llt_pivoting::factor::cholesky_in_place_scratch::<usize, f64>(n, Par::Seq, Spec::default()),
    );
    match llt_pivoting::factor::cholesky_in_place(
        lower.as_mut(),
        &mut permutation,
        &mut inverse_permutation,
        Par::Seq,
        MemStack::new(&mut scratch),
        Spec::default(),
    ) {
        // The rank that faer returns can include a zero pivot.
        Ok((info, _)) if info.rank == n && (0..n).all(|i| lower[(i, i)] > 0.0) => {
            let pivots = (0..n).map(|i| lower[(i, i)]);
            pivots.clone().fold(f64::INFINITY, f64::min) / pivots.fold(0.0, f64::max)
        },
        _ => 0.0,
    }
}

/// The rank test uses the `dpstrf` cutoff from firthmodels-py: `n * eps * max(abs(diag(matrix)))`.
/// The function returns `RankDeficient` when the square of a pivot of the pivoted Cholesky factor
/// is at most that cutoff.
pub(crate) fn pivoted_cholesky_inverse_logdet(matrix: &Mat<f64>) -> Result<CholeskySummary> {
    let n = check_square_finite(matrix)?;
    let diagonal_scale = (0..n).map(|i| matrix[(i, i)].abs()).fold(0.0_f64, f64::max);
    if diagonal_scale == 0.0 {
        return Err(Error::RankDeficient {
            rank: 0,
            required: n,
        });
    }

    let factor_diagonal_cutoff = (n as f64 * f64::EPSILON * diagonal_scale).sqrt();

    let mut lower = matrix.clone();
    let mut permutation = vec![0usize; n];
    let mut inverse_permutation = vec![0usize; n];
    let factor_scratch =
        llt_pivoting::factor::cholesky_in_place_scratch::<usize, f64>(n, Par::Seq, Spec::default());
    let inverse_scratch = llt_pivoting::inverse::inverse_scratch::<usize, f64>(n, Par::Seq);
    let mut scratch = MemBuffer::new(factor_scratch.or(inverse_scratch));
    let (info, permutation) = llt_pivoting::factor::cholesky_in_place(
        lower.as_mut(),
        &mut permutation,
        &mut inverse_permutation,
        Par::Seq,
        MemStack::new(&mut scratch),
        Spec::default(),
    )
    .map_err(|_| Error::NotPositiveDefinite)?;
    if info.rank < n {
        return Err(Error::RankDeficient {
            rank: info.rank,
            required: n,
        });
    }

    let mut log_determinant = 0.0;
    let mut minimum_pivot = f64::INFINITY;
    for i in 0..n {
        let pivot = lower[(i, i)];
        if !pivot.is_finite() || pivot <= 0.0 {
            return Err(Error::NotPositiveDefinite);
        }
        if pivot <= factor_diagonal_cutoff {
            return Err(Error::RankDeficient {
                rank: i,
                required: n,
            });
        }
        log_determinant += 2.0 * pivot.ln();
        minimum_pivot = minimum_pivot.min(pivot);
    }

    let mut inverse = Mat::zeros(n, n);
    llt_pivoting::inverse::inverse(
        inverse.as_mut(),
        lower.as_ref(),
        permutation,
        Par::Seq,
        MemStack::new(&mut scratch),
    );
    // faer writes the lower triangle in the original coordinate order.
    copy_lower_to_upper(&mut inverse);

    Ok(CholeskySummary {
        inverse,
        log_determinant,
        minimum_pivot,
    })
}

/// Returns the absolute diagonal of `R` from the column-pivoted QR of the `nrows` by `ncols` matrix
/// with entries `entry(i, j)`, after each column is scaled to unit norm. Scaling makes the pivots
/// independent of the units of the columns. A zero column stays zero.
pub(crate) fn normalized_qr_pivots(
    nrows: usize,
    ncols: usize,
    entry: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    let mut matrix = Mat::from_fn(nrows, ncols, entry);
    for j in 0..ncols {
        let norm = matrix.col(j).norm_l2();
        if norm > 0.0 {
            for value in matrix.col_mut(j).iter_mut() {
                *value /= norm;
            }
        }
    }
    let r = ColPivQr::new(matrix.as_ref()).thin_R().to_owned();
    (0..nrows.min(ncols)).map(|j| r[(j, j)].abs()).collect()
}

/// Computes the hat diagonal of `weighted_design` and the log-determinant of
/// `weighted_design.T @ weighted_design` with column-pivoted QR. The rank test uses the threshold
/// from firthmodels-py: `max(n, k) * eps * abs(R[0, 0])`.
pub(crate) fn pivoted_qr_hat_logdet(
    weighted_design: &Mat<f64>,
    hat_diagonal: &mut [f64],
) -> Result<f64> {
    let (n, k) = weighted_design.shape();
    debug_assert_eq!(hat_diagonal.len(), n);
    if !weighted_design.as_ref().is_all_finite() {
        return Err(Error::NonFinite);
    }

    let qr = ColPivQr::new(weighted_design.as_ref());
    let r = qr.thin_R();
    let tolerance = n.max(k) as f64 * f64::EPSILON * r[(0, 0)].abs();
    let rank = (0..k).filter(|&j| r[(j, j)].abs() > tolerance).count();
    if rank < k {
        return Err(Error::RankDeficient { rank, required: k });
    }

    let log_determinant = 2.0 * (0..k).map(|j| r[(j, j)].abs().ln()).sum::<f64>();

    let q = qr.compute_thin_Q();
    hat_diagonal.fill(0.0);
    for j in 0..k {
        for (leverage, value) in hat_diagonal.iter_mut().zip(q.col(j).iter()) {
            *leverage += value * value;
        }
    }
    Ok(log_determinant)
}

/// Solves the square system `matrix * x = rhs` for every column of `rhs` with one LU
/// decomposition with partial pivoting.
pub(crate) fn general_solve(matrix: &Mat<f64>, rhs: MatRef<'_, f64>) -> Result<Mat<f64>> {
    let n = check_square_finite(matrix)?;
    debug_assert_eq!(rhs.nrows(), n);
    if !rhs.is_all_finite() {
        return Err(Error::NonFinite);
    }

    let factor = PartialPivLu::new(matrix.as_ref());
    let upper = factor.U();
    if (0..n).any(|i| upper[(i, i)] == 0.0 || !upper[(i, i)].is_finite()) {
        return Err(Error::SingularSystem);
    }

    let solution = factor.solve(rhs);
    if !solution.is_all_finite() {
        return Err(Error::SingularSystem);
    }
    Ok(solution)
}

/// Solves a least-squares system with column-pivoted QR. When `matrix` is rank deficient, the
/// function solves with the SVD pseudoinverse instead.
pub(crate) fn least_squares(matrix: &Mat<f64>, rhs: MatRef<'_, f64>) -> Result<Mat<f64>> {
    let (m, n) = matrix.shape();
    debug_assert!(m >= n && rhs.nrows() == m);
    if !rhs.is_all_finite() {
        return Err(Error::NonFinite);
    }

    let qr = ColPivQr::new(matrix.as_ref());
    let r = qr.thin_R();
    let leading = r[(0, 0)].abs();
    let tolerance = m.max(n) as f64 * f64::EPSILON * leading;
    let rank = (0..n).filter(|&j| r[(j, j)].abs() > tolerance).count();

    if rank == n {
        Ok(qr.solve_lstsq(rhs))
    } else {
        let svd = Svd::new_thin(matrix.as_ref()).map_err(|_| Error::SingularSystem)?;
        Ok(svd.pseudoinverse() * rhs)
    }
}

/// Byte limit for the `packed` and `product` buffers of `MatrixScratch`. Small buffers use less of
/// the L2 cache. The value comes from measurements on the M5 and M2. A buffer always has room for
/// one sample, even when one sample needs more bytes than the limit.
#[cfg(accelerate)]
const MATRIX_SCRATCH_BYTES: usize = 2 * 1024 * 1024;

const ROW_CHUNK: usize = 256;

/// faer is faster for k<8 (measured on M5 and M2).
#[cfg(accelerate)]
const ACCELERATE_MIN_COLS: usize = 8;

#[cfg(accelerate)]
const ACCELERATE_GEMM_CHUNK: usize = 8_192;

/// Computes `x.T @ diag(weights) @ x`.
pub(crate) fn weighted_gram(
    x: MatRef<f64>,
    weights: &[f64],
    scratch: &mut MatrixScratch,
) -> Mat<f64> {
    let mut gram = Mat::zeros(x.ncols(), x.ncols());
    add_weighted_gram(gram.as_mut(), x, weights, scratch);
    copy_lower_to_upper(&mut gram);
    gram
}

/// Adds the lower triangle of `x.T @ diag(weights) @ x` to `gram`.
pub(crate) fn add_weighted_gram(
    gram: MatMut<f64>,
    x: MatRef<f64>,
    weights: &[f64],
    scratch: &mut MatrixScratch,
) {
    #[cfg(accelerate)]
    if x.ncols() >= ACCELERATE_MIN_COLS {
        return add_weighted_gram_accelerate(gram, x, weights, scratch);
    }
    dispatch!(Level::new(), _ => add_weighted_gram_faer(gram, x, weights, scratch));
}

pub(crate) fn copy_lower_to_upper(matrix: &mut Mat<f64>) {
    for j in 0..matrix.ncols() {
        for i in 0..j {
            matrix[(i, j)] = matrix[(j, i)];
        }
    }
}

/// Adds the lower triangle of `x.T @ diag(weights) @ x` to `gram` with faer.
#[inline(always)]
fn add_weighted_gram_faer(
    mut gram: MatMut<f64>,
    x: MatRef<f64>,
    weights: &[f64],
    scratch: &mut MatrixScratch,
) {
    let (n, k) = x.shape();
    debug_assert_eq!(weights.len(), n);
    let chunk_rows = ROW_CHUNK.min(n);
    // The scaling loop accesses `x` and the scaled copy in memory order because the copy has the
    // layout of `x`.
    let row_major = x.col_stride() == 1;
    let mut scaled = if row_major {
        scratch.row_major_rows(chunk_rows, k)
    } else {
        scratch.rows(chunk_rows, k)
    };
    let mut start = 0;
    while start < n {
        let rows = chunk_rows.min(n - start);
        let x_chunk = x.subrows(start, rows);
        let mut scaled_chunk = scaled.rb_mut().subrows_mut(0, rows);
        let chunk_weights = &weights[start..start + rows];
        if row_major {
            for (i, &weight) in chunk_weights.iter().enumerate() {
                let row = scaled_chunk.rb_mut().row_mut(i);
                let row = row.try_as_row_major_mut().unwrap().as_slice_mut();
                let x_row = x_chunk.row(i).try_as_row_major().unwrap().as_slice();
                for (value, &x) in row.iter_mut().zip(x_row) {
                    *value = weight * x;
                }
            }
        } else {
            for j in 0..k {
                let column = scaled_chunk.rb_mut().col_mut(j);
                for ((value, &weight), &x) in column
                    .iter_mut()
                    .zip(chunk_weights)
                    .zip(x_chunk.col(j).iter())
                {
                    *value = weight * x;
                }
            }
        }
        triangular::matmul(
            gram.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Add,
            x_chunk.transpose(),
            BlockStructure::Rectangular,
            scaled_chunk.rb(),
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        start += rows;
    }
}

#[cfg(accelerate)]
const ACCELERATE_GRAM_CHUNK: usize = 16_384;

/// Packs the transpose of each block of `sqrt(weights) * x` because Accelerate's `NoTrans` dsyrk is
/// much faster than `Trans`.
#[cfg(accelerate)]
fn add_weighted_gram_accelerate(
    mut gram: MatMut<f64>,
    x: MatRef<f64>,
    weights: &[f64],
    scratch: &mut MatrixScratch,
) {
    let (n, k) = x.shape();
    debug_assert_eq!(weights.len(), n);
    debug_assert!(weights.iter().all(|&weight| weight >= 0.0));
    let mut packed = scratch.packed(k, ACCELERATE_GRAM_CHUNK.min(n));
    let chunk_samples = packed.ncols();
    let mut start = 0;
    while start < n {
        let cols = chunk_samples.min(n - start);
        crate::accelerate::pack_scaled_rows(
            x.subrows(start, cols),
            &weights[start..start + cols],
            packed.rb_mut().subcols_mut(0, cols),
        );
        crate::accelerate::a_mul_a_transpose_lower(packed.rb().subcols(0, cols), gram.rb_mut());
        start += cols;
    }
}

/// Writes `x @ v` to `out` for a column-major `x`.
pub(crate) fn mul_vec(x: MatRef<f64>, v: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.row_stride(), 1);
    #[cfg(accelerate)]
    return crate::accelerate::a_mul_x(x, v, out);
    #[cfg(not(accelerate))]
    matmul(
        faer::ColMut::from_slice_mut(out),
        Accum::Replace,
        x,
        faer::ColRef::from_slice(v),
        1.0,
        Par::Seq,
    );
}

/// Adds `x.T @ v` to `out` for a column-major `x`.
pub(crate) fn add_transpose_mul_vec(x: MatRef<f64>, v: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.row_stride(), 1);
    #[cfg(accelerate)]
    {
        let mut product = vec![0.0; x.ncols()];
        crate::accelerate::a_transpose_mul_x(x, v, &mut product);
        for (out, product) in out.iter_mut().zip(product) {
            *out += product;
        }
    }
    #[cfg(not(accelerate))]
    matmul(
        faer::ColMut::from_slice_mut(out),
        Accum::Add,
        x.transpose(),
        faer::ColRef::from_slice(v),
        1.0,
        Par::Seq,
    );
}

/// Computes `x @ matrix` into `out`.
pub(crate) fn rows_mul(x: MatRef<f64>, matrix: &Mat<f64>, out: MatMut<f64>) {
    #[cfg(accelerate)]
    if x.ncols() >= ACCELERATE_MIN_COLS {
        let (n, mut out) = (x.nrows(), out);
        let mut start = 0;
        while start < n {
            let rows = ACCELERATE_GEMM_CHUNK.min(n - start);
            crate::accelerate::a_mul_b(
                x.subrows(start, rows),
                matrix.as_ref(),
                out.rb_mut().subrows_mut(start, rows),
            );
            start += rows;
        }
        return;
    }
    matmul(out, Accum::Replace, x, matrix.as_ref(), 1.0, Par::Seq);
}

/// Computes x[i, :] @ matrix @ x[i, :].T for each row of x.
pub(crate) fn quadratic_form_rows(
    x: MatRef<f64>,
    matrix: &Mat<f64>,
    values: &mut [f64],
    scratch: &mut MatrixScratch,
) {
    let (n, k) = x.shape();
    debug_assert_eq!(values.len(), n);
    #[cfg(accelerate)]
    if k >= ACCELERATE_MIN_COLS {
        return quadratic_form_rows_accelerate(x, matrix, values, scratch);
    }
    let chunk_rows = ROW_CHUNK.min(n);
    let mut product = scratch.rows(chunk_rows, k);
    let mut start = 0;
    while start < n {
        let rows = chunk_rows.min(n - start);
        let x_chunk = x.subrows(start, rows);
        matmul(
            product.rb_mut().subrows_mut(0, rows),
            Accum::Replace,
            x_chunk,
            matrix.as_ref(),
            1.0,
            Par::Seq,
        );
        row_dot_products(
            product.rb().subrows(0, rows),
            x_chunk,
            &mut values[start..start + rows],
        );
        start += rows;
    }
}

/// Writes the dot product of row `i` of `left` and row `i` of `right` to `values[i]`. The products
/// of each row are added in column order.
fn row_dot_products(left: MatRef<f64>, right: MatRef<f64>, values: &mut [f64]) {
    let (n, k) = left.shape();
    debug_assert_eq!(right.shape(), (n, k));
    debug_assert_eq!(values.len(), n);
    let mut start = 0;
    while start + 4 <= n {
        let mut sums = [0.0; 4];
        for j in 0..k {
            for (offset, sum) in sums.iter_mut().enumerate() {
                *sum += left[(start + offset, j)] * right[(start + offset, j)];
            }
        }
        values[start..start + 4].copy_from_slice(&sums);
        start += 4;
    }
    for (i, value) in values.iter_mut().enumerate().skip(start) {
        *value = left
            .row(i)
            .iter()
            .zip(right.row(i).iter())
            .map(|(a, b)| a * b)
            .sum();
    }
}

// The product buffer matches the layout of `x`, so the dot products read both in memory order.
#[cfg(accelerate)]
fn quadratic_form_rows_accelerate(
    x: MatRef<f64>,
    matrix: &Mat<f64>,
    values: &mut [f64],
    scratch: &mut MatrixScratch,
) {
    let (n, k) = x.shape();
    let rows_in_budget = (MATRIX_SCRATCH_BYTES / (k * 8)).max(1);
    let chunk_rows = ACCELERATE_GEMM_CHUNK.min(n).min(rows_in_budget);
    let column_major = x.row_stride() == 1;
    let size = chunk_rows * k;
    if scratch.product.len() < size {
        scratch.product = vec![0.0; size];
    }
    let product = &mut scratch.product;
    let mut start = 0;
    while start < n {
        let rows = chunk_rows.min(n - start);
        let x_chunk = x.subrows(start, rows);
        let buffer = &mut product[..rows * k];
        let mut p = if column_major {
            MatMut::from_column_major_slice_mut(buffer, rows, k)
        } else {
            MatMut::from_row_major_slice_mut(buffer, rows, k)
        };
        crate::accelerate::a_mul_b(x_chunk, matrix.as_ref(), p.rb_mut());
        let chunk_values = &mut values[start..start + rows];
        if column_major {
            column_dot_products(p.rb(), x_chunk, chunk_values);
        } else {
            for (i, value) in chunk_values.iter_mut().enumerate() {
                *value = p
                    .rb()
                    .row(i)
                    .iter()
                    .zip(x_chunk.row(i).iter())
                    .map(|(z, x)| z * x)
                    .sum();
            }
        }
        start += rows;
    }
}

/// Returns the order of a square matrix, or `NonFinite` if an entry is not finite.
fn check_square_finite(matrix: &Mat<f64>) -> Result<usize> {
    debug_assert_eq!(matrix.nrows(), matrix.ncols());
    if !matrix.as_ref().is_all_finite() {
        return Err(Error::NonFinite);
    }
    Ok(matrix.nrows())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;
    use proptest::prelude::*;

    fn assert_inverse_product(matrix: &Mat<f64>, inverse: &Mat<f64>, tolerance: f64) {
        let n = matrix.nrows();
        for i in 0..n {
            for j in 0..n {
                let product: f64 = (0..n).map(|k| matrix[(i, k)] * inverse[(k, j)]).sum();
                assert!(
                    (product - f64::from(i == j)).abs() < tolerance,
                    "entry ({i},{j}) of matrix @ inverse is {product}"
                );
            }
        }
    }

    fn assert_rank_deficient(matrix: &Mat<f64>, expected_rank: usize, required: usize) {
        match pivoted_cholesky_inverse_logdet(matrix) {
            Err(Error::RankDeficient { rank, required: n }) => {
                assert_eq!(rank, expected_rank);
                assert_eq!(n, required);
            },
            _ => panic!("expected rank {expected_rank} of {required}"),
        }
    }

    #[test]
    fn cholesky_inverse_logdet_2x2() {
        let a = mat![[4.0, 2.0], [2.0, 3.0]];
        let factor = cholesky_factor_logdet(&a).unwrap();
        assert!((factor.log_determinant - 8.0_f64.ln()).abs() < 1e-12);
        assert!((factor.minimum_relative_pivot - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((factor.minimum_pivot - 2.0_f64.sqrt()).abs() < 1e-12);
        let output = cholesky_inverse_logdet(&a).unwrap();
        assert!((output.log_determinant - 8.0_f64.ln()).abs() < 1e-12);
        assert!((output.minimum_pivot - 2.0_f64.sqrt()).abs() < 1e-12);
        let expected = mat![[0.375, -0.25], [-0.25, 0.5]];
        for i in 0..2 {
            for j in 0..2 {
                let (obs, exp) = (output.inverse[(i, j)], expected[(i, j)]);
                assert!(
                    (obs - exp).abs() < 1e-12,
                    "({i},{j}): got {obs}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn pivoted_cholesky_rank_uses_relative_cutoff() {
        let tiny = mat![[1.0e-30, 0.0], [0.0, 1.0e-30]];
        let summary = pivoted_cholesky_inverse_logdet(&tiny).unwrap();
        assert!((summary.log_determinant - 2.0 * 1.0e-30_f64.ln()).abs() < 1e-10);
        assert_inverse_product(&tiny, &summary.inverse, 1e-12);

        let above_cutoff = mat![[1.0, 0.0], [0.0, 4.0 * f64::EPSILON]];
        assert!(pivoted_cholesky_inverse_logdet(&above_cutoff).is_ok());
        assert_rank_deficient(&mat![[1.0, 0.0], [0.0, 2.0 * f64::EPSILON]], 1, 2);

        assert_rank_deficient(&mat![[0.0]], 0, 1);
        assert_rank_deficient(&mat![[1.0, 1.0], [1.0, 1.0]], 1, 2);
        // Pivoted Cholesky cannot distinguish this indefinite remainder
        // from a rank-deficient one. LAPACK `dpstrf` also returns rank 1 for this matrix.
        assert_rank_deficient(&mat![[1.0, 2.0], [2.0, 1.0]], 1, 2);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn pivoted_cholesky_agrees_on_generated_spd_matrices(
            entries in proptest::array::uniform9(-3.0_f64..3.0),
            diagonal_shift in 0.01_f64..3.0,
        ) {
            let b = Mat::from_fn(3, 3, |i, j| entries[3 * i + j]);
            let mut a = b.transpose() * b.as_ref();
            for i in 0..3 {
                a[(i, i)] += diagonal_shift;
            }

            let unpivoted = cholesky_inverse_logdet(&a).unwrap();
            let pivoted = pivoted_cholesky_inverse_logdet(&a).unwrap();
            let logdet_scale = unpivoted
                .log_determinant
                .abs()
                .max(pivoted.log_determinant.abs());
            prop_assert!(
                (unpivoted.log_determinant - pivoted.log_determinant).abs()
                    <= 1e-10 * (1.0 + logdet_scale),
                "log determinants differ: ordinary {}, pivoted {}",
                unpivoted.log_determinant,
                pivoted.log_determinant
            );
            assert_inverse_product(&a, &pivoted.inverse, 1e-10);
        }
    }

    #[test]
    fn quadratic_form_rows_wide_design() {
        // Exercise both layouts across full and partial blocks.
        let (n, k) = (20_003, 12);
        let x = Mat::from_fn(n, k, |i, j| ((7 * i + 3 * j) % 13) as f64 - 6.0);
        let row_major: Vec<f64> = (0..n * k).map(|idx| x[(idx / k, idx % k)]).collect();
        let row_major = MatRef::from_row_major_slice(&row_major, n, k);
        let mut a = x.transpose() * x.as_ref();
        for i in 0..k {
            a[(i, i)] += 1.0;
        }
        let factor = cholesky_factor_logdet(&a).unwrap();
        let inverse = cholesky_inverse_logdet(&a).unwrap().inverse;
        let mut factored = vec![0.0; n];
        factor.quadratic_form_rows(x.as_ref(), &mut factored, &mut MatrixScratch::default());
        let mut inverted = vec![0.0; n];
        quadratic_form_rows(
            row_major,
            &inverse,
            &mut inverted,
            &mut MatrixScratch::default(),
        );
        let mut product = vec![0.0; n * k];
        rows_mul(
            row_major,
            &inverse,
            MatMut::from_row_major_slice_mut(&mut product, n, k),
        );
        for i in 0..n {
            let direct: f64 = (0..k)
                .map(|r| {
                    (0..k)
                        .map(|s| x[(i, r)] * inverse[(r, s)] * x[(i, s)])
                        .sum::<f64>()
                })
                .sum();
            let multiplied: f64 = (0..k).map(|s| product[i * k + s] * x[(i, s)]).sum();
            for value in [factored[i], inverted[i], multiplied] {
                assert!(
                    (value - direct).abs() < 1e-9 * direct.max(1.0),
                    "row {i}: {value} vs {direct}"
                );
            }
        }
    }

    #[test]
    fn weighted_gram_matches_explicit_product() {
        // The first case exceeds `ROW_CHUNK` rows and the second exceeds `ACCELERATE_GRAM_CHUNK`
        // rows. With odd n and k in the second case, `pack_scaled_rows` on Apple silicon exercises
        // its scalar tail for both the final sample and the final column.
        for (n, k) in [(300, 3), (20_001, 11)] {
            let x = Mat::from_fn(n, k, |i, j| ((7 * i + 3 * j) % 13) as f64 - 6.0);
            let row_major: Vec<f64> = (0..n * k).map(|idx| x[(idx / k, idx % k)]).collect();
            let weights: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.5).collect();
            for gram in [
                weighted_gram(x.as_ref(), &weights, &mut MatrixScratch::default()),
                weighted_gram(
                    MatRef::from_row_major_slice(&row_major, n, k),
                    &weights,
                    &mut MatrixScratch::default(),
                ),
            ] {
                for r in 0..k {
                    for s in 0..k {
                        let expected: f64 =
                            (0..n).map(|i| weights[i] * x[(i, r)] * x[(i, s)]).sum();
                        assert!(
                            (gram[(r, s)] - expected).abs() < 1e-9 * expected.abs().max(1.0),
                            "k={k} ({r},{s}): got {}, expected {expected}",
                            gram[(r, s)]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pivoted_qr_hat_logdet_matches_cholesky_path() {
        let x = mat![
            [1.0, 0.5, -1.0],
            [1.0, -0.3, 2.0],
            [1.0, 1.7, 0.4],
            [1.0, 0.2, -0.6],
            [1.0, -1.1, 1.3]
        ];
        let mut hat_diagonal = vec![0.0; 5];
        let log_determinant = pivoted_qr_hat_logdet(&x, &mut hat_diagonal).unwrap();

        let xtx = x.transpose() * x.as_ref();
        let cholesky = cholesky_inverse_logdet(&xtx).unwrap();
        assert!((log_determinant - cholesky.log_determinant).abs() < 1e-12);
        for i in 0..5 {
            let expected: f64 = (0..3)
                .map(|r| {
                    (0..3)
                        .map(|s| x[(i, r)] * cholesky.inverse[(r, s)] * x[(i, s)])
                        .sum::<f64>()
                })
                .sum();
            assert!(
                (hat_diagonal[i] - expected).abs() < 1e-12,
                "row {i}: got {}, expected {expected}",
                hat_diagonal[i]
            );
        }
    }

    #[test]
    fn pivoted_qr_hat_logdet_survives_where_normal_equations_fail() {
        // Rounding makes X'X indefinite even though X itself has rank two. In exact arithmetic, the
        // determinant of X'X is 20 * d^2.
        let d = 5.0e-9;
        let x = mat![
            [1.0, 1.0],
            [1.0, 1.0 + d],
            [1.0, 1.0 - d],
            [1.0, 1.0 + 2.0 * d]
        ];
        let xtx = x.transpose() * x.as_ref();
        assert!(matches!(
            cholesky_inverse_logdet(&xtx),
            Err(Error::NotPositiveDefinite)
        ));

        let mut hat_diagonal = vec![0.0; 4];
        let log_determinant = pivoted_qr_hat_logdet(&x, &mut hat_diagonal).unwrap();
        assert!((log_determinant - (20.0 * d * d).ln()).abs() < 1e-8);
        assert!((hat_diagonal.iter().sum::<f64>() - 2.0).abs() < 1e-8);
    }

    #[test]
    fn pivoted_qr_hat_logdet_reports_rank_deficiency() {
        // The second column is twice the first.
        let x = mat![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let mut hat_diagonal = vec![0.0; 3];
        assert!(matches!(
            pivoted_qr_hat_logdet(&x, &mut hat_diagonal),
            Err(Error::RankDeficient {
                rank: 1,
                required: 2
            })
        ));
        // The second pivot equals the cutoff. The rank is the number of pivots above the cutoff.
        let at_cutoff = mat![[1.0, 0.0], [0.0, 2.0 * f64::EPSILON]];
        assert!(matches!(
            pivoted_qr_hat_logdet(&at_cutoff, &mut hat_diagonal[..2]),
            Err(Error::RankDeficient {
                rank: 1,
                required: 2
            })
        ));
    }

    #[test]
    fn general_solve_reports_singular_system() {
        assert!(matches!(
            general_solve(&mat![[1.0, 2.0], [2.0, 4.0]], mat![[1.0], [1.0]].as_ref()),
            Err(Error::SingularSystem)
        ));
    }

    #[test]
    fn row_kernels_match_fresh_scratch_after_reuse() {
        let mut scratch = MatrixScratch::default();
        // On Apple silicon, the kernels use faer when k = 3 and Accelerate when k = 11.
        for k in [3, 11] {
            let matrix = Mat::from_fn(k, k, |i, j| if i == j { 2.0 } else { 0.5 });
            let factor = cholesky_factor_logdet(&matrix).unwrap();
            // 20,001 rows exceed the chunk limits of both backends. The last chunk is partial in
            // both backends.
            for n in [20_001, 37] {
                let x = Mat::from_fn(n, k, |i, j| ((i * 7 + j * 13 + n) % 29) as f64 / 29.0 - 0.5);
                let weights: Vec<_> = (0..n).map(|i| (i % 7) as f64 * 0.5).collect();
                let row_major: Vec<_> = (0..n * k).map(|i| x[(i / k, i % k)]).collect();
                for view in [x.as_ref(), MatRef::from_row_major_slice(&row_major, n, k)] {
                    let gram = weighted_gram(view, &weights, &mut scratch);
                    let expected = weighted_gram(view, &weights, &mut MatrixScratch::default());
                    for j in 0..k {
                        for (&value, &expected) in gram.col(j).iter().zip(expected.col(j).iter()) {
                            assert!((value - expected).abs() < 1e-12 * expected.abs().max(1.0));
                        }
                    }

                    let mut values = vec![f64::NAN; n];
                    let mut expected = vec![f64::NAN; n];
                    quadratic_form_rows(view, &matrix, &mut values, &mut scratch);
                    quadratic_form_rows(
                        view,
                        &matrix,
                        &mut expected,
                        &mut MatrixScratch::default(),
                    );
                    for (&value, &expected) in values.iter().zip(&expected) {
                        assert!((value - expected).abs() < 1e-12 * expected.abs().max(1.0));
                    }

                    factor.quadratic_form_rows(view, &mut values, &mut scratch);
                    factor.quadratic_form_rows(view, &mut expected, &mut MatrixScratch::default());
                    for (&value, &expected) in values.iter().zip(&expected) {
                        assert!((value - expected).abs() < 1e-12 * expected.abs().max(1.0));
                    }
                }
            }
        }
    }

    #[cfg(accelerate)]
    #[test]
    fn apple_matrix_scratch_preserves_results_and_bounds() {
        // This width needs padding. At this width, the byte limit determines the chunk sizes.
        let k = 129;
        let b: Vec<_> = (1..=k).map(|j| j as f64 / 128.0).collect();
        let b_squared_sum: f64 = b.iter().map(|value| value * value).sum();
        let identity = Mat::identity(k, k);
        let mut scratch = MatrixScratch::default();
        // With the second input, `quadratic_form_rows` grows the product buffer and leaves one row
        // in the last chunk.
        for n in [2000, 2033] {
            let a: Vec<_> = (1..=n).map(|i| i as f64 / 2048.0).collect();
            let weights: Vec<_> = (0..n).map(|i| (i % 5) as f64).collect();
            let x = Mat::from_fn(n, k, |i, j| a[i] * b[j]);
            let gram = weighted_gram(x.as_ref(), &weights, &mut scratch);
            let weighted_a_squared_sum: f64 = a.iter().zip(&weights).map(|(a, w)| w * a * a).sum();
            for r in 0..k {
                for s in 0..k {
                    let expected = b[r] * b[s] * weighted_a_squared_sum;
                    assert!((gram[(r, s)] - expected).abs() < 1e-11 * expected);
                }
            }
            let mut values = vec![0.0; n];
            quadratic_form_rows(x.as_ref(), &identity, &mut values, &mut scratch);
            for (&observed, &a) in values.iter().zip(&a) {
                let expected = a * a * b_squared_sum;
                assert!((observed - expected).abs() < 1e-11 * expected);
            }
            assert!(scratch.packed.len() * 8 <= MATRIX_SCRATCH_BYTES);
            assert!(scratch.product.capacity() * 8 <= MATRIX_SCRATCH_BYTES);
        }
    }
}
