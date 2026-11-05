use fearless_simd::{Level, dispatch};

use crate::{Error, Result};

/// Borrowed view of a dense `f64` matrix in row-major or column-major order.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a> {
    data: &'a [f64],
    nrows: usize,
    ncols: usize,
    row_stride: usize,
    col_stride: usize,
}

impl<'a> MatrixView<'a> {
    /// Creates a view of `data` in row-major (C) order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDimensions`] if either dimension is zero, if `nrows * ncols`
    /// overflows, or if `data.len()` is not exactly `nrows * ncols`.
    pub fn from_row_major(data: &'a [f64], nrows: usize, ncols: usize) -> Result<Self> {
        Self::with_strides(data, nrows, ncols, ncols, 1)
    }

    /// Creates a view of `data` in column-major (Fortran) order.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`MatrixView::from_row_major`].
    pub fn from_column_major(data: &'a [f64], nrows: usize, ncols: usize) -> Result<Self> {
        Self::with_strides(data, nrows, ncols, 1, nrows)
    }

    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.ncols
    }

    pub(crate) const fn is_column_major(&self) -> bool {
        self.row_stride == 1
    }

    /// Returns a row slice when its entries are contiguous.
    pub(crate) fn contiguous_row(&self, row: usize) -> Option<&[f64]> {
        if self.col_stride != 1 {
            return None;
        }
        let start = row * self.row_stride;
        Some(&self.data[start..start + self.ncols])
    }

    /// Returns the element at (`row`, `col`).
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(
            row < self.nrows && col < self.ncols,
            "index ({row}, {col}) is out of bounds for a {} by {} matrix",
            self.nrows,
            self.ncols
        );
        self.at(row, col)
    }

    /// Returns the element at (`row`, `col`) for callers whose indices are in range by
    /// construction. Only the flat offset is checked, so an out-of-range index can return another
    /// element.
    pub(crate) fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.row_stride + col * self.col_stride]
    }

    /// Returns true when every element is finite.
    pub(crate) fn all_finite(&self) -> bool {
        // LLVM vectorizes this fold. It does not vectorize `all`, which returns at the first
        // non-finite value.
        dispatch!(Level::new(), _ => self
            .data
            .iter()
            .fold(true, |all, value| all & value.is_finite()))
    }

    fn with_strides(
        data: &'a [f64],
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<Self> {
        if nrows == 0 || ncols == 0 {
            return Err(Error::InvalidDimensions(format!(
                "matrix dimensions must be nonzero, got {nrows} by {ncols}"
            )));
        }
        let expected = nrows.checked_mul(ncols).ok_or_else(|| {
            Error::InvalidDimensions(format!(
                "matrix dimensions {nrows} by {ncols} overflow usize"
            ))
        })?;
        if data.len() != expected {
            return Err(Error::InvalidDimensions(format!(
                "buffer length {} does not match {nrows} by {ncols}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            nrows,
            ncols,
            row_stride,
            col_stride,
        })
    }
}
