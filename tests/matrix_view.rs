use firthmodels::{Error, MatrixView};

#[test]
fn row_and_col_major_transpose() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = MatrixView::from_row_major(&data, 2, 3).unwrap();
    let col = MatrixView::from_column_major(&data, 3, 2).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(row.get(i, j), col.get(j, i));
        }
    }
    assert_eq!(row.get(1, 0), 4.0);
    assert_eq!(col.get(1, 0), 2.0);
}

#[test]
fn get_checks_each_index() {
    // Both flat offsets are inside the buffer.
    let data = [1.0, 2.0, 3.0, 4.0];
    let row = MatrixView::from_row_major(&data, 2, 2).unwrap();
    let col = MatrixView::from_column_major(&data, 2, 2).unwrap();
    assert!(std::panic::catch_unwind(|| row.get(0, 2)).is_err());
    assert!(std::panic::catch_unwind(|| col.get(2, 0)).is_err());
}

#[test]
fn invalid_dimensions() {
    let data = [1.0, 2.0, 3.0, 4.0];
    // zero dimension
    assert!(matches!(
        MatrixView::from_row_major(&data, 0, 2),
        Err(Error::InvalidDimensions(_))
    ));
    assert!(matches!(
        MatrixView::from_column_major(&data, 2, 0),
        Err(Error::InvalidDimensions(_))
    ));
    // length mismatch
    assert!(matches!(
        MatrixView::from_row_major(&data, 2, 5),
        Err(Error::InvalidDimensions(_))
    ));
    // overflow
    assert!(matches!(
        MatrixView::from_row_major(&data, usize::MAX, 2),
        Err(Error::InvalidDimensions(_))
    ));
}
