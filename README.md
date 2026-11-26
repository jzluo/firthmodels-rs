# firthmodels

[![Crates.io](https://img.shields.io/crates/v/firthmodels.svg)](https://crates.io/crates/firthmodels)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24548826.svg)](https://doi.org/10.5281/zenodo.24548826)

Firth-penalized logistic and Cox proportional hazards regression in Rust.

- **Penalized and unpenalized fits** for logistic and Cox proportional hazards regression, with configurable `penalty_weight` (defaults to 0.5 for Firth's penalty, or 0 for standard maximum likelihood).
- **Wald, likelihood ratio, and profile-likelihood inference**, including joint penalized likelihood ratio tests on arbitrary subsets of parameters and Venzon-Moolgavkar profile confidence bounds.
- **Prepared data structures** (`LogisticData` and `CoxData`) that check and store the data once for reuse across many fits (such as bootstrap replicates). A `LogisticData` can also be reused across outcomes (such as in permutation tests or multi-outcome scans).
- **Single-threaded, `Send + Sync` execution** allowing callers to parallelize across fits or inference queries.

## Platforms

Requires Rust 1.89 or later. The crate uses [faer](https://crates.io/crates/faer) for linear
algebra and [fearless_simd](https://crates.io/crates/fearless_simd) for SIMD kernels.

On Apple silicon, the crate also calls Apple Accelerate for BLAS, LAPACK, and vForce operations.
Some of these symbols require macOS 13.3 or later, so set `MACOSX_DEPLOYMENT_TARGET=13.3` (or
newer) when building binaries for distribution to other Macs.

## Installation

```sh
cargo add firthmodels
```

## Examples

### Logistic regression

```rust
use firthmodels::{LogisticData, MatrixView, ProfileOptions, Result};

fn main() -> Result<()> {
    // Every row with x = 1 has y = 1, so the unpenalized estimate for x is infinite.
    let x = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let features = MatrixView::from_row_major(&x, x.len(), 1)?;

    let fit = LogisticData::new(features)?.model(&y).fit()?;
    assert!(fit.status().is_converged());
    println!("coefficient {:.3}, SE {:.3}", fit.coefficients()[0], fit.standard_errors()[0]);

    // The second argument is `warm_start` - see https://github.com/jzluo/firthmodels/pull/44
    let lrt = fit.likelihood_ratio_test(&[0], true)?;
    assert!(lrt.is_valid());
    println!("LRT p-value {:.4}", lrt.p_value());

    let [lower, upper] = fit.profile_confidence_interval(0, 0.05, ProfileOptions::default())?;
    assert!(lower.is_valid() && upper.is_valid());
    println!("95% profile interval [{:.3}, {:.3}]", lower.bound(), upper.bound());
    Ok(())
}
```

### Cox regression

```rust
use firthmodels::{CoxData, MatrixView, ProfileOptions, Result};

fn main() -> Result<()> {
    // The only samples with 1.0 in the first column also have the two earliest events, so the 
    // unpenalized estimate of the first coefficient is infinite.
    let x = [
        1.0, 0.5,
        1.0, 1.2,
        0.0, 0.8,
        0.0, 1.5,
        0.0, 0.3,
        0.0, 1.1,
    ];
    let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let event = [true, true, true, false, true, false];
    let features = MatrixView::from_row_major(&x, 6, 2)?;

    let fit = CoxData::new(features, &time, &event)?.model().fit()?;
    assert!(fit.status().is_converged());
    println!("coefficients {:?}", fit.coefficients());

    // Joint test that both coefficients are zero, with 2 degrees of freedom.
    let lrt = fit.likelihood_ratio_test(&[0, 1], true)?;
    assert!(lrt.is_valid());
    println!("joint LRT p-value {:.4}", lrt.p_value());

    let [lower, upper] = fit.profile_confidence_interval(0, 0.05, ProfileOptions::default())?;
    assert!(lower.is_valid() && upper.is_valid());
    println!("95% profile interval [{:.3}, {:.3}]", lower.bound(), upper.bound());

    let c_index = fit.concordance(features, &time, &event, None)?;
    println!("Harrell's C-index {c_index:.3}");

    // The survival curves have one row per sample and one column per unique event time.
    let survival = fit.predict_survival(features, None)?;
    assert_eq!(survival.len(), 6 * fit.unique_times().len());
    Ok(())
}
```

### Reusing prepared data across fits

To fit many models on the same data, construct one `LogisticData` or `CoxData` and call `model` for each fit with different 
sample weights (such as bootstrap frequency weights), offsets, or options. Because `LogisticData` stores only the feature 
matrix, you can also reuse it across different outcomes. `CoxData::new` sorts samples by survival time and groups tied 
times up front, so a different survival outcome requires a separate `CoxData`.

```rust
use firthmodels::{CoxData, MatrixView, Result};

fn main() -> Result<()> {
    let features = MatrixView::from_row_major(&[0.2, 1.1, 0.7, 0.4, 1.5, 0.9], 6, 1)?;
    let time = [1.0, 2.0, 2.0, 4.0, 5.0, 7.0];
    let event = [true, true, false, true, false, true];
    let data = CoxData::new(features, &time, &event)?;

    // Both fits reuse the sorted data with different frequency weights.
    for weights in [[1.0, 2.0, 1.0, 1.0, 0.0, 1.0], [2.0, 1.0, 0.0, 1.0, 1.0, 1.0]] {
        let fit = data.model().sample_weight(&weights).fit()?;
        assert!(fit.status().is_converged());
    }
    Ok(())
}
```

## References

- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Heinze, G., Schemper, M. (2001). A solution to the problem of monotone likelihood in Cox
  regression. *Biometrics*, 57(1), 114–119.
- Heinze, G., Schemper, M. (2002). A solution to the problem of separation in logistic regression.
  *Statistics in Medicine*, 21(16), 2409–2419.
- Kosmidis, I., Firth, D. (2021). Jeffreys-prior penalty, finiteness and shrinkage in
  binomial-response generalized linear models. *Biometrika*, 108(1), 71–82.
- Venzon, D. J., Moolgavkar, S. H. (1988). A method for computing profile-likelihood-based
  confidence intervals. *Applied Statistics*, 37(1), 87–94.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
