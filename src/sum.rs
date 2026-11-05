/// Neumaier summation.
#[derive(Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    compensation: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, term: f64) {
        let sum = self.sum + term;
        self.compensation += if self.sum.abs() >= term.abs() {
            (self.sum - sum) + term
        } else {
            (term - sum) + self.sum
        };
        self.sum = sum;
    }

    pub(crate) fn scale(&mut self, factor: f64) {
        self.sum *= factor;
        self.compensation *= factor;
    }

    pub(crate) fn total(&self) -> f64 {
        self.sum + self.compensation
    }
}
