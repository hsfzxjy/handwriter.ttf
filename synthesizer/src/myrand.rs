// Copyright 2018 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Math helper functions

#[derive(Clone, Copy, Debug)]
pub struct Bv2dNormal {
    pub mu: [f64; 2],
    pub sigma: [f64; 2],
    pub ro: f64,
}

impl rand::distributions::Distribution<[f64; 2]> for Bv2dNormal {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> [f64; 2] {
        use rand_distr::StandardNormal;
        let z1: f64 = rng.sample(StandardNormal);
        let z2: f64 = rng.sample(StandardNormal);
        [
            self.mu[0] + self.sigma[0] * z1,
            self.mu[1] + self.sigma[1] * (z1 * self.ro + z2 * (1. - self.ro * self.ro).sqrt()),
        ]
    }
}
