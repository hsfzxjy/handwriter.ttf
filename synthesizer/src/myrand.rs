// Copyright 2018 Developers of the Rand project.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Math helper functions

#![allow(unused)]

use tiny_rng::Rand;

use crate::ziggurat_tables;
use core::{cmp, ops};
use std::{cmp::Ordering, sync::atomic};

pub struct MyRng {
    tiny: tiny_rng::Rng,
}

impl MyRng {
    pub fn new() -> Self {
        static SEED: atomic::AtomicU64 = atomic::AtomicU64::new(42);
        Self {
            tiny: tiny_rng::Rng::from_seed(SEED.fetch_add(1, atomic::Ordering::Relaxed)),
        }
    }
    pub fn gen<T>(&mut self) -> T
    where
        Standard: Distribution<T>,
    {
        Standard.sample(self)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.tiny.rand_u64()
    }
    pub fn sample<D: Distribution<T>, T>(&mut self, d: D) -> T {
        d.sample(self)
    }
}

pub trait Distribution<T> {
    fn sample(&self, rng: &mut MyRng) -> T;
}

#[derive(Clone, Debug)]
pub struct WeightedIndex<F> {
    cumulative_weights: Vec<F>,
    total_weight: F,
}

impl<F: Float + Default> WeightedIndex<F> {
    pub fn new<I>(weights: I) -> Self
    where
        I: IntoIterator<Item = F>,
    {
        let iter = weights.into_iter();
        let mut cum = Vec::<F>::with_capacity(iter.size_hint().0);
        let mut total: F = Default::default();
        for w in iter {
            cum.push(total);
            total += w;
        }
        Self {
            cumulative_weights: cum,
            total_weight: total,
        }
    }
}

impl Distribution<usize> for WeightedIndex<f32> {
    fn sample(&self, rng: &mut MyRng) -> usize {
        let chosen: f32 = Open01.sample(rng);
        let chosen = chosen * self.total_weight;
        self.cumulative_weights
            .binary_search_by(|w| {
                if *w <= chosen {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_err()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bv2dNormal {
    pub mu: [f64; 2],
    pub sigma: [f64; 2],
    pub ro: f64,
}

impl Distribution<[f64; 2]> for Bv2dNormal {
    fn sample(&self, rng: &mut MyRng) -> [f64; 2] {
        let z1: f64 = rng.sample(StandardNormal);
        let z2: f64 = rng.sample(StandardNormal);
        [
            self.mu[0] + self.sigma[0] * z1,
            self.mu[1] + self.sigma[1] * (z1 * self.ro + z2 * (1. - self.ro * self.ro).sqrt()),
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StandardNormal;

impl Distribution<f32> for StandardNormal {
    #[inline]
    fn sample(&self, rng: &mut MyRng) -> f32 {
        // TODO: use optimal 32-bit implementation
        let x: f64 = self.sample(rng);
        x as f32
    }
}

impl Distribution<f64> for StandardNormal {
    fn sample(&self, rng: &mut MyRng) -> f64 {
        #[inline]
        fn pdf(x: f64) -> f64 {
            (-x * x / 2.0).exp()
        }
        #[inline]
        fn zero_case(rng: &mut MyRng, u: f64) -> f64 {
            // compute a random number in the tail by hand

            // strange initial conditions, because the loop is not
            // do-while, so the condition should be true on the first
            // run, they get overwritten anyway (0 < 1, so these are
            // good).
            let mut x = 1.0f64;
            let mut y = 0.0f64;

            while -2.0 * y < x * x {
                let x_: f64 = rng.sample(Open01);
                let y_: f64 = rng.sample(Open01);

                x = x_.ln() / ziggurat_tables::ZIG_NORM_R;
                y = y_.ln();
            }

            if u < 0.0 {
                x - ziggurat_tables::ZIG_NORM_R
            } else {
                ziggurat_tables::ZIG_NORM_R - x
            }
        }

        ziggurat(
            rng,
            true, // this is symmetric
            &ziggurat_tables::ZIG_NORM_X,
            &ziggurat_tables::ZIG_NORM_F,
            pdf,
            zero_case,
        )
    }
}

pub struct Standard;

impl Distribution<u32> for Standard {
    fn sample(&self, rng: &mut MyRng) -> u32 {
        rng.tiny.rand_u32()
    }
}

impl Distribution<u64> for Standard {
    fn sample(&self, rng: &mut MyRng) -> u64 {
        rng.tiny.rand_u64()
    }
}

#[derive(Copy, Clone, Default)]
struct Open01;

macro_rules! float_impls {
    ($ty:ident, $uty:ident, $f_scalar:ident, $u_scalar:ty,
     $fraction_bits:expr, $exponent_bias:expr) => {
        impl IntoFloat for $uty {
            type F = $ty;
            #[inline(always)]
            fn into_float_with_exponent(self, exponent: i32) -> $ty {
                // The exponent is encoded using an offset-binary representation
                let exponent_bits: $u_scalar =
                    (($exponent_bias + exponent) as $u_scalar) << $fraction_bits;
                $ty::from_bits(self | exponent_bits)
            }
        }

        impl Distribution<$ty> for Standard {
            fn sample(&self, rng: &mut MyRng) -> $ty {
                // Multiply-based method; 24/53 random bits; [0, 1) interval.
                // We use the most significant bits because for simple RNGs
                // those are usually more random.
                let float_size = std::mem::size_of::<$f_scalar>() as u32 * 8;
                let precision = $fraction_bits + 1;
                let scale = 1.0 / ((1 as $u_scalar << precision) as $f_scalar);

                let value: $uty = rng.gen();
                let value = value >> (float_size - precision);
                scale * (value as $ty)
            }
        }

        impl Distribution<$ty> for Open01 {
            fn sample(&self, rng: &mut MyRng) -> $ty {
                // Transmute-based method; 23/52 random bits; (0, 1) interval.
                // We use the most significant bits because for simple RNGs
                // those are usually more random.
                use core::$f_scalar::EPSILON;
                let float_size = std::mem::size_of::<$f_scalar>() as u32 * 8;

                let value: $uty = rng.gen();
                let fraction = value >> (float_size - $fraction_bits);
                fraction.into_float_with_exponent(0) - (1.0 - EPSILON / 2.0)
            }
        }
    };
}

float_impls! { f32, u32, f32, u32, 23, 127 }
float_impls! { f64, u64, f64, u64, 52, 1023 }

pub trait IntoFloat {
    type F;

    /// Helper method to combine the fraction and a contant exponent into a
    /// float.
    ///
    /// Only the least significant bits of `self` may be set, 23 for `f32` and
    /// 52 for `f64`.
    /// The resulting value will fall in a range that depends on the exponent.
    /// As an example the range with exponent 0 will be
    /// [2<sup>0</sup>..2<sup>1</sup>), which is [1..2).
    fn into_float_with_exponent(self, exponent: i32) -> Self::F;
}

/// Trait for floating-point scalar types
///
/// This allows many distributions to work with `f32` or `f64` parameters and is
/// potentially extensible. Note however that the `Exp1` and `StandardNormal`
/// distributions are implemented exclusively for `f32` and `f64`.
///
/// The bounds and methods are based purely on internal
/// requirements, and will change as needed.
pub trait Float:
    Copy
    + Sized
    + cmp::PartialOrd
    + ops::Neg<Output = Self>
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::Div<Output = Self>
    + ops::AddAssign
    + ops::SubAssign
    + ops::MulAssign
    + ops::DivAssign
{
    /// The constant π
    fn pi() -> Self;
    /// Support approximate representation of a f64 value
    fn from(x: f64) -> Self;
    /// Support converting to an unsigned integer.
    fn to_u64(self) -> Option<u64>;

    /// Take the absolute value of self
    fn abs(self) -> Self;
    /// Take the largest integer less than or equal to self
    fn floor(self) -> Self;

    /// Take the exponential of self
    fn exp(self) -> Self;
    /// Take the natural logarithm of self
    fn ln(self) -> Self;
    /// Take square root of self
    fn sqrt(self) -> Self;
    /// Take self to a floating-point power
    fn powf(self, power: Self) -> Self;

    /// Take the tangent of self
    fn tan(self) -> Self;
    /// Take the logarithm of the gamma function of self
    fn log_gamma(self) -> Self;
}

impl Float for f32 {
    #[inline]
    fn pi() -> Self {
        core::f32::consts::PI
    }
    #[inline]
    fn from(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_u64(self) -> Option<u64> {
        if self >= 0. && self <= ::core::u64::MAX as f32 {
            Some(self as u64)
        } else {
            None
        }
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn floor(self) -> Self {
        self.floor()
    }

    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn powf(self, power: Self) -> Self {
        self.powf(power)
    }

    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline]
    fn log_gamma(self) -> Self {
        let result = log_gamma(self.into());
        assert!(result <= ::core::f32::MAX.into());
        assert!(result >= ::core::f32::MIN.into());
        result as f32
    }
}

impl Float for f64 {
    #[inline]
    fn pi() -> Self {
        core::f64::consts::PI
    }
    #[inline]
    fn from(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_u64(self) -> Option<u64> {
        if self >= 0. && self <= ::core::u64::MAX as f64 {
            Some(self as u64)
        } else {
            None
        }
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn floor(self) -> Self {
        self.floor()
    }

    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn powf(self, power: Self) -> Self {
        self.powf(power)
    }

    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline]
    fn log_gamma(self) -> Self {
        log_gamma(self)
    }
}

/// Calculates ln(gamma(x)) (natural logarithm of the gamma
/// function) using the Lanczos approximation.
///
/// The approximation expresses the gamma function as:
/// `gamma(z+1) = sqrt(2*pi)*(z+g+0.5)^(z+0.5)*exp(-z-g-0.5)*Ag(z)`
/// `g` is an arbitrary constant; we use the approximation with `g=5`.
///
/// Noting that `gamma(z+1) = z*gamma(z)` and applying `ln` to both sides:
/// `ln(gamma(z)) = (z+0.5)*ln(z+g+0.5)-(z+g+0.5) + ln(sqrt(2*pi)*Ag(z)/z)`
///
/// `Ag(z)` is an infinite series with coefficients that can be calculated
/// ahead of time - we use just the first 6 terms, which is good enough
/// for most purposes.
pub(crate) fn log_gamma(x: f64) -> f64 {
    // precalculated 6 coefficients for the first 6 terms of the series
    let coefficients: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    // (x+0.5)*ln(x+g+0.5)-(x+g+0.5)
    let tmp = x + 5.5;
    let log = (x + 0.5) * tmp.ln() - tmp;

    // the first few terms of the series for Ag(x)
    let mut a = 1.000000000190015;
    let mut denom = x;
    for &coeff in &coefficients {
        denom += 1.0;
        a += coeff / denom;
    }

    // get everything together
    // a is Ag(x)
    // 2.5066... is sqrt(2pi)
    log + (2.5066282746310005 * a / x).ln()
}

/// Sample a random number using the Ziggurat method (specifically the
/// ZIGNOR variant from Doornik 2005). Most of the arguments are
/// directly from the paper:
///
/// * `rng`: source of randomness
/// * `symmetric`: whether this is a symmetric distribution, or one-sided with P(x < 0) = 0.
/// * `X`: the $x_i$ abscissae.
/// * `F`: precomputed values of the PDF at the $x_i$, (i.e. $f(x_i)$)
/// * `F_DIFF`: precomputed values of $f(x_i) - f(x_{i+1})$
/// * `pdf`: the probability density function
/// * `zero_case`: manual sampling from the tail when we chose the
///    bottom box (i.e. i == 0)

// the perf improvement (25-50%) is definitely worth the extra code
// size from force-inlining.
#[inline(always)]
pub(crate) fn ziggurat<P, Z>(
    rng: &mut MyRng,
    symmetric: bool,
    x_tab: ziggurat_tables::ZigTable,
    f_tab: ziggurat_tables::ZigTable,
    mut pdf: P,
    mut zero_case: Z,
) -> f64
where
    P: FnMut(f64) -> f64,
    Z: FnMut(&mut MyRng, f64) -> f64,
{
    loop {
        // As an optimisation we re-implement the conversion to a f64.
        // From the remaining 12 most significant bits we use 8 to construct `i`.
        // This saves us generating a whole extra random number, while the added
        // precision of using 64 bits for f64 does not buy us much.
        let bits = rng.next_u64();
        let i = bits as usize & 0xff;

        let u = if symmetric {
            // Convert to a value in the range [2,4) and substract to get [-1,1)
            // We can't convert to an open range directly, that would require
            // substracting `3.0 - EPSILON`, which is not representable.
            // It is possible with an extra step, but an open range does not
            // seem neccesary for the ziggurat algorithm anyway.
            (bits >> 12).into_float_with_exponent(1) - 3.0
        } else {
            // Convert to a value in the range [1,2) and substract to get (0,1)
            (bits >> 12).into_float_with_exponent(0) - (1.0 - std::f64::EPSILON / 2.0)
        };
        let x = u * x_tab[i];

        let test_x = if symmetric { x.abs() } else { x };

        // algebraically equivalent to |u| < x_tab[i+1]/x_tab[i] (or u < x_tab[i+1]/x_tab[i])
        if test_x < x_tab[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        // algebraically equivalent to f1 + DRanU()*(f0 - f1) < 1
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * rng.gen::<f64>() < pdf(x) {
            return x;
        }
    }
}
