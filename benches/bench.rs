#![feature(test)]

extern crate test;

use my_fft::fft;
use num::{Complex, Zero};
use rand::Rng;

#[bench]
fn bench_fft_1e16(b: &mut test::Bencher) {
    let mut rng = rand::thread_rng();
    let input: Vec<Complex<f64>> = (0..1 << 16).map(|_| Complex::new(rng.gen(), 0.0)).collect();

    b.iter(move || {
        fft(&input);
    });
}

#[bench]
fn bench_rust_fft_1e16(b: &mut test::Bencher) {
    let mut rng = rand::thread_rng();
    let mut input: Vec<Complex<f64>> = (0..1 << 16).map(|_| Complex::new(rng.gen(), 0.0)).collect();
    let mut output = vec![Complex::zero(); input.len()];

    b.iter(move || {
        use rustfft::FFTplanner;

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(input.len());

        fft.process(&mut input, &mut output);
    });
}
