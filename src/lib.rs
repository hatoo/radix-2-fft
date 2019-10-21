use num::{traits::float::FloatConst, Complex, Float};
use rayon::prelude::*;

pub fn fft<T: Float + FloatConst + Send + Sync>(array: &[Complex<T>]) -> Vec<Complex<T>> {
    let len = array.len();
    assert!(len.count_ones() == 1);

    let w: Vec<Complex<T>> = (0..len)
        .into_par_iter()
        .map(|i| {
            Complex::new(
                T::zero(),
                -T::from(2).unwrap() * T::PI() * T::from(i).unwrap() / T::from(len).unwrap(),
            )
            .exp()
        })
        .collect();

    let ltz = len.trailing_zeros();
    // Step 1
    let buf: Vec<Complex<T>> = (0..len)
        .into_par_iter()
        .map(|i| {
            if i & 1 == 0 {
                let fi = i.reverse_bits().rotate_left(ltz);
                let fi1 = (i + 1).reverse_bits().rotate_left(ltz);
                *unsafe { array.get_unchecked(fi) } + unsafe { array.get_unchecked(fi1) }
            } else {
                let fi_1 = (i - 1).reverse_bits().rotate_left(ltz);
                let fi = i.reverse_bits().rotate_left(ltz);
                *unsafe { array.get_unchecked(fi_1) } - unsafe { array.get_unchecked(fi) }
            }
        })
        .collect();

    let mut current = buf;
    let mut swap = Vec::with_capacity(len);

    for x in 1..ltz {
        let l = 1 << x;
        let tmp = len >> (x + 1);

        current.par_iter_mut().enumerate().for_each(|(i, b)| {
            if (i / l) & 1 == 1 {
                let idx = (i % l) * tmp;
                let weight = unsafe { w.get_unchecked(idx) };
                *b = *b * weight;
            }
        });

        (0..len)
            .into_par_iter()
            .map(|i| {
                if (i / l) & 1 == 0 {
                    *unsafe { current.get_unchecked(i) } + unsafe { current.get_unchecked(i + l) }
                } else {
                    -*unsafe { current.get_unchecked(i) } + unsafe { current.get_unchecked(i - l) }
                }
            })
            .collect_into_vec(&mut swap);

        std::mem::swap(&mut current, &mut swap);
    }

    current
}

#[cfg(test)]
mod tests {
    use crate::fft;
    use num::{Complex, Zero};
    use rand::Rng;

    fn do_rust_fft(input: &mut [Complex<f64>]) -> Vec<Complex<f64>> {
        use rustfft::FFTplanner;

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(input.len());
        let mut output = vec![Complex::zero(); input.len()];

        fft.process(input, &mut output);
        output
    }

    fn cross_check(input: &[Complex<f64>]) {
        let actual = fft(input);
        let expected = do_rust_fft(&mut input.to_vec());

        assert_eq!(actual.len(), expected.len());

        dbg!(&actual);
        dbg!(&expected);

        for (x, y) in actual.into_iter().zip(expected.into_iter()) {
            assert!((x - y).norm() < 1e-9);
        }
    }

    #[test]
    fn it_works() {
        cross_check(&vec![
            Complex::<f64>::new(0.3535, 0.0),
            Complex::<f64>::new(0.3535, 0.0),
            Complex::<f64>::new(0.6464, 0.0),
            Complex::<f64>::new(1.0607, 0.0),
            Complex::<f64>::new(0.3535, 0.0),
            Complex::<f64>::new(-1.0607, 0.0),
            Complex::<f64>::new(-1.3535, 0.0),
            Complex::<f64>::new(-0.3535, 0.0),
        ]);
    }

    #[test]
    fn test_rand_1024() {
        let mut rng = rand::thread_rng();
        let input: Vec<Complex<f64>> = (0..1024).map(|_| Complex::new(rng.gen(), 0.0)).collect();

        cross_check(&input);
    }
}
