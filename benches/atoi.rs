// Benchmark comparison between atoi_simd and sonic-number
use std::hint::black_box;

use atoi_simd::parse as atoi_simd_parse;
use criterion::{Criterion, criterion_group, criterion_main};

const NUMBERS: &[&str] = &[
    "1",
    "12",
    "123",
    "1234",
    "12345",
    "123456",
    "1234567",
    "12345678",
    "123456789",
    "1234567890",
    "12345678901",
    "123456789012",
    "1234567890123",
    "12345678901234",
    "123456789012345",
    "1234567890123456", // max 16 digits for `simd_str2int`
];

fn bench_i64(c: &mut Criterion) {
    for (i, s) in NUMBERS.iter().enumerate() {
        let expected = s.parse::<u64>().unwrap();
        let bench_name_neon = format!("sonic-number neon len={}", i + 1);
        let bench_name_sve = format!("sonic-number sve len={}", i + 1);
        let bench_name_atoi = format!("atoi_simd parse len={}", i + 1);
        let bench_name_sse = format!("sonic-number sse len={}", i + 1);

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        c.bench_function(&bench_name_neon, |b| {
            b.iter(|| {
                let val = unsafe {
                    sonic_number_simd::neon::simd_str2int_neon(black_box(s.as_bytes()), s.len())
                };
                assert_eq!(val.0, expected);
            })
        });

        #[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
        c.bench_function(&bench_name_sve, |b| {
            b.iter(|| {
                let val = unsafe {
                    sonic_number_simd::sve::simd_str2int_sve2(black_box(s.as_bytes()), s.len())
                };
                assert_eq!(val.0, expected);
            })
        });

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "pclmulqdq",
            target_feature = "avx2",
            target_feature = "sse2"
        ))]
        c.bench_function(&bench_name_sse, |b| {
            b.iter(|| {
                let val = unsafe {
                    sonic_number_simd::x86_64::simd_str2int(black_box(s.as_bytes()), s.len())
                };
                assert_eq!(val.0, expected);
            })
        });

        c.bench_function(&bench_name_atoi, |b| {
            b.iter(|| {
                let val = atoi_simd_parse::<u64, false, false>(black_box(s.as_bytes())).unwrap();
                assert_eq!(val, expected);
            })
        });
    }
}

criterion_group!(benches, bench_i64);
criterion_main!(benches);
