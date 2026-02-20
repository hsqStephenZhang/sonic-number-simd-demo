## what's it

simd based strtoint implementation, support neon and sve2 on arm, x86 is ignored.

## benchmark

### machine

AWS Graviton4 AWS Graviton4 CPU @ 2.8GHz, Neoverse-V2, 2 Core, 8GB Memory.

### result

```
sonic-number neon       time:   [10.444 ns 10.445 ns 10.445 ns]
                        change: [−1.1025% −1.0766% −1.0462%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  1 (1.00%) low severe
  2 (2.00%) low mild
  1 (1.00%) high mild
  6 (6.00%) high severe

sonic-number sve        time:   [9.6467 ns 9.6475 ns 9.6483 ns]
                        change: [−84.254% −84.252% −84.249%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 5 outliers among 100 measurements (5.00%)
  1 (1.00%) low severe
  1 (1.00%) low mild
  1 (1.00%) high mild
  2 (2.00%) high severe

atoi_simd parse         time:   [10.652 ns 10.654 ns 10.657 ns]
                        change: [−2.1087% −1.8645% −1.6100%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 9 outliers among 100 measurements (9.00%)
  4 (4.00%) high mild
  5 (5.00%) high severe
```
