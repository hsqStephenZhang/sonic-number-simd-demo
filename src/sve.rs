const D8D: [i16; 8] = [1000, 100, 10, 1, 1000, 100, 10, 1];
const D2D: [i64; 2] = [10000, 1];
const LDIGITS: [u8; 16] = [
    b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'0', 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/// use `splice` to align lanes to the right
/// there are two branches to handle <=8 and >8 digits
/// where in cpp's version, it's handled in a big switch-case
#[inline(always)]
pub unsafe fn simd_str2int_sve2(c: &[u8], need: usize) -> (u64, usize) {
    let mut count: u64;
    let mut res: u64;

    unsafe {
        core::arch::asm!(
            "whilelo p0.b, xzr, {need}",
            "ptrue p2.b, vl16",

            "ld1b {{z0.b}}, p0/z, [{ptr}]",
            "ld1b {{z1.b}}, p2/z, [{digits_ptr}]",
            "match p1.b, p0/z, z0.b, z1.b",
            "not p1.b, p0/z, p1.b",
            "brkb p1.b, p0/z, p1.b",
            "cntp {count}, p0, p1.b",

            "sub z0.b, z0.b, #48",
            "dup z7.b, #0",

            "cmp {count}, #8",
            "b.hi 2f",

            // <= 8
            "1:",
            "uunpklo z2.h, z0.b",
            "mov {shift}, #8",
            "sub {shift}, {shift}, {count}",
            "whilelo p3.h, xzr, {shift}",
            "splice z7.h, p3, z7.h, z2.h",

            "ld1h {{z3.h}}, p2/z, [{w8_ptr}]",
            "dup z4.d, #0",
            "sdot z4.d, z3.h, z7.h",

            "ld1d {{z5.d}}, p2/z, [{w2_ptr}]",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {res}, d6",
            "b 3f",

            // 9-16
            "2:",
            "mov {shift}, #16",
            "sub {shift}, {shift}, {count}",
            "whilelo p3.b, xzr, {shift}",
            "splice z7.b, p3, z7.b, z0.b",

            "ld1h {{z3.h}}, p2/z, [{w8_ptr}]",
            "ld1d {{z5.d}}, p2/z, [{w2_ptr}]",

            "uunpklo z2.h, z7.b",
            "dup z4.d, #0",
            "sdot z4.d, z3.h, z2.h",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {tmp}, d6",

            "uunpkhi z2.h, z7.b",
            "dup z4.d, #0",
            "sdot z4.d, z3.h, z2.h",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {res}, d6",

            "madd {res}, {tmp}, {pow8}, {res}",

            "3:",

            ptr = in(reg) c.as_ptr(),
            digits_ptr = in(reg) LDIGITS.as_ptr(),
            w8_ptr = in(reg) D8D.as_ptr(),
            w2_ptr = in(reg) D2D.as_ptr(),
            need = in(reg) need,
            pow8 = in(reg) 100_000_000_u64,
            count = out(reg) count,
            res = out(reg) res,
            shift = out(reg) _,
            tmp = out(reg) _,
            out("z0") _, out("z1") _, out("z2") _, out("z3") _,
            out("z4") _, out("z5") _, out("z6") _, out("z7") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _,

            // 核心优化 2：赋予 LLVM 最高级别的优化权限！
            options(pure, readonly, nostack)
        );

        (res, count as usize)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_str2int_sve2_optimized() {
        for _ in 0..1000 {
            let length = rand::random::<usize>() % 8 + 1; // 1-16 位随机长度
            let s: String = (0..length)
                .map(|_| (b'0' + (rand::random::<u8>() % 10)) as char)
                .collect();
            let expected_num: u64 = s.parse().unwrap();
            let (num, count) = unsafe { simd_str2int_sve2(s.as_bytes(), s.len()) };
            assert_eq!(count, length, "Length mismatch for input '{}'", s);
            assert_eq!(num, expected_num, "Value mismatch for input '{}'", s);
        }
    }

    #[test]
    fn t1() {
        let s = "666861";
        let expected_num: u64 = s.parse().unwrap();
        let (num, count) = unsafe { simd_str2int_sve2(s.as_bytes(), s.len()) };
        assert_eq!(count, s.len(), "Length mismatch for input '{}'", s);
        assert_eq!(num, expected_num, "Value mismatch for input '{}'", s);
    }
}
