#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve2")]
pub unsafe fn simd_str2int_sve2(c: &[u8], need: usize) -> (u64, usize) {
    let d8d: [i16; 8] = [1000, 100, 10, 1, 1000, 100, 10, 1];
    let d2d: [i64; 2] = [10000, 1];
    let ldigits: [u8; 16] = [
        b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'0', 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF,
    ];

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
            "dup z7.b, #0",                  // 核心技巧：准备一个全 0 向量 z7

            "cmp {count}, #8",
            "b.hi 2f",                       // 整个逻辑只有一个极简分支

            // ==========================================
            // 标签 1: <= 8 位逻辑
            // ==========================================
            "1:",
            "uunpklo z2.h, z0.b",            // 解包前 8 字节为 16-bit 存入 z2
            "mov {shift}, #8",
            "sub {shift}, {shift}, {count}", // shift = 8 - count (即需要补几个 0)
            "whilelo p3.h, xzr, {shift}",    // 生成前导补零谓词
            "splice z7.h, p3, z7.h, z2.h",   // 将全0向量(z7)和数据(z2)拼接，结果必须存回 z7！

            "ld1h {{z3.h}}, p2/z, [{w8_ptr}]",
            "dup z4.d, #0",                  // 严格清空 128 位累加器
            "sdot z4.d, z3.h, z7.h",         // 注意：使用拼接后的 z7.h 参与计算！

            "ld1d {{z5.d}}, p2/z, [{w2_ptr}]",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {res}, d6",
            "b 3f",

            // ==========================================
            // 标签 2: > 8 位逻辑 (9-16位)
            // ==========================================
            "2:",
            "mov {shift}, #16",
            "sub {shift}, {shift}, {count}", // shift = 16 - count
            "whilelo p3.b, xzr, {shift}",    
            "splice z7.b, p3, z7.b, z0.b",   // 对完整的 16 字节字符进行右对齐操作，结果存回 z7！

            "ld1h {{z3.h}}, p2/z, [{w8_ptr}]",
            "ld1d {{z5.d}}, p2/z, [{w2_ptr}]",

            // [提取高 8 字节计算]
            "uunpklo z2.h, z7.b",            // 注意：从已经对齐的 z7 中解包高位放入 z2
            "dup z4.d, #0",
            "sdot z4.d, z3.h, z2.h",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {tmp}, d6",

            // [提取低 8 字节计算]
            "uunpkhi z2.h, z7.b",            // 注意：从已经对齐的 z7 中解包低位放入 z2
            "dup z4.d, #0",
            "sdot z4.d, z3.h, z2.h",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {res}, d6",

            // 标量合并结果 (高位永远乘 10^8)
            "madd {res}, {tmp}, {pow8}, {res}",

            "3:",

            ptr = in(reg) c.as_ptr(),
            digits_ptr = in(reg) ldigits.as_ptr(),
            w8_ptr = in(reg) d8d.as_ptr(),
            w2_ptr = in(reg) d2d.as_ptr(),
            need = in(reg) need,
            pow8 = in(reg) 100_000_000_u64,
            count = out(reg) count,
            res = out(reg) res,
            shift = out(reg) _,
            tmp = out(reg) _,
            out("z0") _, out("z1") _, out("z2") _, out("z3") _,
            out("z4") _, out("z5") _, out("z6") _, out("z7") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _
        )
    };
    (res, count as usize)
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
