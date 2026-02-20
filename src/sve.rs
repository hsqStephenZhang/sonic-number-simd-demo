
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
            // 1. 准备谓词
            "whilelo p0.b, xzr, {need}",   // p0 限制在 need 长度，保障内存读取安全
            "ptrue p2.b, vl16",            // p2 是全真谓词，用于后续完整的权重计算

            // 2. 加载数据与匹配数字
            "ld1b {{z0.b}}, p0/z, [{ptr}]",
            "ld1b {{z1.b}}, p2/z, [{digits_ptr}]",
            "match p1.b, p0/z, z0.b, z1.b",
            "not p1.b, p0/z, p1.b",
            "brkb p1.b, p0/z, p1.b",
            "cntp {count}, p0, p1.b",

            // 3. ASCII -> 数值 (公共步骤)
            "sub z0.b, z0.b, #48",

            // --- 分支：判断 count 是否 > 8 ---
            "cmp {count}, #8",
            "b.hi 2f", // 如果 > 8，跳转到标签 2 (16位逻辑)

            // ==========================================
            // 标签 1: <= 8 位逻辑
            // ==========================================
            "1:",
            "mov {shift}, #8",
            "sub {shift}, {shift}, {count}",     // shift = 8 - count
            "dup z6.b, {shift:w}",
            "index z7.b, #0, #1",
            "sub z7.b, z7.b, z6.b",
            "tbl z0.b, z0.b, z7.b",              // 右对齐低 8 字节

            "uunpklo z2.h, z0.b",
            "ld1h {{z3.h}}, p2/z, [{w8_ptr}]",   // 使用 p2 加载全部权重
            "movi d4, #0",
            "sdot z4.d, z3.h, z2.h",

            "ld1d {{z5.d}}, p2/z, [{w2_ptr}]",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {res}, d6",
            "b 3f",                              // 跳转到结束

            // ==========================================
            // 标签 2: > 8 位 (9-16位) 逻辑
            // ==========================================
            "2:",
            "mov {shift}, #16",
            "sub {shift}, {shift}, {count}",     // shift = 16 - count
            "dup z6.b, {shift:w}",
            "index z7.b, #0, #1",
            "sub z7.b, z7.b, z6.b",
            "tbl z0.b, z0.b, z7.b",              // 右对齐全部 16 个字节

            "ld1h {{z3.h}}, p2/z, [{w8_ptr}]",
            "ld1d {{z5.d}}, p2/z, [{w2_ptr}]",

            // [处理高 8 字节] 提取高位
            "uunpklo z2.h, z0.b",
            "movi d4, #0",
            "sdot z4.d, z3.h, z2.h",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {tmp}, d6",                    // 暂存高位结果到 {tmp}

            // [处理低 8 字节] 提取低位
            "uunpkhi z2.h, z0.b",                // uunpkhi 取后 8 个字节
            "movi d4, #0",
            "sdot z4.d, z3.h, z2.h",
            "mul z4.d, p2/m, z4.d, z5.d",
            "uaddv d6, p2, z4.d",
            "fmov {res}, d6",                    // 暂存低位结果到 {res}

            // 标量合并: res = tmp * 10^8 + res
            "madd {res}, {tmp}, {pow8}, {res}",

            // ==========================================
            // 标签 3: 结束
            // ==========================================
            "3:",

            ptr = in(reg) c.as_ptr(),
            digits_ptr = in(reg) ldigits.as_ptr(),
            w8_ptr = in(reg) d8d.as_ptr(),
            w2_ptr = in(reg) d2d.as_ptr(),
            need = in(reg) need,
            pow8 = in(reg) 100_000_000_u64, // 通过寄存器传入 10^8 常量
            count = out(reg) count,
            res = out(reg) res,
            shift = out(reg) _,             // 临时寄存器，无需绑定外部变量
            tmp = out(reg) _,               // 用于暂存高位结果
            out("z0") _, out("z1") _, out("z2") _, out("z3") _,
            out("z4") _, out("z5") _, out("z6") _, out("z7") _,
            out("p0") _, out("p1") _, out("p2") _
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
