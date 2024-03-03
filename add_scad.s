
0000000100003d04 <_cadd>:
100003d04: d2800008     mov     x8, #0
100003d08: 5290d409     mov     w9, #34464
100003d0c: 72a00029     movk    w9, #1, lsl #16
100003d10: cb00004a     sub     x10, x2, x0
100003d14: cb01004b     sub     x11, x2, x1
100003d18: f101015f     cmp     x10, #64
100003d1c: 5280080a     mov     w10, #64
100003d20: fa4a2160     ccmp    x11, x10, #0, hs
100003d24: 1a9f27ea     cset    w10, lo
100003d28: 9100804b     add     x11, x2, #32
100003d2c: 9100800c     add     x12, x0, #32
100003d30: 9100802d     add     x13, x1, #32
100003d34: 1400000b     b       0x100003d60 <_cadd+0x5c>
100003d38: d37ef50e     lsl     x14, x8, #2
100003d3c: b86e684f     ldr     w15, [x2, x14]
100003d40: b86e6810     ldr     w16, [x0, x14]
100003d44: b86e6831     ldr     w17, [x1, x14]
100003d48: 0b0f020f     add     w15, w16, w15
100003d4c: 0b1101ef     add     w15, w15, w17
100003d50: b82e684f     str     w15, [x2, x14]
100003d54: 91000508     add     x8, x8, #1
100003d58: eb09011f     cmp     x8, x9
100003d5c: 54000400     b.eq    0x100003ddc <_cadd+0xd8>
100003d60: aa0d03ee     mov     x14, x13
100003d64: aa0c03ef     mov     x15, x12
100003d68: aa0b03f0     mov     x16, x11
100003d6c: 5290d411     mov     w17, #34464
100003d70: 72a00031     movk    w17, #1, lsl #16
100003d74: aa0003e3     mov     x3, x0
100003d78: aa0103e4     mov     x4, x1
100003d7c: aa0203e5     mov     x5, x2
100003d80: 5290d406     mov     w6, #34464
100003d84: 72a00026     movk    w6, #1, lsl #16
100003d88: 3600010a     tbz     w10, #0, 0x100003da8 <_cadd+0xa4>
100003d8c: b840446e     ldr     w14, [x3], #4
100003d90: b840448f     ldr     w15, [x4], #4
100003d94: 0b0e01ee     add     w14, w15, w14
100003d98: b80044ae     str     w14, [x5], #4
100003d9c: f10004c6     subs    x6, x6, #1
100003da0: 54ffff61     b.ne    0x100003d8c <_cadd+0x88>
100003da4: 17ffffe5     b       0x100003d38 <_cadd+0x34>
100003da8: ad7f05e0     ldp     q0, q1, [x15, #-32]
100003dac: acc20de2     ldp     q2, q3, [x15], #64
100003db0: ad7f15c4     ldp     q4, q5, [x14, #-32]
100003db4: acc21dc6     ldp     q6, q7, [x14], #64
100003db8: 4ea08480     add.4s  v0, v4, v0
100003dbc: 4ea184a1     add.4s  v1, v5, v1
100003dc0: 4ea284c2     add.4s  v2, v6, v2
100003dc4: 4ea384e3     add.4s  v3, v7, v3
100003dc8: ad3f0600     stp     q0, q1, [x16, #-32]
100003dcc: ac820e02     stp     q2, q3, [x16], #64
100003dd0: f1004231     subs    x17, x17, #16
100003dd4: 54fffea1     b.ne    0x100003da8 <_cadd+0xa4>
100003dd8: 17ffffd8     b       0x100003d38 <_cadd+0x34>
100003ddc: 52800000     mov     w0, #0
100003de0: d65f03c0     ret

0000000100003de4 <_main>:
100003de4: d10203ff     sub     sp, sp, #128
100003de8: a9045ff8     stp     x24, x23, [sp, #64]
100003dec: a90557f6     stp     x22, x21, [sp, #80]
100003df0: a9064ff4     stp     x20, x19, [sp, #96]
100003df4: a9077bfd     stp     x29, x30, [sp, #112]
100003df8: 9101c3fd     add     x29, sp, #112
100003dfc: 52835000     mov     w0, #6784
100003e00: 72a000c0     movk    w0, #6, lsl #16
100003e04: 94000055     bl      0x100003f58 <_printf+0x100003f58>
100003e08: aa0003f3     mov     x19, x0
100003e0c: 52835000     mov     w0, #6784
100003e10: 72a000c0     movk    w0, #6, lsl #16
100003e14: 94000051     bl      0x100003f58 <_printf+0x100003f58>
100003e18: aa0003f4     mov     x20, x0
100003e1c: 52800037     mov     w23, #1
100003e20: 52800020     mov     w0, #1
100003e24: 52835001     mov     w1, #6784
100003e28: 72a000c1     movk    w1, #6, lsl #16
100003e2c: 94000048     bl      0x100003f4c <_printf+0x100003f4c>
100003e30: aa0003f5     mov     x21, x0
100003e34: 90000016     adrp    x22, 0x100003000 <_main+0x50>
100003e38: 913e02d6     add     x22, x22, #3968
100003e3c: aa1303e0     mov     x0, x19
100003e40: aa1603e1     mov     x1, x22
100003e44: 52835002     mov     w2, #6784
100003e48: 72a000c2     movk    w2, #6, lsl #16
100003e4c: 94000046     bl      0x100003f64 <_printf+0x100003f64>
100003e50: aa1403e0     mov     x0, x20
100003e54: aa1603e1     mov     x1, x22
100003e58: 52835002     mov     w2, #6784
100003e5c: 72a000c2     movk    w2, #6, lsl #16
100003e60: 94000041     bl      0x100003f64 <_printf+0x100003f64>
100003e64: 5290d408     mov     w8, #34464
100003e68: 72a00028     movk    w8, #1, lsl #16
100003e6c: a902dfe8     stp     x8, x23, [sp, #40]
100003e70: a901fff5     stp     x21, xzr, [sp, #24]
100003e74: a900d7f7     stp     x23, x21, [sp, #8]
100003e78: f90003e8     str     x8, [sp]
100003e7c: aa1303e0     mov     x0, x19
100003e80: aa1303e1     mov     x1, x19
100003e84: d2800002     mov     x2, #0
100003e88: 5290d403     mov     w3, #34464
100003e8c: 72a00023     movk    w3, #1, lsl #16
100003e90: 52800024     mov     w4, #1
100003e94: aa1403e5     mov     x5, x20
100003e98: aa1403e6     mov     x6, x20
100003e9c: d2800007     mov     x7, #0
100003ea0: 9400000f     bl      0x100003edc <_add>
100003ea4: 52834f88     mov     w8, #6780
100003ea8: 72a000c8     movk    w8, #6, lsl #16
100003eac: b8686aa8     ldr     w8, [x21, x8]
100003eb0: f90003e8     str     x8, [sp]
100003eb4: 90000000     adrp    x0, 0x100003000 <_main+0xd0>
100003eb8: 913df000     add     x0, x0, #3964
100003ebc: 9400002d     bl      0x100003f70 <_printf+0x100003f70>
100003ec0: 52800000     mov     w0, #0
100003ec4: a9477bfd     ldp     x29, x30, [sp, #112]
100003ec8: a9464ff4     ldp     x20, x19, [sp, #96]
100003ecc: a94557f6     ldp     x22, x21, [sp, #80]
100003ed0: a9445ff8     ldp     x24, x23, [sp, #64]
100003ed4: 910203ff     add     sp, sp, #128
100003ed8: d65f03c0     ret

0000000100003edc <_add>:
100003edc: 5290d3ea     mov     w10, #34463
100003ee0: aa1f03e9     mov     x9, xzr
100003ee4: f9400fe8     ldr     x8, [sp, #24]
100003ee8: 72a0002a     movk    w10, #1, lsl #16
100003eec: aa1f03eb     mov     x11, xzr
100003ef0: d37ef56c     lsl     x12, x11, #2
100003ef4: 9100056f     add     x15, x11, #1
100003ef8: eb0a017f     cmp     x11, x10
100003efc: aa0f03eb     mov     x11, x15
100003f00: b86c682d     ldr     w13, [x1, x12]
100003f04: b86c68ce     ldr     w14, [x6, x12]
100003f08: 0b0d01cd     add     w13, w14, w13
100003f0c: b82c690d     str     w13, [x8, x12]
100003f10: 54ffff03     b.lo    0x100003ef0 <_add+0x14>
100003f14: d37ef52b     lsl     x11, x9, #2
100003f18: 9100052f     add     x15, x9, #1
100003f1c: eb0a013f     cmp     x9, x10
100003f20: b86b690c     ldr     w12, [x8, x11]
100003f24: b86b682d     ldr     w13, [x1, x11]
100003f28: b86b68ce     ldr     w14, [x6, x11]
100003f2c: 0b0c01a9     add     w9, w13, w12
100003f30: 0b0e012c     add     w12, w9, w14
100003f34: aa0f03e9     mov     x9, x15
100003f38: b82b690c     str     w12, [x8, x11]
100003f3c: 54fffd83     b.lo    0x100003eec <_add+0x10>
100003f40: 52884800     mov     w0, #16960
100003f44: 72a001e0     movk    w0, #15, lsl #16
100003f48: d65f03c0     ret

Disassembly of section __TEXT,__stubs:

0000000100003f4c <__stubs>:
100003f4c: b0000010     adrp    x16, 0x100004000 <__stubs+0x4>
100003f50: f9400210     ldr     x16, [x16]
100003f54: d61f0200     br      x16
100003f58: b0000010     adrp    x16, 0x100004000 <__stubs+0x10>
100003f5c: f9400610     ldr     x16, [x16, #8]
100003f60: d61f0200     br      x16
100003f64: b0000010     adrp    x16, 0x100004000 <__stubs+0x1c>
100003f68: f9400a10     ldr     x16, [x16, #16]
100003f6c: d61f0200     br      x16
100003f70: b0000010     adrp    x16, 0x100004000 <__stubs+0x28>
100003f74: f9400e10     ldr     x16, [x16, #24]
100003f78: d61f0200     br      x16

Disassembly of section __TEXT,__cstring:

0000000100003f7c <__cstring>:
100003f7c: 000a6925     <unknown>

Disassembly of section __TEXT,__const:

0000000100003f80 <__const>:
100003f80: 0000000a     udf     #10
100003f84: 0000000a     udf     #10
100003f88: 0000000a     udf     #10
100003f8c: 0000000a     udf     #10

Disassembly of section __TEXT,__unwind_info:

0000000100003f90 <__unwind_info>:
100003f90: 00000001     udf     #1
100003f94: 0000001c     udf     #28
100003f98: 00000000     udf     #0
100003f9c: 0000001c     udf     #28
100003fa0: 00000000     udf     #0
100003fa4: 0000001c     udf     #28
100003fa8: 00000002     udf     #2
100003fac: 00003d04     udf     #15620
100003fb0: 00000040     udf     #64
100003fb4: 00000040     udf     #64
100003fb8: 00003f4c     udf     #16204
100003fbc: 00000000     udf     #0
100003fc0: 00000040     udf     #64
                ...
100003fd0: 00000003     udf     #3
100003fd4: 0003000c     <unknown>
100003fd8: 00030018     <unknown>
100003fdc: 00000000     udf     #0
100003fe0: 010000e0     <unknown>
100003fe4: 020001d8     <unknown>
100003fe8: 02000000     <unknown>
100003fec: 04000007     add     z7.b, p0/m, z7.b, z0.b
                ...

Disassembly of section __DATA_CONST,__got:

0000000100004000 <__got>:
100004000: 00000000     udf     #0
100004004: 80100000     <unknown>
100004008: 00000001     udf     #1
10000400c: 80100000     <unknown>
100004010: 00000002     udf     #2
100004014: 80100000     <unknown>
100004018: 00000003     udf     #3
10000401c: 80000000     <unknown>