cadd:
        push    {r4, lr}
        ldr     r12, .LCPI0_0
        mov     lr, #0
.LBB0_1:                                @ =>This Loop Header: Depth=1
        mov     r3, #0
.LBB0_2:                                @   Parent Loop BB0_1 Depth=1
        ldr     r1, [r0, r3, lsl #2]
        ldr     r4, [r2, r3, lsl #2]
        add     r1, r4, r1
        str     r1, [r2, r3, lsl #2]
        add     r3, r3, #1
        cmp     r3, #128
        bne     .LBB0_2
        add     lr, lr, #1
        cmp     lr, r12
        bne     .LBB0_1
        mov     r0, #0
        pop     {r4, lr}
        bx      lr
.LCPI0_0:
        .long   10000000                        @ 0x989680