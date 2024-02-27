add:
        push    {r4, r5, r6, r7, r8, lr}
        mov     r12, #1696
        mov     lr, #0
        orr     r12, r12, #98304
.LBB0_1:                                @ =>This Loop Header: Depth=1
        mov     r3, r0
        mov     r4, r1
        mov     r5, r2
        mov     r6, r12
.LBB0_2:                                @   Parent Loop BB0_1 Depth=1
        ldr     r8, [r3], #4
        ldr     r7, [r4], #4
        subs    r6, r6, #1
        add     r7, r7, r8
        str     r7, [r5], #4
        bne     .LBB0_2
        ldr     r3, [r2, lr, lsl #2]
        ldr     r4, [r0, lr, lsl #2]
        add     r3, r4, r3
        ldr     r4, [r1, lr, lsl #2]
        add     r3, r3, r4
        str     r3, [r2, lr, lsl #2]
        add     lr, lr, #1
        cmp     lr, r12
        bne     .LBB0_1
        mov     r0, #0
        pop     {r4, r5, r6, r7, r8, lr}
        bx      lr