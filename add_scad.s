add:                                    # @add
        mov     rax, qword ptr [rsp + 48]
        mov     rcx, qword ptr [rsp + 8]
        xor     edx, edx
.LBB0_1:                                # %.preheader
        mov     rdi, -4
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
        mov     r8d, dword ptr [rcx + 4*rdi + 16]
        add     r8d, dword ptr [rsi + 4*rdi + 16]
        mov     dword ptr [rax + 4*rdi + 16], r8d
        mov     r8d, dword ptr [rcx + 4*rdi + 20]
        add     r8d, dword ptr [rsi + 4*rdi + 20]
        mov     dword ptr [rax + 4*rdi + 20], r8d
        mov     r8d, dword ptr [rcx + 4*rdi + 24]
        add     r8d, dword ptr [rsi + 4*rdi + 24]
        mov     dword ptr [rax + 4*rdi + 24], r8d
        mov     r8d, dword ptr [rcx + 4*rdi + 28]
        add     r8d, dword ptr [rsi + 4*rdi + 28]
        mov     dword ptr [rax + 4*rdi + 28], r8d
        add     rdi, 4
        cmp     rdi, 99996
        jb      .LBB0_2
        mov     edi, dword ptr [rsi + 4*rdx]
        add     edi, dword ptr [rax + 4*rdx]
        add     edi, dword ptr [rcx + 4*rdx]
        mov     dword ptr [rax + 4*rdx], edi
        cmp     rdx, 99999
        lea     rdx, [rdx + 1]
        jb      .LBB0_1
        mov     eax, 1000000
        retarith.addi