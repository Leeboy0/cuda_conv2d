"""
gen_testdata.py — Generate conv2d test cases using PyTorch as ground truth.

For each test case, writes four files to ./testdata/:
    caseNN_input.bin     — input tensor  [N, C_in, H, W]    float32
    caseNN_weight.bin    — weight tensor [C_out, C_in, KH, KW] float32
    caseNN_expected.bin  — PyTorch's conv2d output  [N, C_out, OH, OW] float32
    caseNN_shape.txt     — one line: N C_in C_out H W KH KW pad stride OH OW

Run once:
    python gen_testdata.py
Then your CUDA test loads these files and compares its output against *_expected.bin.
"""

import os
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# (N, C_in, C_out, H, W, KH, KW, pad, stride)
cases = [
    (1,  1,  1,   8,  8, 3, 3, 1, 1),   # tiny, padded
    (1,  1,  1,   8,  8, 3, 3, 0, 1),   # tiny, no padding
    (1,  2,  2,  10, 12, 3, 5, 1, 1),   # asymmetric kernel
    (1,  3,  4,   7, 11, 3, 3, 1, 1),   # non-square input
    (1,  3,  4,  16, 16, 3, 3, 1, 2),   # stride 2
    (2,  4,  8,  16, 16, 1, 1, 0, 1),   # 1x1 pointwise conv
    (4, 16, 32,  32, 32, 3, 3, 1, 1),   # realistic
]

os.makedirs("testdata", exist_ok=True)

for i, (N, C_in, C_out, H, W, KH, KW, pad, stride) in enumerate(cases):
    inp = torch.randn(N, C_in, H, W, dtype=torch.float32)
    w   = torch.randn(C_out, C_in, KH, KW, dtype=torch.float32)
    out = F.conv2d(inp, w, padding=pad, stride=stride)  # ground truth

    OH, OW = out.shape[2], out.shape[3]

    tag = f"case{i:02d}"
    inp.numpy().tofile(f"testdata/{tag}_input.bin")
    w.numpy().tofile(f"testdata/{tag}_weight.bin")
    out.numpy().tofile(f"testdata/{tag}_expected.bin")
    with open(f"testdata/{tag}_shape.txt", "w") as f:
        f.write(f"{N} {C_in} {C_out} {H} {W} {KH} {KW} {pad} {stride} {OH} {OW}\n")

    print(f"{tag}: N={N} Cin={C_in} Cout={C_out} H={H} W={W} "
          f"KH={KH} KW={KW} pad={pad} stride={stride}  -> OH={OH} OW={OW}")

print(f"\nWrote {len(cases)} test cases to ./testdata/")