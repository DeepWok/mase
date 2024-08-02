import generate_memory as gm

fs = ["elu", "sigmoid", "logsigmoid", "softshrink"]

d_width = 16
f_width = 8

d_widths = [1, 2, 4, 6, 8, 10, 12, 14, 16]
f_widths = [0, 1, 2, 3, 4, 5, 6, 7, 8]
for f in fs:
    gm.generate_sv_lut(f, d_width, f_width, dir="/workspace/luts_for_test")

for f in ["silu"]:
    for d, frac in zip(d_widths, f_widths):
        gm.generate_sv_lut(f, d, frac, dir="/workspace/luts_for_test")
