from mase_components.helper.generate_memory import (
    generate_sv_lut,
    FUNCTION_TABLE,
)


def generate_activation_luts():
    for func, _ in FUNCTION_TABLE.items():
        generate_sv_lut(func, 8, 4, data_width=8, f_width=4, path_with_dtype=False)
        print(f"Generating LUT for {func} activation")


if __name__ == "__main__":
    generate_activation_luts()
