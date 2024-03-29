from pathlib import Path
import re
import pandas as pd
import altair as alt
import numpy as np


def timing_util(folder: Path):
    print("-----", folder)
    timingrpt = folder / "timing.rpt"
    utilrpt = folder / "utilization.rpt"
    if not (timingrpt.exists() and utilrpt.exists()):
        return None

    # Parse Timing Report
    with open(timingrpt, "r") as f:
        timing_text = f.readlines()

    design_timing_sum_line = None
    clock_summary_line = None
    for line_num, line in enumerate(timing_text):
        if line.find("Design Timing Summary") != -1:
            design_timing_sum_line = line_num
        if line.find("Clock Summary") != -1:
            clock_summary_line = line_num
        if design_timing_sum_line != None and line_num == design_timing_sum_line + 6:
            wns = float(line.split()[0])
        if clock_summary_line != None and line_num == clock_summary_line + 6:
            clk_period = float(line.split()[3])

    # Parse Utilization Report
    with open(utilrpt, "r") as f:
        util_text = f.readlines()

    lut_logic = None
    lut_mem = None
    registers = None
    carry8 = None
    dsp = None
    bram = None

    for line in util_text:
        if lut_logic == None and line.find("LUT as Logic") != -1:
            lut_logic = int(line.split("|")[2].strip())

        if lut_mem == None and line.find("LUT as Memory") != -1:
            lut_mem = int(line.split("|")[2].strip())

        if registers == None and line.find("CLB Registers") != -1:
            registers = int(line.split("|")[2].strip())

        if carry8 == None and line.find("CARRY8") != -1:
            carry8 = int(line.split("|")[2].strip())

        if dsp == None and line.find("DSPs") != -1:
            dsp = int(line.split("|")[2].strip())

        if bram == None and line.find("Block RAM Tile") != -1:
            bram = int(line.split("|")[2].strip())

    return {
        "wns": wns,
        "clk_period": clk_period,
        "lut_logic": lut_logic,
        "lut_mem": lut_mem,
        "registers": registers,
        "carry8": carry8,
        "dsp": dsp,
        "bram": bram,
    }


def insert_data(data_dict: dict, new_val: dict):
    for k, v in new_val.items():
        if k in data_dict:
            data_dict[k].append(v)
        else:
            data_dict[k] = [v]


def gather_data(build_dir: Path):
    data = {}
    for module_path in build_dir.glob("*"):
        top = module_path.name
        for bitwidth_path in module_path.glob("*bit"):
            bitwidth = int(re.search(r"(\d+)bit", bitwidth_path.name).groups()[0])
            timing_util_data = timing_util(bitwidth_path)
            if timing_util_data == None:
                continue
            print(timing_util_data)
            insert_data(
                data,
                {
                    "norm": top,
                    "width": bitwidth,
                    "frac_width": bitwidth // 2,
                    **timing_util_data,
                },
            )
    return pd.DataFrame(data)


if __name__ == "__main__":
    data = gather_data(Path("build"))
    data["ns"] = data["clk_period"] - data["wns"]
    data["fmax"] = 1 / (data["ns"] * (10**-9))
    data["fmax_mhz"] = data["fmax"] / 1_000_000

    print(data)

    def plot(col):
        alt.Chart(data).mark_line().encode(
            x="width",
            y=col,
            color="norm",
        ).properties(
            width=400,
            height=200,
        ).save(f"{col}_plot.png", scale_factor=3)

    plot("wns")
    plot("lut_logic")
    plot("lut_mem")
    plot("dsp")
    plot("bram")
    plot("registers")
    plot("fmax_mhz")
