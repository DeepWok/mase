import os

DSE_MODES = ["codegen", "synth", "report", "all"]


def get_tcl_buff(project=None, top=None, cpp=None):
    assert project is not None
    assert top is not None
    assert cpp is not None
    return f"""
open_project -reset {project} 
set_top {top}
add_files {{ {cpp} }}
open_solution -reset "solution1"
set_part {{xcu250-figd2104-2L-e}}
create_clock -period 4 -name default
config_bind -effort high
config_compile -pipeline_loops 1
csynth_design
"""


class HLSResults:
    def __init__(self):
        """
        HLS Results
        """

        self.top = None
        self.project = None
        self.latency_min = -1
        self.latency_max = -1
        self.clock_period = -1
        self.bram = -1
        self.dsp = -1
        self.ff = -1
        self.lut = -1
        self.uram = -1


def get_hls_results(project=None, top=None):
    assert project is not None
    assert top is not None

    report = os.path.join(project, "solution1", "syn", "report", f"{top}_csynth.rpt")

    if not os.path.isfile(report):
        print(f"Cannot find synthesis report for {project}")
        return None

    timing_done = False
    area_done = False
    latency_start = False
    latency_done = False

    f = open(report)
    for line in f.readlines():
        """
        +--------+---------+----------+------------+
        |  Clock |  Target | Estimated| Uncertainty|
        +--------+---------+----------+------------+
        |ap_clk  |  4.00 ns|  2.503 ns|     1.08 ns|
        +--------+---------+----------+------------+
        """
        if "ap_clk" in line and not timing_done:
            line = line.split("|")
            clock_period = float(line[3].replace("ns", ""))
            timing_done = True

        """
        + Latency:
            * Summary:
            +---------+---------+----------+----------+-----+-----+---------+
            |  Latency (cycles) |  Latency (absolute) |  Interval | Pipeline|
            |   min   |   max   |    min   |    max   | min | max |   Type  |
            +---------+---------+----------+----------+-----+-----+---------+
            |      515|      515|  2.060 us|  2.060 us|  516|  516|       no|
            +---------+---------+----------+----------+-----+-----+---------+
        """
        if latency_start > 0 and not latency_done:
            latency_start += 1
        if "Latency:" in line:
            latency_start += 1
        if latency_start == 7 and not latency_done:
            line = line.split("|")
            latency_min = int(line[1])
            latency_max = int(line[2])
            latency_done = True

        """
        +---------------------+---------+-------+---------+---------+------+
        |         Name        | BRAM_18K|  DSP  |    FF   |   LUT   | URAM |
        +---------------------+---------+-------+---------+---------+------+
        |DSP                  |        -|      -|        -|        -|     -|
        |Expression           |        -|      -|        0|      209|     -|
        |FIFO                 |        -|      -|        -|        -|     -|
        |Instance             |        -|      -|        -|        -|     -|
        |Memory               |        0|      -|        4|        6|     0|
        |Multiplexer          |        -|      -|        -|      113|     -|
        |Register             |        -|      -|       45|        -|     -|
        +---------------------+---------+-------+---------+---------+------+
        |Total                |        0|      0|       49|      328|     0|
        +---------------------+---------+-------+---------+---------+------+
        """

        if "|Total" in line and not area_done:
            line = line.split("|")
            bram = int(line[2])
            dsp = int(line[3])
            ff = int(line[4])
            lut = int(line[5])
            uram = int(line[6])
            area_done = True

    f.close()

    hr = HLSResults()
    hr.top = top
    hr.project = project
    hr.latency_min = latency_min
    hr.latency_max = latency_max
    hr.clock_period = clock_period
    hr.bram = bram
    hr.dsp = dsp
    hr.ff = ff
    hr.lut = lut
    hr.uram = uram
    return hr


def bash_gen(commands, top, name):
    for i, thread in enumerate(commands):
        f = open(os.path.join(top, f"thread_{i}.sh"), "w")
        for command in thread:
            f.write(command + "\n")
        f.close()

    f = open(os.path.join(top, f"run.sh"), "w")
    f.write(f'echo "{name}" ')
    for i in range(0, len(commands)):
        f.write(f"& bash thread_{i}.sh ")
    f.close()


def csv_gen(data_points, top, name):
    f = open(os.path.join(top, f"{name}.csv"), "w")
    for line in data_points:
        for data in line:
            f.write(f"{data},")
        f.write("\n")
    f.close()
