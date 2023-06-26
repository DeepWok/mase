# Max resources of supported FPGA boards

fpga_board_info = {
    "xcu250-figd2104-2L-e": {
        "LUT": 1728000,
        "LLUT": 1728000,
        "LUTRAM": 791040,
        "FF": 3456000,
        "RAM36": 2688,
        "RAMB18": 5376,
        "URAM": 1280,
        "DSP": 12288,
        "CLK": 4,  # 4ns, 250MHz
        "UTIL": 0.75,  # Empircally choose 75% logic utilization to reduce congestion
    }
}
