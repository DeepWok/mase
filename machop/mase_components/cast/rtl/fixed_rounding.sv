`timescale 1ns / 1ps

module fixed_rounding #(
    parameter IN_SIZE = 3,
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 3,
    parameter OUT_WIDTH = 3,
    parameter OUT_FRAC_WIDTH = 1
) (
    input  [ IN_WIDTH - 1:0] data_in [IN_SIZE - 1:0],
    output [OUT_WIDTH - 1:0] data_out[IN_SIZE - 1:0]
);
  for (genvar i = 0; i < IN_SIZE; i++) begin : parallel_round
    fixed_round #(
        .IN_WIDTH(IN_WIDTH),
        .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
    ) fr_inst (
        .data_in (data_in[i]),
        .data_out(data_out[i])
    );
  end

endmodule
