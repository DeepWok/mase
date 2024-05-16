`timescale 1ns / 1ps

// TODO: Add signed param. fixed_adder_tree_layer already supports signedness

module fixed_adder_tree #(
    parameter IN_SIZE   = 2,
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = $clog2(IN_SIZE) + IN_WIDTH
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                 clk,
    input  logic                 rst,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [ IN_WIDTH-1:0] data_in       [IN_SIZE-1:0],
    input  logic                 data_in_valid,
    output logic                 data_in_ready,
    output logic [OUT_WIDTH-1:0] data_out,
    output logic                 data_out_valid,
    input  logic                 data_out_ready
);

  localparam LEVELS = $clog2(IN_SIZE);

  initial begin
    assert (IN_SIZE > 0);
  end

  generate
    if (LEVELS == 0) begin : gen_skip_adder_tree

      assign data_out = data_in[0];
      assign data_out_valid = data_in_valid;
      assign data_in_ready = data_out_ready;

    end else begin : gen_adder_tree

      // data & sum wires are oversized on purpose for vivado.
      logic [OUT_WIDTH*IN_SIZE-1:0] data[LEVELS:0];
      logic [OUT_WIDTH*IN_SIZE-1:0] sum[LEVELS-1:0];
      logic valid[IN_SIZE-1:0];
      logic ready[IN_SIZE-1:0];

      // Generate adder for each layer
      for (genvar i = 0; i < LEVELS; i++) begin : level

        localparam LEVEL_IN_SIZE = (IN_SIZE + ((1 << i) - 1)) >> i;
        localparam LEVEL_OUT_SIZE = (LEVEL_IN_SIZE + 1) / 2;
        localparam LEVEL_IN_WIDTH = IN_WIDTH + i;
        localparam LEVEL_OUT_WIDTH = LEVEL_IN_WIDTH + 1;

        fixed_adder_tree_layer #(
            .IN_SIZE (LEVEL_IN_SIZE),
            .IN_WIDTH(LEVEL_IN_WIDTH)
        ) layer (
            .data_in (data[i]),  // flattened LEVEL_IN_SIZE * LEVEL_IN_WIDTH
            .data_out(sum[i])    // flattened LEVEL_OUT_SIZE * LEVEL_OUT_WIDTH
        );

        skid_buffer #(
            .DATA_WIDTH(LEVEL_OUT_SIZE * LEVEL_OUT_WIDTH)
        ) register_slice (
            .clk           (clk),
            .rst           (rst),
            .data_in       (sum[i]),
            .data_in_valid (valid[i]),
            .data_in_ready (ready[i]),
            .data_out      (data[i+1]),
            .data_out_valid(valid[i+1]),
            .data_out_ready(ready[i+1])
        );

      end

      for (genvar i = 0; i < IN_SIZE; i++) begin : gen_input_assign
        assign data[0][(i+1)*IN_WIDTH-1 : i*IN_WIDTH] = data_in[i];
      end

      assign valid[0] = data_in_valid;
      assign data_in_ready = ready[0];

      assign data_out = data[LEVELS][OUT_WIDTH-1:0];
      assign data_out_valid = valid[LEVELS];
      assign ready[LEVELS] = data_out_ready;

    end
  endgenerate


endmodule
