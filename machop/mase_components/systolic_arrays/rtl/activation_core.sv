`timescale 1ns / 1ps
module activation_core #(
    parameter PRECISION   = 0,
    parameter DATA_WIDTH  = 32,
    parameter FLOAT_WIDTH = 32
) (
    input logic core_clk,
    input logic resetn,

    input logic [7:0] sel_activation,

    input logic                  in_feature_valid,
    input logic [DATA_WIDTH-1:0] in_feature,

    output logic                  activated_feature_valid,
    output logic [DATA_WIDTH-1:0] activated_feature,

    input logic [DATA_WIDTH-1:0] layer_config_leaky_relu_alpha_value
);

  logic                  activated_feature_valid_comb;
  logic [DATA_WIDTH-1:0] activated_feature_comb;

  logic                  leaky_relu_activation_valid_comb;
  logic [DATA_WIDTH-1:0] leaky_relu_activation_comb;

  assign activated_feature_valid_comb = (sel_activation == 0) ? in_feature_valid
                               : (sel_activation == 1) ? in_feature_valid
                               : (sel_activation == 2) ? leaky_relu_activation_valid_comb
                               : '0;

  always_comb begin
    case (sel_activation)

      0: begin
        activated_feature_comb = in_feature;  // none
      end

      1: begin
        activated_feature_comb = in_feature[FLOAT_WIDTH-1] ? '0 : in_feature;  // relu
      end

      2: begin
        activated_feature_comb = in_feature[FLOAT_WIDTH-1] ? leaky_relu_activation_comb : in_feature; // leaky relu
      end

    endcase
  end

  // Leaky ReLU
  // -----------------------------------------------------------------

  if (PRECISION == 0) begin

`ifdef SIMULATION
    assign leaky_relu_activation_valid_comb = in_feature_valid;
    assign leaky_relu_activation_comb = in_feature;

`else
    float_multiplier activation_mult (
        .a_operand(in_feature),
        .b_operand(layer_config_leaky_relu_alpha_value),
        .result(leaky_relu_activation_comb),
        .Exception(),
        .Overflow(),
        .Underflow()
    );

    assign leaky_relu_activation_valid_comb = '1;
`endif

  end else begin

    // Fixed point
    always_comb begin
      leaky_relu_activation_valid_comb = in_feature_valid;
      leaky_relu_activation_comb = in_feature * layer_config_leaky_relu_alpha_value;
    end
  end

  // Register activated feature
  // -----------------------------------------------------------------

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      activated_feature_valid <= '0;
      activated_feature <= '0;

    end else begin
      activated_feature_valid <= activated_feature_valid_comb;
      activated_feature       <= activated_feature_comb;
    end

  end

endmodule
