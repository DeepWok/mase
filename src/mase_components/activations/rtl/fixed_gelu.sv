`timescale 1ns / 1ps


module fixed_gelu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,


    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + 2*DATA_IN_0_PRECISION_1 + APPROXIMATION_PRECISION-1,
    parameter DATA_OUT_0_PRECISION_1 = 2 * DATA_IN_0_PRECISION_1 + APPROXIMATION_PRECISION,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 8,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,

    parameter APPROXIMATION_PRECISION = 16,
    parameter APPROXIMATION_N = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0                                 [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0 -1 :0] data_out_0                              [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  parameter OUTPUT_PRECISION_0 = DATA_IN_0_PRECISION_0 + 2*DATA_IN_0_PRECISION_1 + APPROXIMATION_PRECISION-1;
  parameter OUTPUT_PRECISION_1 = 2 * DATA_IN_0_PRECISION_1 + APPROXIMATION_PRECISION;
  logic [DATA_OUT_0_PRECISION_0 -1 :0] output_data [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0];

  logic [DATA_IN_0_PRECISION_0-1:0]                         data_in_0_delayed0          [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 -1:0];
  logic [DATA_IN_0_PRECISION_0-1:0]                         data_in_0_delayed1          [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 -1:0];
  logic data_out_0_valid_delayed0;
  logic data_out_0_valid_delayed1;

  logic [APPROXIMATION_PRECISION-1 : 0]                         coefficient_a               [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION-1 : 0]                         coefficient_b               [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_0 - 1 :0]                         coefficient_c               [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_0 - 1 :0]   product_bx_scaled     [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_0 - 1 :0]   coefficient_c_scaled     [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_0 - 1 :0] product_ax2                  [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [2*DATA_IN_0_PRECISION_0 - 1 :0]                           product_x2                  [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + DATA_IN_0_PRECISION_0 - 1 :0]   product_bx                  [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];

  logic [APPROXIMATION_PRECISION*3 - 1:0]                       coefficients                                [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_IN_0_PRECISION_0 + APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_1 -1 :0]   sum                     [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [2:0] index[DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];


  genvar i;
  generate
    for (
        i = 0; i <= (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 - 1); i = i + 1
    ) begin


      const
      logic [APPROXIMATION_PRECISION*3 - 1:0]
      lut[0:APPROXIMATION_N-1] = {
        48'b111111111010111111111101100100011111101101001100,
        48'b111111011010010111110001101010001110100111100001,
        48'b111111110110010011110110100011111110110011001001,
        48'b000100110010100100011100101010011111111110101111,
        48'b000100110101101100100011001001011111111110111001,
        48'b111111111000000001001001000110111110110100001001,
        48'b111111011001110101001110011111011110100110110100,
        48'b111111111010110101000010011111111111101100110001
      };

      fixed_mult #(
          .IN_A_WIDTH(DATA_IN_0_PRECISION_0),
          .IN_B_WIDTH(DATA_IN_0_PRECISION_0)
      ) MX_multiplier_x2 (
          .data_a (data_in_0_delayed1[i]),
          .data_b (data_in_0_delayed1[i]),
          .product(product_x2[i])
      );

      fixed_mult #(
          .IN_A_WIDTH(2 * DATA_IN_0_PRECISION_0),
          .IN_B_WIDTH(APPROXIMATION_PRECISION)
      ) MX_multiplier_ax2 (
          .data_a (product_x2[i]),
          .data_b (coefficient_a[i]),
          .product(product_ax2[i])
      );

      fixed_mult #(
          .IN_A_WIDTH(DATA_IN_0_PRECISION_0),
          .IN_B_WIDTH(APPROXIMATION_PRECISION)
      ) MX_multiplier_bx (
          .data_a (data_in_0_delayed1[i]),
          .data_b (coefficient_b[i]),
          .product(product_bx[i])
      );

      always_ff @(posedge clk) begin
        coefficients[i] <= lut[index[i]];
      end


      always_ff @(posedge clk) begin
        if (data_in_0_valid) begin
          data_in_0_delayed0[i] <= data_in_0[i];
          data_in_0_delayed1[i] <= data_in_0_delayed0[i];
        end
      end

      always_comb begin
        if ($signed(data_in_0[i][DATA_IN_0_PRECISION_0-1 : DATA_IN_0_PRECISION_1]) >= 4)
          output_data[i] = ($signed(
              data_in_0_delayed1[i]
          )) <<< APPROXIMATION_PRECISION - 2 + DATA_IN_0_PRECISION_1;
        else if ($signed(data_in_0[i][DATA_IN_0_PRECISION_0-1 : DATA_IN_0_PRECISION_1]) <= -4)
          output_data[i] = 0;
        else output_data[i] = $signed(sum[i]);
      end

      always_comb sum[i] = product_ax2[i] + product_bx_scaled[i] + coefficient_c_scaled[i];

      always_comb
        index[i] = (data_in_0_delayed0[i][DATA_IN_0_PRECISION_1+2 : DATA_IN_0_PRECISION_1]) + 4;

      always_comb
        coefficient_c_scaled[i] = ($signed(coefficient_c[i])) <<< (2 * DATA_IN_0_PRECISION_1);

      always_comb product_bx_scaled[i] = ($signed(product_bx[i])) <<< (DATA_IN_0_PRECISION_1);

      assign coefficient_c[i] = $signed(coefficients[i][APPROXIMATION_PRECISION-1 : 0]);

      assign
        coefficient_b[i] = coefficients[i] [((APPROXIMATION_PRECISION)*2-1) : APPROXIMATION_PRECISION];

      assign
        coefficient_a[i] = coefficients[i] [((APPROXIMATION_PRECISION*3)-1) : APPROXIMATION_PRECISION*2];


    end
  endgenerate



  always_ff @(posedge clk) begin
    if (rst) data_out_0_valid_delayed1 <= 0;
    else if (data_out_0_ready && !data_in_0_valid) begin
      data_out_0_valid_delayed0 <= 0;
      data_out_0_valid_delayed1 <= data_out_0_valid_delayed0;
    end else if (data_in_0_valid) begin
      data_out_0_valid_delayed0 <= 1;
      data_out_0_valid_delayed1 <= data_out_0_valid_delayed0;
    end else data_out_0_valid_delayed1 <= data_out_0_valid_delayed1;
  end

  assign data_in_0_ready  = 1;

  assign data_out_0_valid = data_out_0_valid_delayed1;

for (genvar i = 0; i < DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1; i++) begin
  fixed_signed_cast #(
    .IN_WIDTH      (OUTPUT_PRECISION_0),
    .IN_FRAC_WIDTH (OUTPUT_PRECISION_1),
    .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
    .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1),
    .SYMMETRIC     (0),
    .ROUND_FLOOR   (1)
  ) output_cast (
    .in_data (output_data[i]),
    .out_data(data_out_0[i])
  );
end


endmodule
