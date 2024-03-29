`timescale 1ns / 1ps


module fixed_softsign #(
    /* verilator lint_off UNUSEDPARAM */

    localparam APPROXIMATION_N = 64,
    localparam APPROXIMATION_PRECISION = 12,

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,


    parameter DATA_OUT_0_PRECISION_0 = APPROXIMATION_PRECISION + 2 * DATA_IN_0_PRECISION_1,
    parameter DATA_OUT_0_PRECISION_1 = 2 * DATA_IN_0_PRECISION_1 + APPROXIMATION_PRECISION - 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 8,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1


) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0                                 [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_1 -1 :0] data_out_0                              [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready


);

  logic [APPROXIMATION_PRECISION-1 : 0]                         coefficient_a               [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION-1 : 0]                         coefficient_b               [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION-1 : 0]                         coefficient_c               [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_0 - 1 :0]   product_bx_scaled     [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_0 - 1 :0]   coefficient_c_scaled     [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_0 - 1 :0] product_ax2                  [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [2*DATA_IN_0_PRECISION_0 - 1 :0]                           product_x2                  [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + DATA_IN_0_PRECISION_0 - 1 :0]   product_bx                  [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_IN_0_PRECISION_0-1:0]                         data_in_0_delayed0          [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 -1:0];
  logic [DATA_IN_0_PRECISION_0-1:0]                         data_in_0_delayed1          [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 -1:0];
  logic data_out_0_valid_delayed0;
  logic data_out_0_valid_delayed1;
  logic sign_bit[DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 -1:0];
  logic sign_bit_delayed[DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 -1:0];
  logic   [DATA_IN_0_PRECISION_0-1:0]                       data_in_0_absolute          [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 -1:0];
  logic   [DATA_IN_0_PRECISION_0-1:0]                       data_in_0_absolute_delayed  [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 -1:0];

  logic [APPROXIMATION_PRECISION*3 - 1:0] coefficients                                [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [APPROXIMATION_PRECISION + 2*DATA_IN_0_PRECISION_0 - 1 :0]   sum                     [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [4:0] index[DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];


  genvar i;
  generate
    for (
        i = 0; i <= (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 - 1); i = i + 1
    ) begin

      const
      logic [APPROXIMATION_PRECISION*3 - 1:0]
      lut[0:APPROXIMATION_N/2-1] = {
        36'b101110110101011101100101000000000110,
        36'b111001111000010011110010000010010101,
        36'b111101001001001101100001000101010111,
        36'b111110011100001001101101001000001010,
        36'b111111000100000111010010001010100011,
        36'b111111011001000101101001001100100100,
        36'b111111100101000100100000001110010001,
        36'b111111101101000011101011001111101110,
        36'b111111110010000011000011010000111101,
        36'b111111110101000010100101010010000001,
        36'b111111111000000010001101010010111101,
        36'b111111111001000001111010010011110001,
        36'b111111111011000001101010010100011111,
        36'b111111111100000001011110010101001000,
        36'b111111111100000001010011010101101101,
        36'b111111111101000001001010010110001110,
        36'b111111111101000001000011010110101101,
        36'b111111111110000000111100010111001000,
        36'b111111111110000000110111010111100001,
        36'b111111111110000000110010010111111000,
        36'b111111111111000000101110011000001101,
        36'b111111111111000000101010011000100000,
        36'b111111111111000000100111011000110010,
        36'b111111111111000000100100011001000011,
        36'b111111111111000000100001011001010010,
        36'b111111111111000000011111011001100001,
        36'b111111111111000000011101011001101110,
        36'b111111111111000000011011011001111011,
        36'b111111111111000000011001011010000111,
        36'b111111111111000000011000011010010010,
        36'b000000000000000000010110011010011101,
        36'b000000000000000000010101011010100111
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

      //Use 5 LSB to read parameters from look up table
      always_ff @(posedge clk) begin
        coefficients[i] <= lut[index[i]];
      end


      always_ff @(posedge clk) begin
        if (data_in_0_valid) begin
          data_in_0_delayed0[i] <= data_in_0[i];
        end
      end

      always_ff @(posedge clk) begin
        if (data_in_0_valid) begin
          data_in_0_delayed1[i] <= data_in_0_absolute[i];
        end
      end

      always_ff @(posedge clk) begin
        if (data_in_0_valid) begin
          sign_bit[i] <= data_in_0[i][DATA_IN_0_PRECISION_0-1];
          sign_bit_delayed[i] <= sign_bit[i];
        end
      end

      always_ff @(posedge clk) begin
        data_in_0_absolute_delayed[i] <= data_in_0_absolute[i];
      end


      always_comb begin
        if ($signed(data_in_0[i][DATA_IN_0_PRECISION_0-1 : DATA_IN_0_PRECISION_1]) >= 16)
          data_out_0[i] = {
            1'b0, {(APPROXIMATION_PRECISION + 2 * DATA_IN_0_PRECISION_1 - 1) {1'b1}}
          };
        else if ($signed(data_in_0[i][DATA_IN_0_PRECISION_0-1 : DATA_IN_0_PRECISION_1]) <= -16)
          data_out_0[i] = {
            1'b1, {(APPROXIMATION_PRECISION + 2 * DATA_IN_0_PRECISION_1 - 2) {1'b0}}, 1'b1
          };
        else begin
          data_out_0[i] = sum[i];
          if (sign_bit_delayed[i] == 1) data_out_0[i] = (~sum[i] + 1);
        end

      end

      always_comb sum[i] = product_ax2[i] + product_bx_scaled[i] + coefficient_c_scaled[i];

      always_comb
        index[i] = (data_in_0_absolute[i][DATA_IN_0_PRECISION_1+3 : DATA_IN_0_PRECISION_1-1]);

      always_comb coefficient_c_scaled[i] = coefficient_c[i] <<< (2 * DATA_IN_0_PRECISION_1);

      always_comb product_bx_scaled[i] = product_bx[i] <<< (DATA_IN_0_PRECISION_1);

      always_comb
        if ($signed(data_in_0[i]) < 0)
          data_in_0_absolute[i] = (~data_in_0_delayed0[i] + 1);  // Negate if negative
        else data_in_0_absolute[i] = data_in_0_delayed0[i];  // Keep as is if non-negative

      assign coefficient_c[i] = coefficients[i][APPROXIMATION_PRECISION-1 : 0];

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


endmodule
