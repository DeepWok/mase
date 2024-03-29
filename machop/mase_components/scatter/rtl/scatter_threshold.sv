`timescale 1ns / 1ps

module scatter_threshold #(
    parameter PRECISION = 16,
    parameter TENSOR_SIZE_DIM = 8,
    parameter HIGH_SLOTS = 2,
    parameter THRESHOLD = 6,
    parameter DESIGN = 1
) (
    input clk,
    input rst,
    input logic [PRECISION-1:0] data_in[TENSOR_SIZE_DIM-1:0],
    output logic [PRECISION-1:0] o_high_precision[TENSOR_SIZE_DIM-1:0],
    output logic [PRECISION-1:0] o_low_precision[TENSOR_SIZE_DIM-1:0]
);

  //Logic to indentify values larger than x (2MSB check) //Can parameterise
  // logic [DATA_IN_0_PRECISION_0-2:0] mantisssa [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0];
  // logic [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] sign;

  logic [TENSOR_SIZE_DIM-1:0] high_precision_req_vec;
  for (genvar i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
    //Check if greater than threshold
    assign high_precision_req_vec[i] = ($signed(
        data_in[i]
    ) > +THRESHOLD | $signed(
        data_in[i]
    ) < -THRESHOLD);
  end


  //Logic to assign indicies based on priority

  //Pack array first
  wire [TENSOR_SIZE_DIM-1:0] output_mask;


  logic [$clog2(TENSOR_SIZE_DIM)-1:0] address_outliers[HIGH_SLOTS-1:0];

  generate
    if (DESIGN == 1) begin : PE_D1
      priority_encoder #(
          .NUM_INPUT_CHANNELS(TENSOR_SIZE_DIM),
          .NUM_OUPUT_CHANNELS(TENSOR_SIZE_DIM),
          .NO_INDICIES(HIGH_SLOTS)
      ) encoder1 (
          .input_channels(high_precision_req_vec),
          // .output_channels(address_outliers)
          .mask(output_mask)
      );

    end else if (DESIGN == 2) begin : PE_D2
      priority_encoder #(
          .NUM_INPUT_CHANNELS(TENSOR_SIZE_DIM),
          .NUM_OUPUT_CHANNELS(TENSOR_SIZE_DIM),
          .NO_INDICIES(HIGH_SLOTS)
      ) encoder1 (
          .input_channels(high_precision_req_vec),
          // .output_channels(address_outliers)
          .mask(output_mask)
      );

    end

  endgenerate



  //Logic to apply mask
  array_zero_mask #(
      .NUM_INPUTS(TENSOR_SIZE_DIM),
      .PRECISION (PRECISION)
  ) masker (
      .data      (data_in),           // Unpacked array of 4 8-bit vectors
      .mask      (output_mask),       // 4-bit mask
      .data_out_0(o_high_precision),
      .data_out_1(o_low_precision)
  );



endmodule
