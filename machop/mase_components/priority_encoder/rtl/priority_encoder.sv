`timescale 1ns / 1ps
module priority_encoder #(
    parameter NUM_INPUT_CHANNELS = 2,
    //parameter NUM_OUPUT_CHANNELS = 1,
    parameter NO_INDICIES = 1

) (
    input [NUM_INPUT_CHANNELS-1:0] input_channels,
    // output logic [$clog2(NUM_INPUT_CHANNELS)-1:0] output_channels [NO_INDICIES-1:0],
    output logic [NUM_INPUT_CHANNELS-1:0] mask
);

  //Can use multiplexer design and check which is better after synthesis
  // logic set;
  // logic [NUM_INPUT_CHANNELS-1:0] idx;
  logic [NUM_INPUT_CHANNELS-1:0] input_channels_temp;
  logic [NUM_INPUT_CHANNELS-1:0] channel_mask;

  always_comb begin
    input_channels_temp = input_channels;
    mask = {NUM_INPUT_CHANNELS{1'b0}};

    for (genvar j = 0; j < NO_INDICIES; j = j + 1) begin : PRIORITY_ENCODER
      channel_mask = input_channels_temp & (~(input_channels_temp - 1));
      input_channels_temp = input_channels_temp & ~channel_mask;
      mask = mask | channel_mask;
    end
  end
  // // end






endmodule


// module index_to_mask #(
//     parameter NUM_INPUT_CHANNELS = 4,
//     parameter NUM_OUPUT_CHANNELS = 1,
//     parameter NO_INDICIES = 1,
//     parameter OUTPUT_WIDTH = 4

// )(
//     input [$clog2(NUM_INPUT_CHANNELS)-1:0] indicies [NO_INDICIES-1:0],
//     output logic [OUTPUT_WIDTH-1:0] output_mask
// );
//     integer i;
//     always @* begin
//         output_mask = 0;
//         for (i=0; i< NO_INDICIES; i=i+1)
//             output_mask[indicies[i]] = 1'b1;
//     end


// endmodule



// module array_zero_mask#(
//     parameter NUM_INPUTS =4,
//     parameter PRECISION = 4
//     )(
//     input signed [PRECISION-1:0] data[NUM_INPUTS-1:0],   // Unpacked array of 4 8-bit vectors
//     input [NUM_INPUTS-1:0] mask,        // 4-bit mask
//     output logic signed [PRECISION-1:0] data_out_0[NUM_INPUTS-1:0],  // Modified array
//     output logic signed [PRECISION-1:0] data_out_1[NUM_INPUTS-1:0]  // Modified array

// );

// // Always block that updates the output based on the mask
// always @(*) begin
//     integer i;
//     for (i = 0; i < NUM_INPUTS; i = i + 1) begin
//         // Check each bit of the mask; if it's 0, copy the original data, else set to zero
//         if (mask[i] == 1'b1) begin
//             data_out_0[i] = data[i];
//             data_out_1[i] = 0; 

//         end else if (mask[i] == 1'b0) begin
//             data_out_0[i] = 0; 
//             data_out_1[i] = data[i];

//         end
//     end
// end

// endmodule


// module parallel_abs_quantize#(
//     NO_INPUTS = 2,
//     INPUT_PRECISION = 16,
//     OUTPUT_PRECISION = 1
//     )(
//     input [INPUT_PRECISION-1:0] input_array [NO_INPUTS-1:0],
//     output logic [OUTPUT_PRECISION-1:0] output_array [NO_INPUTS-1:0]
// );

//     logic sign  [INPUT_PRECISION-1:0];
//     logic [OUTPUT_PRECISION-1:0] msbs [INPUT_PRECISION-1:0];

//     for (genvar i = 0; i < NO_INPUTS; i = i + 1) begin
//         //Extract sign bit
//         assign sign[i] = input_array[i][INPUT_PRECISION-1];
//         assign msbs[i] = input_array[i][INPUT_PRECISION-2:INPUT_PRECISION-2-OUTPUT_PRECISION];

//         assign output_array[i] = (sign[i])? !msbs[i]:msbs[i];
//     end

// endmodule
