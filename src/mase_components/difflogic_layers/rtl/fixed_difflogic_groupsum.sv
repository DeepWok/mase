module fixed_difflogic_groupsum #(
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0  = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 2
) (
    input logic clk,
    input logic rst,

    input logic [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] data_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [$clog2(
(DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_OUT_0_TENSOR_SIZE_DIM_0)
):0] data_out_0[0:DATA_OUT_0_TENSOR_SIZE_DIM_0-1],
    output logic data_out_0_valid,
    input logic data_out_0_ready

);

  localparam GROUP_SIZE = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_OUT_0_TENSOR_SIZE_DIM_0;

  genvar i;
  generate
    for (i = 0; i < DATA_OUT_0_TENSOR_SIZE_DIM_0; i = i + 1) begin : GROUP_SUM
      logic [GROUP_SIZE-1:0] group_data;
      logic [$clog2(GROUP_SIZE):0] sum;

      assign group_data = data_in_0[((i*GROUP_SIZE)+(GROUP_SIZE-1)):(i*GROUP_SIZE)];

      always_comb begin
        data_out_0[i] = $countones(group_data);
      end

      // always_comb begin
      //     sum = 0;
      //     for (int j = 0; j < GROUP_SIZE; j = j + 1) begin
      //         sum = sum + {{($clog2(GROUP_SIZE)-1){0'b0}}, group_data[j]};
      //     end
      //     data_out_0[i] = sum;
      // end

      // fixed_adder_tree_comb FTA (
      //     .clk(clk),
      //     .rst(rst),
      //     .data_in_0(group_data),
      //     .data_out_0(data_out_0[i])
      // );

    end
  endgenerate

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule


// module fixed_adder_tree #(
//     parameter IN_SIZE   = 2,
//     parameter DATA_IN_0_TENSOR_SIZE_DIM_0  = 32,
//     parameter OUT_WIDTH = $clog2(IN_SIZE) + DATA_IN_0_TENSOR_SIZE_DIM_0
// ) (
//     /* verilator lint_off UNUSEDSIGNAL */
//     input  logic                 clk,
//     input  logic                 rst,
//     /* verilator lint_on UNUSEDSIGNAL */
//     input  logic [ DATA_IN_0_TENSOR_SIZE_DIM_0-1:0] data_in_0       [IN_SIZE-1:0],
//     input  logic                 data_in_0_valid,
//     output logic                 data_in_0_ready,
//     output logic [OUT_WIDTH-1:0] data_out_0,
//     output logic                 data_out_0_valid,
//     input  logic                 data_out_0_ready
// );

//   localparam LEVELS = $clog2(IN_SIZE);

//   initial begin
//     assert (IN_SIZE > 0);
//   end

//   generate
//     if (LEVELS == 0) begin : gen_skip_adder_tree

//       assign data_out_0 = data_in_0[0];
//       assign data_out_0_valid = data_in_0_valid;
//       assign data_in_0_ready = data_out_0_ready;

//     end else begin : gen_adder_tree

//       // data & sum wires are oversized on purpose for vivado.
//       logic [OUT_WIDTH*IN_SIZE-1:0] data[LEVELS:0];
//       logic [OUT_WIDTH*IN_SIZE-1:0] sum[LEVELS-1:0];
//       logic valid[IN_SIZE-1:0];
//       logic ready[IN_SIZE-1:0];

//       // Generate adder for each layer
//       for (genvar i = 0; i < LEVELS; i++) begin : level

//         localparam LEVEL_IN_SIZE = (IN_SIZE + ((1 << i) - 1)) >> i;
//         localparam LEVEL_OUT_SIZE = (LEVEL_IN_SIZE + 1) / 2;
//         localparam LEVEL_DATA_IN_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0 + i;
//         localparam LEVEL_OUT_WIDTH = LEVEL_DATA_IN_0_TENSOR_SIZE_DIM_0 + 1;

//         fixed_adder_tree_layer #(
//             .IN_SIZE (LEVEL_IN_SIZE),
//             .DATA_IN_0_TENSOR_SIZE_DIM_0(LEVEL_DATA_IN_0_TENSOR_SIZE_DIM_0)
//         ) layer (
//             .data_in_0 (data[i]),  // flattened LEVEL_IN_SIZE * LEVEL_DATA_IN_0_TENSOR_SIZE_DIM_0
//             .data_out_0(sum[i])    // flattened LEVEL_OUT_SIZE * LEVEL_OUT_WIDTH
//         );

//         skid_buffer #(
//             .DATA_WIDTH(LEVEL_OUT_SIZE * LEVEL_OUT_WIDTH)
//         ) register_slice (
//             .clk           (clk),
//             .rst           (rst),
//             .data_in_0       (sum[i]),
//             .data_in_0_valid (valid[i]),
//             .data_in_0_ready (ready[i]),
//             .data_out_0      (data[i+1]),
//             .data_out_0_valid(valid[i+1]),
//             .data_out_0_ready(ready[i+1])
//         );

//         assign valid[i+1] = valid[i];
//         assign ready[i+1] = ready[i];

//       end

//       for (genvar i = 0; i < IN_SIZE; i++) begin : gen_input_assign
//         assign data[0][(i+1)*DATA_IN_0_TENSOR_SIZE_DIM_0-1 : i*DATA_IN_0_TENSOR_SIZE_DIM_0] = data_in_0[i];
//       end

//       assign valid[0] = data_in_0_valid;
//       assign data_in_0_ready = ready[0];

//       assign data_out_0 = data[LEVELS][OUT_WIDTH-1:0];
//       assign data_out_0_valid = valid[LEVELS];
//       assign ready[LEVELS] = data_out_0_ready;

//     end
//   endgenerate


// endmodule
