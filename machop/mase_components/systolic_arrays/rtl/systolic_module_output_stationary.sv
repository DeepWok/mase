//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:
// Design Name: 
// Module Name: systolic_module_output_stationary
// Project Name:
// Target Devices:
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module systolic_module_output_stationary #(
    parameter PRECISION = top_pkg::FLOAT_32,
    parameter FLOAT_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter MATRIX_N = 4
) (
    input logic core_clk,
    input logic resetn,

    input logic pulse_systolic_module,

    input logic [MATRIX_N-1:0]                 sys_module_forward_in_valid,
    input logic [MATRIX_N-1:0][DATA_WIDTH-1:0] sys_module_forward_in,

    input logic [MATRIX_N-1:0]                 sys_module_down_in_valid,
    input logic [MATRIX_N-1:0][DATA_WIDTH-1:0] sys_module_down_in,

    output logic [MATRIX_N-1:0]                 sys_module_forward_out_valid,
    output logic [MATRIX_N-1:0][DATA_WIDTH-1:0] sys_module_forward_out,

    output logic [MATRIX_N-1:0]                 sys_module_down_out_valid,
    output logic [MATRIX_N-1:0][DATA_WIDTH-1:0] sys_module_down_out,

    input logic                  bias_valid,
    input logic [DATA_WIDTH-1:0] bias,

    input logic                                             activation_valid,
    input logic [$bits(top_pkg::ACTIVATION_FUNCTION_e)-1:0] activation,

    input logic shift_valid,

    // Accumulators for each Processing Element, from which output matrix can be constructed
    // One more row than required to shift in zeros into last row during SHIFT phase
    output logic [MATRIX_N:0][MATRIX_N-1:0][DATA_WIDTH-1:0] sys_module_pe_acc,

    output logic diagonal_flush_done,

    input logic [DATA_WIDTH-1:0] layer_config_leaky_relu_alpha_value,

    output logic [MATRIX_N-1:0][MATRIX_N-1:0][DATA_WIDTH-1:0] debug_update_counter
);

  // ============================================================================================
  // Declarations
  // ============================================================================================

  //   <    row    > <    col   > <      data      >
  logic [MATRIX_N-1:0][  MATRIX_N:0][           0:0] sys_module_pe_forward_valid;
  logic [MATRIX_N-1:0][  MATRIX_N:0][DATA_WIDTH-1:0] sys_module_pe_forward;

  //   <    row    > <    col   > <      data      >
  logic [  MATRIX_N:0][MATRIX_N-1:0][           0:0] sys_module_pe_down_valid;
  logic [  MATRIX_N:0][MATRIX_N-1:0][DATA_WIDTH-1:0] sys_module_pe_down;

  logic [MATRIX_N-1:0]                               forward_flush_done;
  logic [MATRIX_N-1:0]                               down_flush_done;

  // ============================================================================================
  // Instances
  // ============================================================================================

  for (genvar row = 0; row < MATRIX_N; row++) begin : rows_gen
    for (genvar col = 0; col < MATRIX_N; col++) begin : cols_gen

      processing_element #(
          .PRECISION  (PRECISION),
          .DATA_WIDTH (DATA_WIDTH),
          .FLOAT_WIDTH(FLOAT_WIDTH)
      ) pe_i (
          .core_clk,
          .resetn,

          .pulse_systolic_module(pulse_systolic_module),

          .pe_forward_in_valid(sys_module_pe_forward_valid[row][col]),
          .pe_forward_in      (sys_module_pe_forward[row][col]),

          .pe_down_in_valid(sys_module_pe_down_valid[row][col]),
          .pe_down_in      (sys_module_pe_down[row][col]),

          .pe_forward_out_valid(sys_module_pe_forward_valid[row][col+1]),
          .pe_forward_out      (sys_module_pe_forward[row][col+1]),

          .pe_down_out_valid(sys_module_pe_down_valid[row+1][col]),
          .pe_down_out      (sys_module_pe_down[row+1][col]),

          .bias_valid(bias_valid),
          .bias      (bias),

          .activation_valid(activation_valid),
          .activation      (activation),

          .shift_valid(shift_valid),
          .shift_data (sys_module_pe_acc[row+1][col]),

          .pe_acc(sys_module_pe_acc[row][col]),

          .layer_config_leaky_relu_alpha_value(layer_config_leaky_relu_alpha_value),

          .debug_update_counter(debug_update_counter[row][col])
      );

    end : cols_gen
  end : rows_gen

  // Input to lowest row during SHIFT phase
  assign sys_module_pe_acc[MATRIX_N] = '0;

  // ============================================================================================
  // Logic
  // ============================================================================================

  for (genvar row = 0; row < MATRIX_N; row++) begin
    always_comb begin
      // Drive forward inputs
      sys_module_pe_forward[row][0]       = sys_module_forward_in[row];
      sys_module_pe_forward_valid[row][0] = sys_module_forward_in_valid[row];

      // Drive forward outputs
      sys_module_forward_out_valid[row]   = sys_module_pe_forward_valid[row][MATRIX_N];
      sys_module_forward_out[row]         = sys_module_pe_forward[row][MATRIX_N];
    end
  end

  assign forward_flush_done = ~sys_module_forward_out_valid;

  for (genvar col = 0; col < MATRIX_N; col++) begin

    always_comb begin
      // Drive down inputs
      sys_module_pe_down[0][col]       = sys_module_down_in[col];
      sys_module_pe_down_valid[0][col] = sys_module_down_in_valid[col];

      // Drive down outputs
      sys_module_down_out_valid[col]   = sys_module_pe_down_valid[MATRIX_N][col];
      sys_module_down_out[col]         = sys_module_pe_down[MATRIX_N][col];
    end
  end

  assign down_flush_done = ~sys_module_down_out_valid;

  assign diagonal_flush_done = &forward_flush_done && &down_flush_done;

  // ============================================================================================
  // Assertions
  // ============================================================================================

  // for (genvar row=0; row < MATRIX_N; row++) begin

  //     P_forward_valid_propagates: assert property (
  //         @(posedge core_clk) disable iff (!resetn)
  //         sys_module_forward_in_valid[row] |-> ##(MATRIX_N) sys_module_pe_forward_valid[row][MATRIX_N]
  //     );

  // end

  // for (genvar col=0; col < MATRIX_N; col++) begin

  //     P_down_valid_propagates: assert property (
  //         @(posedge core_clk) disable iff (!resetn)
  //         sys_module_down_in_valid[col] |-> ##(MATRIX_N) sys_module_pe_down_valid[MATRIX_N][col]
  //     );

  // end


endmodule
