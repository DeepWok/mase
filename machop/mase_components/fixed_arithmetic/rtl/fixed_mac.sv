//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:
// Design Name: 
// Module Name: fixed_mac
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

module fixed_mac #(
    parameter DATA_WIDTH = 16
) (
    input logic core_clk,
    input logic resetn,

    input  logic in_valid,
    output logic in_ready,

    input logic [DATA_WIDTH-1:0] a,
    input logic [DATA_WIDTH-1:0] b,

    output logic [DATA_WIDTH-1:0] accumulator,

    input logic                  overwrite,
    input logic [DATA_WIDTH-1:0] overwrite_data
);

  logic [DATA_WIDTH-1:0] acc_reg;

  // Accumulator
  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      acc_reg <= '0;
    end else begin
      acc_reg <= overwrite ? overwrite_data : in_valid && in_ready ? (acc_reg + a * b) : acc_reg;
    end
  end

  assign accumulator = acc_reg;

  assign in_ready = '1;


endmodule
