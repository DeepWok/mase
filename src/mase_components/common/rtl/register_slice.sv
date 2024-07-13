`timescale 1ns / 1ps
module register_slice #(
    parameter DATA_WIDTH = 32,
    parameter type MYDATA = logic [DATA_WIDTH-1:0]
) (
    input logic clk,
    input logic rst,

    input  MYDATA data_in,
    input  logic  data_in_valid,
    output logic  data_in_ready,

    output MYDATA data_out,
    output logic  data_out_valid,
    input  logic  data_out_ready
);

  // The buffer stores the intermeidate data being computed in the register slice
  logic [DATA_WIDTH-1:0] buffer;
  // The shift register stores the validity of the data in the buffer
  logic shift_reg;


  // There are eight cases:
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // | hold_any_input | in_valid | out_ready | accept_next_in | release_current_in | hold_any_input_next |
  // |   (shift_reg)  |          |           |                |                    |   (shift_reg_next)  |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // |              0 |        0 |         0 |              0 |                  0 |                   0 |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // |              0 |        0 |         1 |              0 |                  0 |                   0 |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // |              0 |        1 |         0 |              1 |                  0 |                   1 |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // |              0 |        1 |         1 |              1 |                  0 |                   1 |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // |              1 |        0 |         0 |              0 |                  0 |                   1 |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // |              1 |        0 |         1 |              0 |                  1 |                   0 |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // |              1 |        1 |         0 |              0 |                  0 |                   1 |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+
  // |              1 |        1 |         1 |              1 |                  1 |                   1 |
  // +----------------+----------+-----------+----------------+--------------------+---------------------+

  // shift_register
  always_ff @(posedge clk) begin
    if (rst) shift_reg <= 1'b0;
    else begin
      shift_reg <= (shift_reg && (!data_out_ready)) || data_in_valid;
    end
  end

  // buffer
  assign to_load = ((!shift_reg) && data_in_valid) || (data_in_valid && data_out_ready);
  always_ff @(posedge clk) begin
    if (rst) buffer <= 0;
    else if (to_load) buffer <= data_in;
  end

  // output 
  assign to_output = shift_reg && data_out_ready;
  always_ff @(posedge clk) begin
    if (rst) data_out <= 0;
    else if (to_load) data_out <= buffer;
  end


  // control logic
  // +----------------+----------+-----------+----------+-----------+
  // | hold_any_input | in_valid | out_ready | in_ready | out_valid |
  // |   (shift_reg)  |          |           |          |           |
  // +----------------+----------+-----------+----------+-----------+
  // |              0 |        0 |         0 |        1 |         0 |
  // +----------------+----------+-----------+----------+-----------+
  // |              0 |        0 |         1 |        1 |         0 |
  // +----------------+----------+-----------+----------+-----------+
  // |              0 |        1 |         0 |        1 |         0 |
  // +----------------+----------+-----------+----------+-----------+
  // |              0 |        1 |         1 |        1 |         0 |
  // +----------------+----------+-----------+----------+-----------+
  // |              1 |        0 |         0 |        1 |         1 |
  // +----------------+----------+-----------+----------+-----------+
  // |              1 |        0 |         1 |        1 |         1 |
  // +----------------+----------+-----------+----------+-----------+
  // |              1 |        1 |         0 |        0 |         1 |
  // +----------------+----------+-----------+----------+-----------+
  // |              1 |        1 |         1 |        1 |         1 |
  // +----------------+----------+-----------+----------+-----------+

  assign data_in_ready  = (!shift_reg) || (!data_in_valid) || data_out_ready;
  assign data_out_valid = shift_reg;

endmodule

