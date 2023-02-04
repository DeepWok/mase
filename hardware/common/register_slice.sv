module register_slice #(
    parameter DATA_WIDTH = 32,
    parameter type MYDATA = logic [DATA_WIDTH-1:0]
) (
    input logic clk,
    input logic rst,

    input  MYDATA w_data,
    input  logic  w_valid,
    output logic  w_ready,

    output MYDATA r_data,
    output logic  r_valid,
    input  logic  r_ready
);

  // The buffer stores the intermeidate data being computed in the register slice
  MYDATA buffer;
  // The shift register stores the validity of the data in the buffer 
  logic  shift_reg;

  // shift_register 
  always_ff @(posedge clk) begin
    if (rst) shift_reg <= 1'b0;
    else begin
      // no backpressure or buffer empty
      if (r_ready || !shift_reg) shift_reg <= w_valid;
      else shift_reg <= shift_reg;
    end
  end

  // buffer 
  always_ff @(posedge clk) begin
    // backpressure && valid output
    if (!r_ready && r_valid) buffer <= buffer;
    else buffer <= w_data;
  end

  always_comb begin
    // empty buffer or no back pressure
    w_ready = (~shift_reg) | r_ready;
    // dummy wiring 
    r_valid = shift_reg;
    r_data  = buffer;
  end

endmodule
