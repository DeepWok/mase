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

  MYDATA buffer;
  logic  buffer_full;

  always_ff @(posedge clk) begin
    if (rst) buffer_full <= 1'b0;
    else begin
      // if r_ready is high, the read transaction is complete and the buffer is empty
      if (r_ready) buffer_full <= 1'b0;
      // if w_valid is high, the write transaction is complete and the buffer is full
      if (w_valid) buffer_full <= 1'b1;

      if (w_valid && w_ready) buffer <= w_data;
    end
  end



  always_comb begin
    // if the buffer is not full, we are ready to accept a write transaction
    w_ready = ~buffer_full;
    // if the buffer is full, we are ready to accept a read transaction
    r_valid = buffer_full;
    // dummy wiring
    r_data  = buffer;
  end

endmodule
