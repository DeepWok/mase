module silu_lut (
    input  logic [1:0] data_in_0,
    output logic [1:0] data_out_0
);
  always_comb begin
    case (data_in_0)
      2'b00:   data_out_0 = 2'b00;
      2'b01:   data_out_0 = 2'b01;
      2'b10:   data_out_0 = 2'b11;
      2'b11:   data_out_0 = 2'b00;
      default: data_out_0 = 2'b0;
    endcase
  end
endmodule
