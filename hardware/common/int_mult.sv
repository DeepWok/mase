// fixed-point multiplier

module int_mult #(
    parameter      DATA_A_WIDTH = 32,
    parameter      DATA_B_WIDTH = 32,
    parameter type MYDATA_A     = logic [             DATA_A_WIDTH-1:0],
    parameter type MYDATA_B     = logic [             DATA_B_WIDTH-1:0],
    parameter type MYPRODUCT    = logic [DATA_A_WIDTH+DATA_B_WIDTH-1:0]
) (
    input  MYDATA_A  data_a,
    input  MYDATA_B  data_b,
    output MYPRODUCT product
);

  assign product = $signed(data_a) * $signed(data_b);

endmodule
