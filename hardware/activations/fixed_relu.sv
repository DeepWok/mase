module fixed_relu #(
    parameter IN_SIZE  = 8,
    parameter IN_WIDTH = 8
) (
    input  logic [IN_WIDTH-1:0] data_in [IN_SIZE-1:0],
    output logic [IN_WIDTH-1:0] data_out[IN_SIZE-1:0]
);

  for (genvar i = 0; i < IN_SIZE; i++) begin : ReLU
    always_comb begin
      // negative value, put to zero
      if ($signed(data_in[i]) <= 0) data_out[i] = '0;
      else data_out[i] = data_in[i];
    end
  end

endmodule
