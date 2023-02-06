module int_relu #(
    parameter NUM = 8,
    // parameter ACT_BIAS = 3,
    parameter ACT_WIDTH = 8
) (
    input  logic [ACT_WIDTH-1:0] x  [NUM-1:0],
    output logic [ACT_WIDTH-1:0] out[NUM-1:0]
);

  for (genvar i = 0; i < NUM; i++) begin : ReLU
    always_comb begin
      // negative value, put to zero
      if ($signed(x[i]) <= 0) out[i] = '0;
      else out[i] = x[i];
    end
  end

endmodule
