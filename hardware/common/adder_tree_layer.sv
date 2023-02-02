module adder_tree_layer #(
    parameter NUM = -1,
    parameter IN_WIDTH = -1
) (
    input logic [NUM-1:0][IN_WIDTH-1:0] in,
    output logic [(NUM+1)/2-1:0][IN_WIDTH:0] out
);

  generate
    for (genvar i = 0; i < NUM / 2; i++) begin : pair
      assign out[i] = {in[i][IN_WIDTH-1], in[i]} + {in[NUM-1-i][IN_WIDTH-1], in[NUM-1-i]};
    end

    if (NUM % 2 != 0) begin : left
      assign out[NUM/2] = {in[NUM/2][IN_WIDTH-1], in[NUM/2]};
    end
  endgenerate

endmodule
