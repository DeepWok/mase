module adder_tree_layer #(
    parameter NUM = 2,
    parameter IN_WIDTH = 32
) (
    input logic [IN_WIDTH-1:0] ins[NUM-1:0],
    output logic [IN_WIDTH:0] outs[(NUM+1)/2-1:0]
);

  generate
    for (genvar i = 0; i < NUM / 2; i++) begin : pair
      assign outs[i] = {ins[i][IN_WIDTH-1], ins[i]} + {ins[NUM-1-i][IN_WIDTH-1], ins[NUM-1-i]};
    end

    if (NUM % 2 != 0) begin : left
      assign outs[NUM/2] = {ins[NUM/2][IN_WIDTH-1], ins[NUM/2]};
    end
  endgenerate

endmodule
