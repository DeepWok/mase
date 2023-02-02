module adder_tree #(
    parameter NUM = -1,
    parameter IN_WIDTH = -1,
    parameter OUT_WIDTH = $clog2(NUM) + IN_WIDTH
) (
    input  logic                               clk,
    input  logic                               rst,
    input  logic [      NUM-1:0][IN_WIDTH-1:0] in,
    input  logic                               in_valid,
    output logic                               in_ready,
    output logic [OUT_WIDTH-1:0]               out,
    output logic                               out_valid,
    input  logic                               out_ready
);

  localparam LEVELS = $clog2(NUM);

  // Declare intermediate values
  for (genvar i = 0; i <= LEVELS; i++) begin : vars
    localparam LEVEL_NUM = (NUM + ((1 << i) - 1)) >> i;
    logic [LEVEL_NUM-1:0][(IN_WIDTH + i)-1:0] data;
    logic valid;
    logic ready;
  end

  // Generate adder for each layer
  for (genvar i = 0; i < LEVELS; i++) begin : level
    localparam LEVEL_NUM = (NUM + ((1 << i) - 1)) >> i;
    localparam NEXT_LEVEL_NUM = (LEVEL_NUM + 1) / 2;
    logic [NEXT_LEVEL_NUM-1:0][(IN_WIDTH + i):0] sum;

    adder_tree_layer #(
        .NUM(LEVEL_NUM),
        .IN_WIDTH(IN_WIDTH + i)
    ) layer (
        .in (vars[i].data),
        .out(sum)
    );

    register_slice #(
        .DATA_WIDTH($bits(sum)),
    ) register_slice (
        .clk    (clk),
        .rst    (rst),
        .w_valid(vars[i].valid),
        .w_ready(vars[i].ready),
        .w_data (sum),
        .r_valid(vars[i+1].valid),
        .r_ready(vars[i+1].ready),
        .r_data (vars[i+1].data)
    );
  end

  // it will zero-extend automatically
  assign vars[0].data = in;
  assign vars[0].valid = in_valid;
  assign in_ready = vars[0].ready;

  assign out = vars[LEVELS].data;
  assign out_valid = vars[LEVELS].valid;
  assign vars[LEVELS].ready = out_ready;

endmodule
