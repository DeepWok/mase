module adder_tree #(
    parameter NUM = 1,
    parameter IN_WIDTH = 32,
    parameter OUT_WIDTH = $clog2(NUM) + IN_WIDTH
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                 clk,
    input  logic                 rst,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [ IN_WIDTH-1:0] ind      [NUM-1:0],
    input  logic                 in_valid,
    output logic                 in_ready,
    output logic [OUT_WIDTH-1:0] outd,
    output logic                 out_valid,
    input  logic                 out_ready
);

  localparam LEVELS = $clog2(NUM);

  // Declare intermediate values at each level
  for (genvar i = 0; i <= LEVELS; i++) begin : vars
    // The number of inputs at each level
    // level_num = ceil(num/(2^i))
    localparam LEVEL_NUM = (NUM + ((1 << i) - 1)) >> i;
    // The input data array at each level 
    // When i = 0, data is the input of the adder tree.
    // When i = level, data is the output of the adder tree.
    logic [(IN_WIDTH + i)-1:0] data[LEVEL_NUM-1:0];
    // Each level has a pair of handshake signals 
    // When i = 0, they are the handshake logic of the input.
    // When i = level, they are the handshake logic of the output.
    logic valid;
    logic ready;
  end

  // Generate adder for each layer
  for (genvar i = 0; i < LEVELS; i++) begin : level
    // The number of inputs at each level
    localparam LEVEL_NUM = (NUM + ((1 << i) - 1)) >> i;
    // The number of adders needed at each level
    // which is the number of the inputs at next level
    localparam NEXT_LEVEL_NUM = (LEVEL_NUM + 1) / 2;
    // The sum array is the output of the adders
    logic [(IN_WIDTH + i):0] sum[NEXT_LEVEL_NUM-1:0];

    // The width of the data increases by 1 for the next
    // level in order to keep the carry bit from the addition
    adder_tree_layer #(
        .NUM(LEVEL_NUM),
        .IN_WIDTH(IN_WIDTH + i)
    ) layer (
        .ins (vars[i].data),
        .outs(sum)
    );

    // Cocotb/verilator does not support array flattening, so
    // we need to manually add some reshaping process.

    // Casting array for sum
    logic [$bits(sum)-1:0] cast_sum;
    for (genvar j = 0; j < NEXT_LEVEL_NUM; j++)
      assign cast_sum[(IN_WIDTH+i+1)*j+(IN_WIDTH+i):(IN_WIDTH+i+1)*j] = sum[j];

    register_slice #(
        .DATA_WIDTH($bits(sum)),
    ) register_slice (
        .clk    (clk),
        .rst    (rst),
        .w_valid(vars[i].valid),
        .w_ready(vars[i].ready),
        .w_data (cast_sum),
        .r_valid(vars[i+1].valid),
        .r_ready(vars[i+1].ready),
        .r_data (cast_data)
    );

    // Casting array for vars[i+1].data 
    logic [$bits(sum)-1:0] cast_data;
    for (genvar j = 0; j < NEXT_LEVEL_NUM; j++)
      assign vars[i+1].data[j] = cast_data[(IN_WIDTH+i+1)*j+(IN_WIDTH+i):(IN_WIDTH+i+1)*j];

  end

  // it will zero-extend automatically
  assign vars[0].data = ind;
  assign vars[0].valid = in_valid;
  assign in_ready = vars[0].ready;

  assign outd = vars[LEVELS].data[0];
  assign out_valid = vars[LEVELS].valid;
  assign vars[LEVELS].ready = out_ready;

endmodule
