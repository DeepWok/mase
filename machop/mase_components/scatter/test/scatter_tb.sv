`timescale 1ns / 1ps

module scatter_threshold_tb;

  // Parameters of the module
  localparam PRECISION = 4;
  localparam TENSOR_SIZE_DIM = 8;
  localparam HIGH_SLOTS = 2;
  localparam THRESHOLD = 6;
  localparam DESIGN = 1;

  // Inputs
  reg clk;
  reg rst;
  reg [PRECISION-1:0] data_in[TENSOR_SIZE_DIM-1:0];

  // Outputs
  wire [PRECISION-1:0] o_high_precision[TENSOR_SIZE_DIM-1:0];
  wire [PRECISION-1:0] o_low_precision[TENSOR_SIZE_DIM-1:0];

  // Instantiate the Unit Under Test (UUT)
  scatter_threshold #(
      .PRECISION(PRECISION),
      .TENSOR_SIZE_DIM(TENSOR_SIZE_DIM),
      .HIGH_SLOTS(HIGH_SLOTS),
      .THRESHOLD(THRESHOLD),
      .DESIGN(DESIGN)
  ) uut (
      .clk(clk),
      .rst(rst),
      .data_in(data_in),
      .o_high_precision(o_high_precision),
      .o_low_precision(o_low_precision)
  );

  // Clock generation
  initial begin
    clk = 0;
    forever #2.5 clk = ~clk;  // 200MHz Clock
  end

  // Random input stimulus
  task generate_random_input;
    integer i;
    begin
      for (i = 0; i < TENSOR_SIZE_DIM; i = i + 1) begin
        data_in[i] = $random % (1 << PRECISION);
      end
    end
  endtask

  // Test sequence
  initial begin
    // Initialize Inputs
    rst = 1;
    generate_random_input();

    // Wait for global reset
    #100;

    // Release the reset
    rst = 0;

    // Change the inputs at random times
    forever begin
      #($random % 50 + 5) generate_random_input();  // Adjusted for faster clock
    end
  end

  // Stop simulation after 2us
  initial begin
    #200;  // 2us
    $finish;
  end

  initial begin
    $dumpfile("dump.vcd");
    $dumpvars(0, scatter_threshold_tb);

  end


endmodule
