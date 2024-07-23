`timescale 1ns / 1ps

module tb_selu;

  parameter CLK_PERIOD = 10;  // Clock period in ns
  parameter FILENAME = "output.txt";

  // Define parameters for the module
  parameter DATA_IN_0_PRECISION_0 = 16;
  parameter DATA_IN_0_PRECISION_1 = 8;
  parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8;
  parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1;
  parameter DATA_IN_0_PARALLELISM_DIM_0 = 1;
  parameter DATA_IN_0_PARALLELISM_DIM_1 = 1;
  parameter DATA_OUT_0_PRECISION_0 = 32;
  parameter DATA_OUT_0_PRECISION_1 = 16;
  parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8;
  parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1;
  parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1;
  parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1;
  parameter SCALE_PRECISION_1 = 16;
  parameter ALPHA_PRECISION_1 = 16;
  parameter INPLACE = 0;

  // Inputs
  logic rst;
  logic clk = 1;
  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];

  // Outputs
  logic data_in_0_valid;
  logic data_in_0_ready;
  logic data_out_0_valid;
  logic data_out_0_ready;
  logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];

  // Instantiate the module
  fixed_selu #(
      .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),
      .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1),
      .DATA_OUT_0_TENSOR_SIZE_DIM_0(DATA_OUT_0_TENSOR_SIZE_DIM_0),
      .DATA_OUT_0_TENSOR_SIZE_DIM_1(DATA_OUT_0_TENSOR_SIZE_DIM_1),
      .DATA_OUT_0_PARALLELISM_DIM_0(DATA_OUT_0_PARALLELISM_DIM_0),
      .DATA_OUT_0_PARALLELISM_DIM_1(DATA_OUT_0_PARALLELISM_DIM_1),
      .SCALE_PRECISION_1(SCALE_PRECISION_1),
      .ALPHA_PRECISION_1(ALPHA_PRECISION_1),
      .INPLACE(INPLACE)
  ) dut (
      .clk(clk),
      .rst(rst),
      .data_in_0(data_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),
      .data_out_0(data_out_0),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready)
  );


  // Clock generation
  always begin
    #((CLK_PERIOD / 2)) clk = ~clk;
  end

  // Reset generation
  initial begin
    rst = 1;
    #20;  // Reset for 20 ns
    rst = 0;
  end

  int  file;  //for writing outputs to file
  real i;  //i iterates over multiple input values
  assign file = $fopen(FILENAME, "w");  // Open the output file
  initial begin

    //initial conditions	
    data_in_0[0] = 0;
    data_in_0_valid = 0;
    data_out_0_ready = 0;

    //waiting for reset to become inactive
    #30;

    // Loop to send input values
    for (i = -4; i <= 4; i += 0.05) begin
      // Apply the current input value
      data_in_0[0] = i * (1 << DATA_IN_0_PRECISION_1);
      data_in_0_valid = 1;
      data_out_0_ready = 1;

      // Wait for 1 clock cyle before sending next data
      #10;
    end

    //input data transmission is over. Hence valid is made zero
    data_in_0_valid  = 0;
    data_out_0_ready = 1;

    #1000;
    // Close the output file
    $fclose(file);

    // Terminate the simulation
    $finish;
  end

  //writing output data to file
  always @(posedge clk) begin
    if (data_out_0_valid) begin
      $fdisplay(file, " %f", $itor(data_out_0[0]) / (1 << DATA_OUT_0_PRECISION_1));
    end
  end

endmodule
