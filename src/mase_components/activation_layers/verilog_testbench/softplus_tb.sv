`timescale 1ns / 1ps

module softplus_tb;

  parameter DATA_IN_0_PRECISION_0 = 16;
  parameter DATA_IN_0_PRECISION_1 = 1;
  parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1;
  parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1;
  parameter DATA_IN_0_PARALLELISM_DIM_0 = 1;
  parameter DATA_IN_0_PARALLELISM_DIM_1 = 1;

  parameter DATA_OUT_0_PRECISION_0 = 32;
  parameter DATA_OUT_0_PRECISION_1 = 1;
  parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1;
  parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1;
  parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1;
  parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1;

  reg rst;
  reg clk;
  reg [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
  reg data_in_0_valid;
  reg data_out_0_ready;

  wire [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  wire data_in_0_ready;
  wire data_out_0_valid;

  fixed_softplus #(
      .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_TENSOR_SIZE_DIM_0(DATA_OUT_0_TENSOR_SIZE_DIM_0),
      .DATA_OUT_0_PARALLELISM_DIM_0(DATA_OUT_0_PARALLELISM_DIM_0)
  ) dut (
      .rst(rst),
      .clk(clk),
      .data_in_0(data_in_0),
      .data_out_0(data_out_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready)
  );

  always #5 clk = ~clk;

  initial begin
    clk = 0;
    rst = 1;
    data_in_0[0] = 0;
    data_in_0_valid = 0;


    #10 rst = 0;

    //Input= -8
    #10 data_in_0[0] = 16'hFFF8;
    data_in_0_valid  = 1;
    data_out_0_ready = 1;
    #10 data_in_0_valid = 0;
    data_out_0_ready = 0;
    #20;

    //Input= -7
    #10 data_in_0[0] = 16'hFFF9;
    data_in_0_valid  = 1;
    data_out_0_ready = 1;
    #10 data_in_0_valid = 0;
    #20;

    //Input= -6
    #10 data_in_0[0] = 16'hFFFA;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;

    //Input= -5
    #10 data_in_0[0] = 16'hFFFB;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;

    //Input= -4
    #10 data_in_0[0] = 16'hFFFC;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;

    //input= -3
    #10 data_in_0[0] = 16'hFFFD;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;

    //input= -2
    #10 data_in_0[0] = 16'hFFFE;
    data_in_0_valid  = 1;
    data_out_0_ready = 1;
    #10 data_in_0_valid = 0;
    #20;

    //input= -1
    #10 data_in_0[0] = 16'hFFFF;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;

    //input= 0
    #10 data_in_0[0] = 16'h0000;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;

    //Input = 1
    #10 data_in_0[0] = 16'd0001;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;

    //Input = 2
    #10 data_in_0[0] = 16'd0002;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;  // Wait for the output

    //Input = 3
    #10 data_in_0[0] = 16'd0003;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;  // Wait for the output

    //Input = 4
    #10 data_in_0[0] = 16'd0004;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;  // Wait for the output

    //Input = 5
    #10 data_in_0[0] = 16'd0005;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;  // Wait for the output

    //Input = 6
    #10 data_in_0[0] = 16'd0006;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;  // Wait for the output

    //Input = 7
    #10 data_in_0[0] = 16'd0007;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;  // Wait for the output

    //Input = 8
    #10 data_in_0[0] = 16'd0008;
    data_in_0_valid = 1;
    #10 data_in_0_valid = 0;
    #20;  // Wait for the output

  end


  //MONITOR THE OUTPUT AND SAVE IT TO A FILE
  integer fresult;
  initial begin
    fresult = $fopen("output.txt", "w");
  end

  //helper task  to save results to a file
  task save_result;
    input [DATA_OUT_0_PRECISION_0-1:0] result;
    begin
      $fwrite(fresult, "%d\n", result);
    end
  endtask

  always @(posedge clk) begin
    if (data_out_0_valid) begin
      save_result(data_out_0[0]);
    end
  end


  always @(posedge clk) begin
    if (data_out_0_valid) begin
      $display("Output: %d\n", data_out_0[0]);
    end
  end

  initial begin
    //SIMULATION TIME :1000ns
    #1000;
    $fclose(fresult);
    //$finish;
  end


endmodule
