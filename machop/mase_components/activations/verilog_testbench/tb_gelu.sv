`timescale 1ns / 1ps

module gelu_test;

  logic rst;
  logic clk;
  logic [15:0] data_in_0[0:1];
  logic [47:0] data_out_0[0:1];

  logic data_in_0_ready;
  logic data_in_0_valid;
  logic data_out_0_ready;
  logic data_out_0_valid;


  fixed_gelu #(
      .DATA_IN_0_PARALLELISM_DIM_0 (2),
      .DATA_OUT_0_PARALLELISM_DIM_0(2)
  ) DUT (
      .*
  );



  initial begin
    clk = '0;
    forever #50 clk = ~clk;
  end

  initial begin
    integer file_descriptor;
    file_descriptor = $fopen("output_file.txt", "w");

    #50;
    data_out_0_ready = 1;
    data_in_0[0] = 10'd0;
    data_in_0_valid = 1;

    for (int i = 0; i < 32768; i++) begin
      data_in_0[0] = i - 16384;
      #100;
      $fwrite(file_descriptor, "%b  %b\n", data_in_0[0], data_out_0[0]);
    end
    /*

        data_in_0[0] = 4;
        #100;
        data_in_0[0] = -4;
        #100;
*/
    $fclose(file_descriptor);
  end

endmodule
;
