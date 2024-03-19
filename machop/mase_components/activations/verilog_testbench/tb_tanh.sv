`timescale 1ns / 1ps

module tb_tanh;

    // Parameters
    parameter CLK_PERIOD = 10; // Clock period in ns
    parameter FILENAME = "output.txt";
       // Define parameters for the module
    parameter DATA_IN_0_PRECISION_0 = 16;
    parameter DATA_IN_0_PRECISION_1 = 8;
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8;
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1;
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1;
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1;
    parameter DATA_OUT_0_PRECISION_0 = 18;
    parameter DATA_OUT_0_PRECISION_1 = 16;
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8;
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1;
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1;
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1;
    parameter SCALE_PRECISION_1 = 16;
    parameter ALPHA_PRECISION_1 = 16;
    parameter INPLACE = 0;
                                                 ;    
    
    // Inputs
    logic rst;
    logic clk = 0;
    logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];

    // Outputs
    logic exp_out;
    logic data_in_0_valid = 1;
    logic data_in_0_ready;
    logic data_out_0_valid;
    logic data_out_0_ready = 1;
    logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];

    // Instantiate the module
    fixed_tanh #(
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
        .DATA_OUT_0_PARALLELISM_DIM_1(DATA_OUT_0_PARALLELISM_DIM_1)
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
        #20; // Reset for 20 ns
        rst = 0;
    end
    
initial begin
// Open the output file
   int file;
   file = $fopen(FILENAME, "w");
           
   // Loop to send input values and write output to file
   for (real i = -4; i <= 4; i += 0.05) begin
   // Apply the current input value
   data_in_0[0] = i * (1 << DATA_IN_0_PRECISION_1);
              
   // Wait for some time
   #100;
                
   // Check if output is valid
   if (data_out_0_valid) begin
       // Write the output to the file
       $fdisplay(file,"%f %f",i,$itor(data_out_0[0])/ (1 << DATA_OUT_0_PRECISION_1));
       //$fdisplay(file, "Data Out 0: %s%d", (data_out_0[0] < 0) ? "-" : "", (data_out_0[0] < 0) ? -data_out_0[0] : data_out_0[0]);
   end
   end
       
            // Close the output file
            $fclose(file);
            
            // Terminate the simulation
            $finish;
        end    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    // Test stimulus
 /*   initial begin
        // Apply some input values
        data_in_0[0] = 16'hfd00; // Example input value
        
        // Wait for some time
        #100;

        // Check the output against expected values
        //if (exp_out == expected_output) begin
       //     $display("Test Passed!");
       // end else begin
       //     $display("Test Failed!");
       // end
        // Terminate the simulation
        $finish;
    end*/

endmodule