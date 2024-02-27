`timescale 1ns / 1ps
module fixed_batch_norm1d #(
    parameter IN_WIDTH          = 8,
    parameter IN_FRAC_WIDTH     = 4, 
    parameter IN_DEPTH          = 16, 
    parameter PARALLELISM       = 16, 

    parameter OUT_WIDTH         = IN_WIDTH,
    parameter OUT_FRAC_WIDTH    = IN_FRAC_WIDTH,
    parameter OUT_DEPTH         = IN_DEPTH,

    // parameter string GAMMA_DATA_PATH =  "/home/sv720/mase_fork/mase_group7/machop/mase_components/normalisation/rtl/memory/batch_norm_gamma_ram.dat",
    // parameter string BETA_DATA_PATH =   "/home/sv720/mase_fork/mase_group7/machop/mase_components/normalisation/rtl/memory/batch_norm_beta_ram.dat",
    // parameter string MEAN_DATA_PATH =   "/home/sv720/mase_fork/mase_group7/machop/mase_components/normalisation/rtl/memory/batch_norm_mean_ram.dat",
    // parameter string STD_DATA_PATH =    "/home/sv720/mase_fork/mase_group7/machop/mase_components/normalisation/rtl/memory/batch_norm_std_ram.dat"

    parameter GAMMA_RAM_WIDTH = IN_WIDTH,
    parameter GAMMA_RAM_DEPTH = IN_DEPTH,

    parameter BETA_RAM_WIDTH = IN_WIDTH,
    parameter BETA_RAM_DEPTH = IN_DEPTH,

    parameter MEAN_RAM_WIDTH = IN_WIDTH,
    parameter MEAN_RAM_DEPTH = IN_DEPTH,

    parameter STD_RAM_WIDTH = IN_WIDTH,
    parameter STD_RAM_DEPTH = IN_DEPTH
) (
    input                   clk, 
    input                   rst, 
    
    // Input ports for data
    input   [IN_WIDTH-1:0]  data_in_0     [IN_DEPTH-1:0], 
    input                   data_in_0_valid,
    output                  data_in_0_ready, 

    // Input ports for gamma AKA weights in torch terminology
    input   [GAMMA_RAM_WIDTH-1:0]  gamma  [GAMMA_RAM_DEPTH-1:0], 
    input                          gamma_valid,
    output                         gamma_ready, 

    // Input ports for beta AKA bias in torch terminology
    input   [BETA_RAM_WIDTH-1:0]  beta     [BETA_RAM_DEPTH-1:0], 
    input                         beta_valid,
    output                        beta_ready, 
   
    // Input ports for mean
    input   [MEAN_RAM_WIDTH-1:0]  mean     [MEAN_RAM_DEPTH-1:0], 
    input                         mean_valid,
    output                        mean_ready,

    // Input ports for standard devitation
    input   [STD_RAM_WIDTH-1:0]   stdv     [STD_RAM_DEPTH-1:0], 
    input                         stdv_valid,
    output                        stdv_ready, 

    // Output ports for data
    output  [OUT_WIDTH-1:0] data_out_0    [OUT_DEPTH-1:0],
    output                  data_out_0_valid,
    input                   data_out_0_ready 

);


    // logic [GAMMA_RAM_WIDTH-1:0] gamma_ram   [0:GAMMA_RAM_DEPTH-1];
    // logic [BETA_RAM_WIDTH-1:0]  beta_ram    [0:BETA_RAM_DEPTH-1];  
    // logic [MEAN_RAM_WIDTH-1:0]  mean_ram    [0:MEAN_RAM_DEPTH-1];
    // logic [STD_RAM_WIDTH-1:0]   std_ram     [0:STD_RAM_DEPTH-1];  


    logic                       valid_out_b; 
    logic                       valid_out_r; 

    assign data_in_0_ready     = 1'b1;
    assign data_out_0_valid    = valid_out_r;

    always_comb 
    begin 
        valid_out_b     = data_in_0_valid; 
    end

    always_ff @(posedge clk)
    begin 
        valid_out_r     <= valid_out_b; 
        //delay line as expect the TB requires small delay (+ real designs will have it so best to check)
    end
    

    // //ACCESS MEMORY: 
    // initial
    // begin
    //     $readmemh(GAMMA_DATA_PATH,  gamma_ram);
    //     $readmemh(BETA_DATA_PATH,   beta_ram);
    //     $readmemh(MEAN_DATA_PATH,   mean_ram);
    //     $readmemh(STD_DATA_PATH,    std_ram); //N.B.: it is assumed that epsilon in inculded here
    //     //TODO: think if should provide gamma/std instead to multiply in hw instead of division
    // end

    generate
        genvar i;
        for (i = 0; i < IN_DEPTH; i++) 
        begin
            assign data_out_0[i] = (data_in_0[i] - mean[i])*gamma[i]/ stdv[i] + beta[i];
        end
    endgenerate    
    
endmodule
