`timescale 1ns / 1ps
module fixed_batch_norm1d #(
    parameter IN_WIDTH      = 8,
    parameter IN_FRAC_WIDTH = 4, 
    parameter IN_DEPTH      = 16, 
    parameter PARALLELISM   = 16, 

    parameter OUT_WIDTH     = IN_WIDTH,
    parameter OUT_DEPTH     = OUT_DEPTH,

    parameter string GAMMA_DATA_PATH = "/home/sv720/mase/top/hardware/rtl/batch_norm_gamma_ram.data",
    parameter string BETA_DATA_PATH = "/home/sv720/mase/top/hardware/rtl/batch_norm_beta_ram.data",
    parameter string MEAN_DATA_PATH = "/home/sv720/mase/top/hardware/rtl/batch_norm_mean_ram.data",
    parameter string STD_DATA_PATH = "/home/sv720/mase/top/hardware/rtl/batch_norm_std_ram.data"


) (
    input                   clk, 
    input   [IN_WIDTH-1:0]  data_in     [IN_DEPTH-1:0], 
    input                   valid_in,

    output  [OUT_WIDTH-1:0] data_out    [OUT_DEPTH-1:0],
    output                  valid_out
);
    localparam  GAMMA_RAM_WIDTH = IN_WIDTH;
    localparam  GAMMA_RAM_DEPTH = IN_DEPTH;

    localparam  BETA_RAM_WIDTH = IN_WIDTH;
    localparam  BETA_RAM_DEPTH = IN_DEPTH;

    localparam  MEAN_RAM_WIDTH = IN_WIDTH;
    localparam  MEAN_RAM_DEPTH = IN_DEPTH;

    localparam  STD_RAM_WIDTH = IN_WIDTH;
    localparam  STD_RAM_DEPTH = IN_DEPTH;


    logic [GAMMA_RAM_WIDTH-1:0] gamma_ram   [0:GAMMA_RAM_DEPTH-1];
    logic [BETA_RAM_WIDTH-1:0]  beta_ram    [0:BETA_RAM_DEPTH-1];  
    logic [MEAN_RAM_WIDTH-1:0]  mean_ram    [0:MEAN_RAM_DEPTH-1];
    logic [STD_RAM_WIDTH-1:0]   std_ram     [0:STD_RAM_DEPTH-1];  

    

    //ACCESS MEMORY: 
    initial
    begin
        $readmemh(GAMMA_DATA_PATH,  gamma_ram);
        $readmemh(BETA_DATA_PATH,   beta_ram);
        $readmemh(MEAN_DATA_PATH,   mean_ram);
        $readmemh(STD_DATA_PATH,    std_ram); //N.B.: it is assumed that epsilon in inculded here
        //TODO: think if should provide gamma/std instead to multiply in hw instead of division
    end

    generate
        genvar i;
        for (i = 0; i <= IN_DEPTH; i++) 
        begin
            assign data_out[i] = (data_in[i] - mean_ram[i])*gamma_ram[i]/std_ram[i] + beta_ram[i];
        end
    endgenerate    
    
endmodule
