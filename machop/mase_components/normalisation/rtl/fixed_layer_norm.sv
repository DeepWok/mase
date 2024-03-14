`timescale 1ns / 1ps
// TODO(jlsand): Currently, our LayerNorm normalises over every dimension of data passed to it.
// LayerNorm requires that the normalisation be possible to only happen over some dimensions
module fixed_layer_norm #(
    
    // parameter NUM_GROUPS
    // parameter NUM_CHANNELS
    // General input for normalisation
    
    parameter IN_WIDTH          = 8,
    parameter IN_FRAC_WIDTH     = 4, 

    parameter OUT_WIDTH = IN_WIDTH,
    parameter OUT_FRAC_WIDTH     = IN_FRAC_WIDTH,
    
    // IN_DEPTH describes the number of data points per sample.
    // In image contexts, IN_DEPTH = sum_product of C, H & W.
    parameter IN_DEPTH          = 16, 
    parameter OUT_DEPTH         = IN_DEPTH,
    parameter PARALLELISM       = 16, 

    // PARTS_PER_NORM describes how many partitions of the input
    // data (sample) should be considered per normalisation.
    // Must divide IN_DEPTH.
    
    // The default is 1. In this case, normalisation
    // is performed over all dimensions of the sample at once.
    // EXAMPLE: Input data is 20 RGBA 10x10 images with data stored as 
    // (N, C, H, W) = (20, 4, 10, 10) matrix yielding IN_DEPTH = C * H * W = 400.  
    // PARTS_PER_NORM = 1 will normalise each image with all channels at once.
    // PARTS_PER_NORM = C = 4 will normalise each image one channel at a time.
    // PARTS_PER_NORM = C * H = 40 will normalise one row at a time per image per channel. 
    parameter PARTS_PER_NORM = IN_DEPTH,

    parameter ELEMENTWISE_AFFINE = 0,
    parameter BIAS = 0

    // parameter STD_RAM_DEPTH = IN_DEPTH
) (
    input                   clk, 
    input                   rst, 
    
    // Input ports for data
    input   [IN_WIDTH-1:0]  data_in_0     [IN_DEPTH-1:0], 
    input                   data_in_0_valid,
    output                  data_in_0_ready, 

    // // Input ports for gamma AKA weights in torch terminology
    // input   [GAMMA_RAM_WIDTH-1:0]  gamma  [GAMMA_RAM_DEPTH-1:0], 
    // input                          gamma_valid,
    // output                         gamma_ready, 

    // // Input ports for beta AKA bias in torch terminology
    // input   [BETA_RAM_WIDTH-1:0]  beta     [BETA_RAM_DEPTH-1:0], 
    // input                         beta_valid,
    // output                        beta_ready, 
   
    // Input ports for mean
    // input   [sum_rAM_WIDTH-1:0]  mean     [sum_rAM_DEPTH-1:0], 
    // input                         mean_valid,
    // output                        sum_ready,

    // Input ports for standard devitation
    // input   [STD_RAM_WIDTH-1:0]   stdv     [STD_RAM_DEPTH-1:0], 
    // input                         stdv_valid,
    // output                        var_rady, 

    // Output ports for data
    output  [OUT_WIDTH-1:0] data_out_0    [OUT_DEPTH-1:0],
    output                  data_out_0_valid,
    input                   data_out_0_ready 

);
    logic valid_out_b; 
    logic valid_out_r; 

    // Sum register must account for largest possible sum of inputs 
    logic [$clog2(IN_DEPTH*(2 ** IN_WIDTH -1)):0]    sum_r; 
    logic [$clog2(IN_DEPTH*(2 ** IN_WIDTH -1)):0]    sum_b; 
    logic [$clog2(IN_DEPTH*(2 ** (2 * IN_WIDTH))):0 ]    var_r; 
    logic [$clog2(IN_DEPTH*(2 ** (2 * IN_WIDTH))):0 ]    var_b; 

    assign data_in_0_ready     = 1'b1;
    assign data_out_0_valid    = valid_out_r;

    always_comb
    begin
        valid_out_b     = data_in_0_valid; 
    end

    always_ff @(posedge clk)
    begin
        valid_out_r     <= valid_out_b;
        sum_r          <= sum_b;
        var_r         <= var_b;
    end

    always_comb
    begin
        sum_b = 0;
        // TODO: Take into account PARTS_PER_NORM
        for (int i = 0; i < IN_DEPTH; i++) begin
            sum_b += data_in_0[i]; // Sum over all elements of the array
        end

        var_b = 0;
        for (int i = 0; i < IN_DEPTH; i++) begin
            var_b += (data_in_0[i] - (sum_r >> 4)) * (data_in_0[i] - (sum_r >> 4));
        end
    end
    

    // //ACCESS MEMORY: 
    // initial
    // begin
    //     $readmemh(GAMMA_DATA_PATH,  gamma_ram);
    //     $readmemh(BETA_DATA_PATH,   beta_ram);
    //     $readmemh(MEAN_DATA_PATH,   sum_ram);
    //     $readmemh(STD_DATA_PATH,    std_ram); //N.B.: it is assumed that epsilon in inculded here
    //     //TODO: think if should provide gamma/std instead to multiply in hw instead of division
    // end

    generate
        genvar i;
        for (i = 0; i < IN_DEPTH; i++) 
        begin
            assign data_out_0[i] = var_b;
            // assign data_out_0[i] = (data_in_0[i] - (sum_r[IN_WIDTH-1:0] >> $clog(IN_DEPTH))) / $sqrt(var_r);
            // assign data_out_0[i] = (data_in_0[i] - sum_r)/ var_r;
        end
    endgenerate    
    
endmodule
