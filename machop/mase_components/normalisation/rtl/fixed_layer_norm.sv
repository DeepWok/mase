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
    input                   reset_n, 
    
    // Input ports for data
    input logic signed  [IN_WIDTH-1:0]  data_in_0     [IN_DEPTH-1:0], 
    input                   data_in_0_valid,
    output                  data_in_0_ready, 

    input logic signed  [IN_WIDTH-1:0]  beta_in     [IN_DEPTH-1:0],
    input logic signed  [IN_WIDTH-1:0]  gamma_in     [IN_DEPTH-1:0],
      
    // Output ports for data
    output  signed [OUT_WIDTH-1:0] data_out_0    [OUT_DEPTH-1:0],
    output                  data_out_0_valid,
    input                   data_out_0_ready 

);
    
    // The max size the sum for calculating the mean is
    // MAX_NUM_ELEMS * MAX_SIZE_PER_ELEM = IN_DEPTH * 2^IN_WIDTH
    parameter SUM_MAX_SIZE = IN_DEPTH*(2 ** IN_WIDTH);

    // We need larger bitwidth than the inputs.
    parameter SUM_EXTRA_FRAC_WIDTH = $clog2(IN_DEPTH); 
    parameter SUM_WIDTH = $clog2(SUM_MAX_SIZE) + SUM_EXTRA_FRAC_WIDTH;
    parameter SUM_FRAC_WIDTH = IN_FRAC_WIDTH + SUM_EXTRA_FRAC_WIDTH;
    parameter SUM_NUM_MSb_PADDING_BITS = SUM_WIDTH - IN_WIDTH - SUM_EXTRA_FRAC_WIDTH;

    parameter SUM_SQUARED_BITS = $clog2(SUM_MAX_SIZE**2) + SUM_EXTRA_FRAC_WIDTH*2;
    parameter SUM_SQUARED_FRAC_WIDTH = 2*(IN_FRAC_WIDTH + SUM_EXTRA_FRAC_WIDTH);

    parameter SUM_OF_SQUARES_BITS = SUM_SQUARED_BITS + $clog2(IN_DEPTH);
    parameter SUM_OF_SQUARES_FRAC_WIDTH = SUM_SQUARED_FRAC_WIDTH;

    parameter SUM_OF_SQUARES_BITS_PADDED = SUM_OF_SQUARES_BITS + $clog2(IN_DEPTH);
    parameter VAR_BITS_PADDED = SUM_OF_SQUARES_BITS_PADDED;
    parameter VAR_BITS = SUM_OF_SQUARES_BITS;
    parameter VAR_FRAC_WIDTH = SUM_SQUARED_FRAC_WIDTH + $clog2(IN_DEPTH); //sv720: division by depth -> less integer bits

    parameter EPSILON = 1; 

    logic rst = ~reset_n;

    logic signed  [IN_WIDTH-1:0]  data_r     [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_b     [IN_DEPTH-1:0]; 

    logic signed  [IN_WIDTH-1:0]  beta_r    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  gamma_r   [IN_DEPTH-1:0];

    logic signed  [IN_WIDTH-1:0]  beta_b    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  gamma_b   [IN_DEPTH-1:0];


    logic signed    [SUM_WIDTH - 1:0]   sum;
    logic signed    [SUM_WIDTH - 1:0]   mean;
    logic signed    [SUM_WIDTH - 1:0]   data_in_zero_padded [IN_DEPTH];

    logic signed    [SUM_WIDTH - 1:0]             data_in_minus_mean          [IN_DEPTH-1:0];
    logic           [SUM_SQUARED_BITS - 1:0]      data_in_minus_mean_squared  [IN_DEPTH-1:0];

    logic           [SUM_OF_SQUARES_BITS - 1:0]         sum_of_squared_differences; 
    logic           [SUM_OF_SQUARES_BITS_PADDED - 1:0]  sum_of_squared_differences_padded;
    logic           [VAR_BITS - 1:0]                    variance;
    logic           [VAR_BITS_PADDED - 1:0]             variance_padded;
    logic           [IN_WIDTH - 1:0]                    variance_in_width;
    logic           [IN_WIDTH - 1:0]                    standard_deviation;

    

    logic signed  [IN_WIDTH-1:0]  normalised_data_b    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  normalised_data_r    [IN_DEPTH-1:0];



    always_comb
    begin
        // Convert the inputs to a larger bitwidth and a FP format with more frac. bits.        
        for (int i = 0; i < IN_DEPTH; i++) begin
            data_in_zero_padded[i][SUM_EXTRA_FRAC_WIDTH-1:0] = 1'b0; 
            data_in_zero_padded[i][SUM_EXTRA_FRAC_WIDTH+IN_WIDTH-1:SUM_EXTRA_FRAC_WIDTH] = data_r[i];
            data_in_zero_padded[i][SUM_WIDTH-1:SUM_EXTRA_FRAC_WIDTH+IN_WIDTH] = {{SUM_NUM_MSb_PADDING_BITS}{data_r[i][IN_WIDTH-1]}}; //sv720: sign extention
        end

        // Sum over the widened inputs.
        sum = '0;
        // TODO: Take into account PARTS_PER_NORM
        for (int i = 0; i < IN_DEPTH; i++) begin
            sum += data_in_zero_padded[i];
        end

        mean = sum / IN_DEPTH;

        sum_of_squared_differences = '0;

        for (int i = 0; i < IN_DEPTH; i++) begin
            data_in_minus_mean[i] =         data_in_zero_padded[i] - mean;
            data_in_minus_mean_squared[i] = data_in_minus_mean[i]**2;
            sum_of_squared_differences +=   data_in_minus_mean_squared[i];
        end

        sum_of_squared_differences_padded = (sum_of_squared_differences << $clog2(IN_DEPTH) );

        variance_padded = sum_of_squared_differences_padded / IN_DEPTH;

        variance = variance_padded[VAR_BITS-1:0];
        variance_in_width = variance[ IN_WIDTH + VAR_FRAC_WIDTH - IN_FRAC_WIDTH -1 : VAR_FRAC_WIDTH - IN_FRAC_WIDTH ];


        for (int i=0; i<IN_DEPTH; i++)
        begin
            normalised_data_b[i] = (data_r[i] - mean)/(standard_deviation + EPSILON)*gamma_r[i] + beta_r[i];
        end
    
    end

    logic sqrt_v_in_ready; //TODO: use this
    logic sqrt_v_out_valid; //TODO: use this

    sqrt #(
        .IN_WIDTH(IN_WIDTH),
        .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
        .NUM_ITERATION(10)
    ) sqrt_cordic (
        .clk(clk),
        .rst(rst),
        .v_in(variance_in_width),
        .v_in_valid(valid_in_sqrt_r), //TODO: set meaningful value
        .v_in_ready(sqrt_v_in_ready),

        .v_out(standard_deviation),
        .v_out_valid(sqrt_v_out_valid),
        .v_out_ready('1) //TODO: assign this and check in module
    );


    // Data outputs.
    assign data_in_0_ready     = 1'b1;

    logic valid_out_b; 
    logic valid_out_r; 

    logic valid_in_sqrt_b;
    logic valid_in_sqrt_r;

    always_comb
    begin
        valid_in_sqrt_b     = data_in_0_valid; 
        valid_out_b         = sqrt_v_out_valid;

        normalised_data_b   = normalised_data_r;

        data_b    = data_r;
        beta_b    = beta_r; 
        gamma_b   = gamma_r;  

        if (data_in_0_valid)
        begin 
            data_b    = data_in_0;
            beta_b    = beta_in;
            gamma_b   = gamma_in;
        end
    end

    always_ff @(posedge clk) //TODO: add asynchronous reset behaviour
    begin
        data_r          <= data_b;
        valid_out_r     <= valid_out_b;
        valid_in_sqrt_r <= valid_in_sqrt_b;
        beta_r          <= beta_b;
        gamma_r         <= gamma_b;
        normalised_data_r <= normalised_data_b;
    end
    
    assign data_out_0_valid     = valid_out_r;


    assign data_out_0 = normalised_data_r;
    // generate
    //     genvar i;
    //     for (i = 0; i < IN_DEPTH; i++) 
    //     begin
    //         assign data_out_0[i] = (sum >>> SUM_EXTRA_FRAC_WIDTH);
    //     end
    // endgenerate
    
endmodule
