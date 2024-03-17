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
    input logic signed  [IN_WIDTH-1:0]  data_in_0     [IN_DEPTH-1:0], 
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
    // logic valid_out_b; 
    // logic valid_out_r; 
    // logic valid_out_rr;

    // parameter MEAN_WIDTH = $clog2(IN_DEPTH*(2 ** IN_WIDTH)); 
    // parameter VAR_WIDTH = $clog2(IN_DEPTH*( (2 ** IN_WIDTH)**2) );

    // // Sum register must account for largest possible sum of inputs 
    // logic signed [MEAN_WIDTH - 1:0]    mean_r; 
    // logic signed [MEAN_WIDTH + $clog2(IN_DEPTH)- 1 :0]    mean_i; 
    // logic signed [MEAN_WIDTH + $clog2(IN_DEPTH)- 1 :0]    sum_i; 
    // logic signed [MEAN_WIDTH - 1:0]    sum_b; 
    // logic signed [VAR_WIDTH -1:0]    var_r; 
    // logic signed [VAR_WIDTH -1:0]    var_b; 

    // logic signed  [IN_WIDTH-1:0]  var_b_i     [IN_DEPTH-1:0];
    
    
    // assign data_in_0_ready     = 1'b1;
    // assign data_out_0_valid    = valid_out_rr;

    // always_comb
    // begin
    //     valid_out_b     = data_in_0_valid; 
    // end


    // // assign mean_i[$clog2(IN_DEPTH):0] = 0;
    // // assign mean_i[$clog2(IN_DEPTH):0] = 0;
    // assign sum_i = sum_b << $clog2(IN_DEPTH);
    // assign mean_i = (sum_i / IN_DEPTH);

    // always_ff @(posedge clk)
    // begin
    //     valid_out_r     <= valid_out_b;
    //     valid_out_rr     <= valid_out_r;
    //     mean_r          <= (mean_i >> $clog2(IN_DEPTH));
    //     var_r         <= (var_b >> $clog2(IN_DEPTH));
    // end


    // always_comb
    // begin
    //     sum_b = 0;
    //     // TODO: Take into account PARTS_PER_NORM
    //     for (int i = 0; i < IN_DEPTH; i++) begin
    //         sum_b += $signed(data_in_0[i]); // Sum over all elements of the array
    //     end

    //     var_b = 0;
    //     for (int i = 0; i < IN_DEPTH; i++) begin
    //         var_b += ((data_in_0[i] << $clog2(IN_DEPTH)) - mean_i) ** 2;
    //     end

    //     for (int i = 0; i < IN_DEPTH; i++) begin
    //         var_b_i[i] = (data_in_0[i] - mean_r) ** 2;
    //     end
    // end

    // generate
    //     genvar i;
    //     for (i = 0; i < IN_DEPTH; i++) 
    //     begin
    //         assign data_out_0[i] = (var_r >> IN_FRAC_WIDTH);
    //         // assign data_out_0[i] = var_r[IN_WIDTH + IN_FRAC_WIDTH - 1:IN_FRAC_WIDTH];
    //         // assign data_out_0[i] = (data_in_0[i] - (sum_r[IN_WIDTH-1:0] >> $clog(IN_DEPTH))) / $sqrt(var_r);
    //         // assign data_out_0[i] = (data_in_0[i] - sum_r)/ var_r;
    //     end
    // endgenerate    

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



    always_comb
    begin
        // Convert the inputs to a larger bitwidth and a FP format with more frac. bits.        
        for (int i = 0; i < IN_DEPTH; i++) begin
            data_in_zero_padded[i][SUM_EXTRA_FRAC_WIDTH-1:0] = 1'b0; 
            data_in_zero_padded[i][SUM_EXTRA_FRAC_WIDTH+IN_WIDTH-1:SUM_EXTRA_FRAC_WIDTH] = data_in_0[i];
            data_in_zero_padded[i][SUM_WIDTH-1:SUM_EXTRA_FRAC_WIDTH+IN_WIDTH] = {{SUM_NUM_MSb_PADDING_BITS}{data_in_0[i][IN_WIDTH-1]}}; //sv720: sign extention
        end

        // Sum over the widened inputs.
        sum = '0;
        // TODO: Take into account PARTS_PER_NORM
        for (int i = 0; i < IN_DEPTH; i++) begin
            sum += data_in_zero_padded[i];
        end

        assign mean = sum / IN_DEPTH;

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
        .v_in_valid('1), //TODO: set meaningful value
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
    end

    always_ff @(posedge clk)
    begin
        valid_out_r     <= valid_out_b;
        valid_in_sqrt_r <= valid_in_sqrt_b;
        
    end
    
    assign data_out_0_valid     = valid_out_r;

    generate
        genvar i;
        for (i = 0; i < IN_DEPTH; i++) 
        begin
            assign data_out_0[i] = (sum >>> SUM_EXTRA_FRAC_WIDTH);
        end
    endgenerate
    
endmodule
