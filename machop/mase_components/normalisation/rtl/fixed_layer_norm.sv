`timescale 1ns / 1ps
// TODO(jlsand): Currently, our LayerNorm normalises over every dimension of data passed to it.
// LayerNorm requires that the normalisation be possible to only happen over some dimensions
module fixed_layer_norm #(


    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 16,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 16,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    
    parameter WEIGHT_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter WEIGHT_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter WEIGHT_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,

    parameter BIAS_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter BIAS_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter BIAS_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,

    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,


    // ------ We need to use the above parameters a lot, rename some of the most used ----------
    parameter IN_WIDTH          = DATA_IN_0_PRECISION_0,
    parameter IN_FRAC_WIDTH     = DATA_IN_0_PRECISION_1, 

    parameter OUT_WIDTH         = IN_WIDTH,
    parameter OUT_FRAC_WIDTH    = IN_FRAC_WIDTH,
    
    // IN_DEPTH describes the number of data points per sample.
    // In image contexts, IN_DEPTH = sum_product of C, H & W.
    parameter IN_DEPTH          = DATA_IN_0_PARALLELISM_DIM_0, 
    parameter OUT_DEPTH         = IN_DEPTH,
    parameter NUM_NORMALIZATION_ZONES = 1, 
    // parameter NUM_NORMALIZATION_ZONES = IN_DEPTH/2, 

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
    parameter PARTS_PER_NORM = IN_DEPTH
) (
    input                   clk, 
    input                   rst, 
    
    // Input ports for data
    input logic signed  [IN_WIDTH-1:0]  data_in_0     [DATA_IN_0_PARALLELISM_DIM_0-1:0], 
    input                               data_in_0_valid,
    output                              data_in_0_ready, 

    input logic signed  [IN_WIDTH-1:0]  bias     [IN_DEPTH-1:0],
    input                               bias_valid,
    output                              bias_ready, 
    
    input logic signed  [IN_WIDTH-1:0]  weight    [IN_DEPTH-1:0],
    input                               weight_valid,
    output                              weight_ready, 

    // Output ports for data
    output  signed [OUT_WIDTH-1:0] data_out_0    [DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output                         data_out_0_valid,
    input                          data_out_0_ready 

);
    
    localparam NORMALIZATION_ZONE_PERIOD = IN_DEPTH/NUM_NORMALIZATION_ZONES;

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

    parameter EPSILON = 0;

    parameter NUM_STATE_BITS = 32;
    parameter VALID_IN_DELAY_DELAY_LINE_SIZE = 6;
    parameter DATA_DELAY_LINE_SIZE = 12;

    typedef enum logic [NUM_STATE_BITS-1:0] {
        RST_STATE       = '0,
        MEAN_SUM_STATE  = 1, 
        MEAN_DIV_STATE  = 2, 
        SUB_STATE       = 3,
        SQUARING_STATE  = 4,
        SUM_SQU_STATE   = 5,
        VAR_DIV_STATE   = 6,  //if right shift -> drop this
        NORM_DIFF_STATE = 7, 
        WAITING_FOR_SQRT    = 8,
        NORM_DIV_STATE  = 9, 
        NORM_MULT_STATE = 10, 
        NORM_ADD_STATE  = 11,

        READY_STATE = 12, 
        UNASSIGNED  = 13,
        DONE        = '1
    } state_t;

    state_t                     state_b; 
    state_t                     state_r;
    // logic rst = ~reset_n;

    logic signed  [IN_WIDTH-1:0]  data_r7    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_r3    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_r     [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_b     [IN_DEPTH-1:0]; 

    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r11   [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r10   [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r9    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r8    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r7    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r6    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r5    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r4    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r3    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r2    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r1    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_r     [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_b     [IN_DEPTH-1:0]; 

    logic signed  [IN_WIDTH-1:0]  beta_r    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  gamma_r   [IN_DEPTH-1:0];

    logic signed  [IN_WIDTH-1:0]  beta_b    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  gamma_b   [IN_DEPTH-1:0];


    logic signed    [SUM_WIDTH - 1:0]   sum_b [NUM_NORMALIZATION_ZONES-1:0];
    logic signed    [SUM_WIDTH - 1:0]   sum_r [NUM_NORMALIZATION_ZONES-1:0];

    logic signed    [SUM_WIDTH - 1:0]   mean_b [NUM_NORMALIZATION_ZONES-1:0];
    logic signed    [SUM_WIDTH - 1:0]   mean_r [NUM_NORMALIZATION_ZONES-1:0];
    logic signed    [SUM_WIDTH - 1:0]   mean_r2 [NUM_NORMALIZATION_ZONES-1:0];
    logic signed    [SUM_WIDTH - 1:0]   mean_r3 [NUM_NORMALIZATION_ZONES-1:0];
    logic signed    [SUM_WIDTH - 1:0]   mean_r4 [NUM_NORMALIZATION_ZONES-1:0];
    logic signed    [SUM_WIDTH - 1:0]   mean_r5 [NUM_NORMALIZATION_ZONES-1:0];

    logic signed    [SUM_WIDTH - 1:0]   data_r_zero_padded      [IN_DEPTH];
    logic signed    [SUM_WIDTH - 1:0]   data_r_r_r_zero_padded  [IN_DEPTH];

    logic signed    [SUM_WIDTH - 1:0]             data_in_minus_mean_b          [IN_DEPTH-1:0];
    logic signed    [SUM_WIDTH - 1:0]             data_in_minus_mean_r          [IN_DEPTH-1:0];
    logic           [SUM_SQUARED_BITS - 1:0]      data_in_minus_mean_squared_b  [IN_DEPTH-1:0];
    logic           [SUM_SQUARED_BITS - 1:0]      data_in_minus_mean_squared_r  [IN_DEPTH-1:0];

    logic           [SUM_OF_SQUARES_BITS - 1:0]         sum_of_squared_differences_b        [NUM_NORMALIZATION_ZONES-1:0]; 
    logic           [SUM_OF_SQUARES_BITS - 1:0]         sum_of_squared_differences_r        [NUM_NORMALIZATION_ZONES-1:0];
     
    logic           [SUM_OF_SQUARES_BITS - 1:0]         sum_of_squared_differences_tmp      [NUM_NORMALIZATION_ZONES-1:0]; 
    logic           [SUM_OF_SQUARES_BITS_PADDED - 1:0]  sum_of_squared_differences_padded   [NUM_NORMALIZATION_ZONES-1:0];
    logic           [VAR_BITS - 1:0]                    variance                            [NUM_NORMALIZATION_ZONES-1:0];
    logic           [VAR_BITS_PADDED - 1:0]             variance_padded_b                   [NUM_NORMALIZATION_ZONES-1:0];
    logic           [VAR_BITS_PADDED - 1:0]             variance_padded_r                   [NUM_NORMALIZATION_ZONES-1:0];
    logic           [IN_WIDTH - 1:0]                    variance_in_width                   [NUM_NORMALIZATION_ZONES-1:0];
    logic           [IN_WIDTH - 1:0]                    sqrt_out                            [NUM_NORMALIZATION_ZONES-1:0];
    logic           [IN_WIDTH - 1:0]                    standard_deviation_b                [NUM_NORMALIZATION_ZONES-1:0];
    logic           [IN_WIDTH - 1:0]                    standard_deviation_r                [NUM_NORMALIZATION_ZONES-1:0];

    logic signed  [IN_WIDTH-1:0]  normalised_data_b    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  normalised_data_r    [IN_DEPTH-1:0];

    logic signed  [IN_WIDTH-1:0]  data_minus_mean_div_by_std_b    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_div_by_std_r    [IN_DEPTH-1:0];

    logic signed  [IN_WIDTH-1:0]  data_minus_mean_div_by_std_times_gamma_b    [IN_DEPTH-1:0];
    logic signed  [IN_WIDTH-1:0]  data_minus_mean_div_by_std_times_gamma_r    [IN_DEPTH-1:0];

    

    

    logic sqrt_v_in_ready; //TODO: use this
    logic [NUM_NORMALIZATION_ZONES-1:0] sqrt_v_out_valid; //TODO: use this

    logic sqrt_valid_out_b; 
    logic sqrt_valid_out_r; 
    logic sqrt_valid_out_r2; 
    logic sqrt_valid_out_r3; 
    logic sqrt_valid_out_r4; 
    logic sqrt_valid_out_r5; 

    // logic valid_in_sqrt_b;
    // logic valid_in_sqrt_r;
    logic valid_in_sqrt;

    logic [VALID_IN_DELAY_DELAY_LINE_SIZE - 1: 0]   data_in_valid_delay_line_b; 
    logic [VALID_IN_DELAY_DELAY_LINE_SIZE - 1: 0]   data_in_valid_delay_line_r; 

    logic           [IN_WIDTH-1:0]                  data_r_delay_line_b [DATA_DELAY_LINE_SIZE-1:0] [IN_DEPTH-1:0] ;
    logic           [IN_WIDTH-1:0]                  data_r_delay_line_r [DATA_DELAY_LINE_SIZE-1:0] [IN_DEPTH-1:0] ;
    logic signed    [IN_WIDTH-1:0]                  data_array_zeros    [IN_DEPTH-1:0];

    assign valid_in_sqrt = data_in_valid_delay_line_r [VALID_IN_DELAY_DELAY_LINE_SIZE - 1];


    assign data_r7 = data_r_delay_line_r[6][IN_DEPTH-1:0];
    assign data_r3 = data_r_delay_line_r[2][IN_DEPTH-1:0];

    
    always_comb
    begin

        for (int i=0; i < IN_DEPTH; i++)
        begin
            data_array_zeros[i] = '1;
        end

        for (int i=0; i < IN_DEPTH; i++)
        begin
            data_in_valid_delay_line_b[i] = '0;
        end

        
        state_b             = state_r; 

        // valid_in_sqrt_b     = '0; 
        sqrt_valid_out_b         = '0;

        normalised_data_b   = normalised_data_r;

        data_b  = data_r; 
        beta_b  = beta_r; 
        gamma_b = gamma_r;
        sum_b   = sum_r;  
        mean_b  = mean_r;

    

        sum_of_squared_differences_b    = sum_of_squared_differences_r;
        
        for (int j=0; j < NUM_NORMALIZATION_ZONES; j++)
        begin
            sum_of_squared_differences_tmp[j]   = '0;
        end

        data_minus_mean_b   = data_minus_mean_r;

        standard_deviation_b = standard_deviation_r;
        data_minus_mean_div_by_std_b = data_minus_mean_div_by_std_r;

        data_minus_mean_div_by_std_times_gamma_b = data_minus_mean_div_by_std_times_gamma_r;

        if (data_in_0_valid)
        begin 
            data_in_valid_delay_line_b[0] = 1'b1;
            state_b   = MEAN_SUM_STATE;
            data_b    = data_in_0;
            beta_b    = bias;
            gamma_b   = weight;
        end
        else
        begin
            data_in_valid_delay_line_b = data_in_valid_delay_line_r << 1;
        end

        data_r_delay_line_b[0] = data_b;
        
        for (int i =1; i < DATA_DELAY_LINE_SIZE; i++)
        begin
            data_r_delay_line_b[i] = data_r_delay_line_r[i-1];
        end


        // Convert the inputs to a larger bitwidth and a FP format with more frac. bits.        
        for (int i = 0; i < IN_DEPTH; i++) begin
            data_r_zero_padded[i][SUM_WIDTH-1:0] = 0; 
            data_r_zero_padded[i][SUM_WIDTH-1:SUM_WIDTH-IN_WIDTH] = data_r[i];
            data_r_zero_padded[i] = data_r_zero_padded[i] >>> (SUM_WIDTH - IN_WIDTH - SUM_EXTRA_FRAC_WIDTH);

        end

        //needed for SUB_STATE
        for (int i = 0; i < IN_DEPTH; i++) begin
            data_r_r_r_zero_padded[i][SUM_WIDTH-1:0] = 0; 
            data_r_r_r_zero_padded[i][SUM_WIDTH-1:SUM_WIDTH-IN_WIDTH] = data_r3[i];
            data_r_r_r_zero_padded[i] = data_r_r_r_zero_padded[i] >>> (SUM_WIDTH - IN_WIDTH - SUM_EXTRA_FRAC_WIDTH);
        end

        

        for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++)
        begin
            sum_of_squared_differences_padded[j] = (sum_of_squared_differences_r[j] << $clog2(IN_DEPTH) );

            variance[j] = variance_padded_b[j][VAR_BITS-1:0];
            variance_in_width[j] = variance[j][ IN_WIDTH + VAR_FRAC_WIDTH - IN_FRAC_WIDTH -1 : VAR_FRAC_WIDTH - IN_FRAC_WIDTH ];
            
            if (&sqrt_v_out_valid)
            begin 
                standard_deviation_b = sqrt_out; 
                sqrt_valid_out_b         = '1;
            end
        end

         // TODO: Take into account PARTS_PER_NORM //sv720:DONE
            for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++)
            begin
                // Sum over the widened inputs.
                sum_b[j] = 0;
                for (int i = 0; i < NORMALIZATION_ZONE_PERIOD; i++) begin
                    sum_b[j] += data_r_zero_padded[i+j*NORMALIZATION_ZONE_PERIOD];
                end
            end

            for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++)
            begin
                mean_b[j]  = sum_r[j] / NORMALIZATION_ZONE_PERIOD;
            end

            for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++)
            begin
                for (int i = 0; i < NORMALIZATION_ZONE_PERIOD; i++) 
                begin
                    data_in_minus_mean_b[i+j*NORMALIZATION_ZONE_PERIOD] =         data_r_r_r_zero_padded[i+ j*NORMALIZATION_ZONE_PERIOD] - mean_r[j];
                end
            end

            for (int i = 0; i < IN_DEPTH; i++) begin
                data_in_minus_mean_squared_b[i] = data_in_minus_mean_r[i]**2;
            end

            for (int j=0; j < NUM_NORMALIZATION_ZONES; j++)
            begin
                sum_of_squared_differences_tmp[j]   = '0;

                for (int i = 0; i < NORMALIZATION_ZONE_PERIOD; i++) begin
                    sum_of_squared_differences_tmp[j] +=   data_in_minus_mean_squared_r[i+j*NORMALIZATION_ZONE_PERIOD];
                end
            end
            
            sum_of_squared_differences_b = sum_of_squared_differences_tmp;

            for (int j=0; j < NUM_NORMALIZATION_ZONES; j++)
            begin 
                variance_padded_b[j] = sum_of_squared_differences_padded[j] / NORMALIZATION_ZONE_PERIOD;
            end

            for (int j=0; j<NUM_NORMALIZATION_ZONES; j++)
            begin
                for (int i=0; i<NORMALIZATION_ZONE_PERIOD; i++)
                begin
                    data_minus_mean_b[i+j*NORMALIZATION_ZONE_PERIOD] = (data_r7[i+j*NORMALIZATION_ZONE_PERIOD] - mean_r5[j][ IN_WIDTH + SUM_FRAC_WIDTH - IN_FRAC_WIDTH - 1:SUM_FRAC_WIDTH - IN_FRAC_WIDTH ]);
                    // data_minus_mean_b[i+j*NORMALIZATION_ZONE_PERIOD] = (data_r[i+j*NORMALIZATION_ZONE_PERIOD] - mean_r[j]);
                end
            end

            for (int i=0; i<IN_DEPTH; i++)
            begin
                data_minus_mean_div_by_std_b[i] = data_minus_mean_r11[i]/(standard_deviation_r[int'(i/NORMALIZATION_ZONE_PERIOD)] + EPSILON);     
            end

            for (int i=0; i<IN_DEPTH; i++)
            begin
                data_minus_mean_div_by_std_times_gamma_b[i] = data_minus_mean_div_by_std_r[i]*gamma_r[i];
            end

            for (int i=0; i<IN_DEPTH; i++)
            begin
                normalised_data_b[i] = data_minus_mean_div_by_std_times_gamma_r[i] + beta_r[i];
            end
        


        //TODO: remove the state machine below - left for debugging only

        if (state_r == MEAN_SUM_STATE)
        begin
            state_b = MEAN_DIV_STATE;
        end
        else if (state_r == MEAN_DIV_STATE)
        begin 
            state_b = SUB_STATE;
        end
        else if (state_r == SUB_STATE)
        begin
            state_b = SQUARING_STATE;
        end
        else if (state_r == SQUARING_STATE)
        begin
            state_b = SUM_SQU_STATE; 
        end
        else if (state_r == SUM_SQU_STATE)
        begin
            state_b = VAR_DIV_STATE;
        end
        else if (state_r == VAR_DIV_STATE)
        begin
            state_b = NORM_DIFF_STATE;
        end
        else if (state_r == NORM_DIFF_STATE)
        begin 
            if (&sqrt_v_out_valid)
            begin
                state_b = NORM_DIV_STATE;
            end
            else
            begin 
                state_b = WAITING_FOR_SQRT;
            end
        end
        else if (state_r == WAITING_FOR_SQRT)
        begin
             if (&sqrt_v_out_valid)
            begin
                state_b = NORM_DIV_STATE;
            end
            else
            begin 
                state_b = WAITING_FOR_SQRT;
            end
        end
        else if (state_r == NORM_DIV_STATE)
        begin 
            
            state_b = NORM_MULT_STATE;
        end
        else if (state_r == NORM_MULT_STATE)
        begin 
            
            state_b = NORM_ADD_STATE;
        end
        else if (state_r == NORM_ADD_STATE)
        begin 
            
            state_b = DONE;  
            
        end
        else if (state_r == DONE)
        begin
            state_b = READY_STATE;

        end
    end

    genvar j;
    generate
        for (j=0; j < NUM_NORMALIZATION_ZONES; j++)
        begin : a_sqrt_module 
            sqrt #(
                .IN_WIDTH(IN_WIDTH),
                .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
                .NUM_ITERATION(10)
            ) sqrt_cordic (
                .clk(clk),
                .rst(rst),
                .v_in(variance_in_width[j]),
                .v_in_valid(valid_in_sqrt), //TODO: set meaningful value
                .v_in_ready(sqrt_v_in_ready),

                .v_out(sqrt_out[j*NORMALIZATION_ZONE_PERIOD]),
                .v_out_valid(sqrt_v_out_valid[j]),
                .v_out_ready('1) //TODO: assign this and check in module
            );

        end
    endgenerate
    


    // Data outputs.
    assign data_in_0_ready     = 1'b1;

    assign data_out_0_valid     = sqrt_valid_out_r5;
    assign data_out_0 = normalised_data_r;

  


    always_ff @(posedge clk) //TODO: add asynchronous reset behaviour
    begin
        state_r                                     <= state_b;
        data_r                                      <= data_b;
        sqrt_valid_out_r                            <= sqrt_valid_out_b;
        sqrt_valid_out_r2                           <= sqrt_valid_out_r;
        sqrt_valid_out_r3                           <= sqrt_valid_out_r2;
        sqrt_valid_out_r4                           <= sqrt_valid_out_r3;
        sqrt_valid_out_r5                           <= sqrt_valid_out_r4;
        
        // valid_in_sqrt_r                             <= valid_in_sqrt_b;
        beta_r                                      <= beta_b;
        gamma_r                                     <= gamma_b;
        normalised_data_r                           <= normalised_data_b;
        sum_r                                       <= sum_b; 
        mean_r                                      <= mean_b;
        mean_r2                                     <= mean_r;
        mean_r3                                     <= mean_r2;
        mean_r4                                     <= mean_r3;
        mean_r5                                     <= mean_r4;
        data_in_minus_mean_r                        <= data_in_minus_mean_b;
        data_in_minus_mean_squared_r                <= data_in_minus_mean_squared_b;
        sum_of_squared_differences_r                <= sum_of_squared_differences_b;
        variance_padded_r                           <= variance_padded_b;
        data_minus_mean_r                           <= data_minus_mean_b; 
        data_minus_mean_r2                          <= data_minus_mean_r; 
        data_minus_mean_r3                          <= data_minus_mean_r2; 
        data_minus_mean_r4                          <= data_minus_mean_r3; 
        data_minus_mean_r5                          <= data_minus_mean_r4; 
        data_minus_mean_r6                          <= data_minus_mean_r5; 
        data_minus_mean_r7                          <= data_minus_mean_r6; 
        data_minus_mean_r8                          <= data_minus_mean_r7; 
        data_minus_mean_r9                          <= data_minus_mean_r8; 
        data_minus_mean_r10                         <= data_minus_mean_r9; 
        data_minus_mean_r11                         <= data_minus_mean_r10; 
        standard_deviation_r                        <= standard_deviation_b;
        data_minus_mean_div_by_std_r                <= data_minus_mean_div_by_std_b;   
        data_minus_mean_div_by_std_times_gamma_r    <= data_minus_mean_div_by_std_times_gamma_b;
        data_in_valid_delay_line_r                  <= data_in_valid_delay_line_b;
        data_r_delay_line_r                         <= data_r_delay_line_b;
        
    end

  
    
endmodule
