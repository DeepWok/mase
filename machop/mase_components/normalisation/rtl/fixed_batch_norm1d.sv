`timescale 1ns / 1ps
module fixed_batch_norm1d #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 16,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    // The different inputs may have different levels of precision:
    // we use an internal FP format large enough to store all.
    // However, the tensor sizes must of course equal
    // those of data in.
    parameter MEAN_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter MEAN_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter MEAN_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter MEAN_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,

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
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1
) (
    input                   clk, 
    input                   rst, 
    
    input  [DATA_IN_0_PRECISION_0-1:0] data_in_0        [DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input                              data_in_0_valid,
    output                             data_in_0_ready,

    // input ports for gamma divided by the standard deviation
    input  [WEIGHT_PRECISION_0-1:0] weight      [WEIGHT_PARALLELISM_DIM_0-1:0],
    input                           weight_valid,
    output                          weight_ready,

    // input ports bias/beta
    input [BIAS_PRECISION_0-1:0] bias [BIAS_PARALLELISM_DIM_0-1:0],
    input                        bias_valid,
    output                       bias_ready,

    input [MEAN_PRECISION_0-1:0] mean [MEAN_PARALLELISM_DIM_0-1:0],
    input                        mean_valid,
    output                       mean_ready,

    // Output ports for data
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0      [DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output data_out_0_valid,
    input data_out_0_ready

);
    // Rename parameters to more descriptive names.
    localparam IN_WIDTH      = DATA_IN_0_PRECISION_0; 
    localparam IN_FRAC_WIDTH = DATA_IN_0_PRECISION_1; 
    localparam IN_DEPTH      = DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1; 

    let max2(v1, v2) = (v1 > v2) ? v1 : v2;
    
    // Intermediate FP format for the result of the data - mean subtraction
    parameter FP_SUB_FRAC_WIDTH = max2(DATA_IN_0_PRECISION_1, MEAN_PRECISION_1);
    parameter FP_SUB_WIDTH = max2(DATA_IN_0_PRECISION_0 - DATA_IN_0_PRECISION_1, MEAN_PRECISION_0 - MEAN_PRECISION_1) + FP_SUB_FRAC_WIDTH;
    logic signed [FP_SUB_WIDTH-1:0] data_sub_format         [DATA_IN_0_PARALLELISM_DIM_0-1:0];
    logic signed [FP_SUB_WIDTH-1:0] mean_sub_format         [DATA_IN_0_PARALLELISM_DIM_0-1:0];
    logic signed [FP_SUB_WIDTH-1:0] sub_res                 [DATA_IN_0_PARALLELISM_DIM_0-1:0];
    fixed_cast #(
        .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0),
        .IN_WIDTH(DATA_IN_0_PRECISION_0),
        .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
        .OUT_WIDTH(FP_SUB_WIDTH),
        .OUT_FRAC_WIDTH(FP_SUB_FRAC_WIDTH)
    ) cast_data_to_sub (
        .data_in(data_in_0),
        .data_out(data_sub_format)
    );
    fixed_cast #(
        .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0),
        .IN_WIDTH(MEAN_PRECISION_0),
        .IN_FRAC_WIDTH(MEAN_PRECISION_1),
        .OUT_WIDTH(FP_SUB_WIDTH),
        .OUT_FRAC_WIDTH(FP_SUB_FRAC_WIDTH)
    ) cast_mean_to_sub (
        .data_in(mean),
        .data_out(mean_sub_format)
    );

    // Intermediate FP format for result (sumres) * weight multiplication
    localparam FP_MULT_FRAC_WIDTH = FP_SUB_FRAC_WIDTH + WEIGHT_PRECISION_1;
    localparam FP_MULT_WIDTH = FP_SUB_WIDTH + WEIGHT_PRECISION_0;
    logic signed [FP_MULT_WIDTH-1:0] mult_res               [DATA_IN_0_PARALLELISM_DIM_0-1:0];

    // Intermediate FP format for the result of the bias subtraction
    parameter FP_FINAL_FRAC_WIDTH = max2(FP_MULT_FRAC_WIDTH, BIAS_PRECISION_1);
    parameter FP_FINAL_WIDTH = max2(FP_MULT_WIDTH - FP_MULT_FRAC_WIDTH, BIAS_PRECISION_0 - BIAS_PRECISION_1) + FP_FINAL_FRAC_WIDTH;
    logic signed [FP_FINAL_WIDTH-1:0] mult_res_final_format        [DATA_IN_0_PARALLELISM_DIM_0-1:0];
    logic signed [FP_FINAL_WIDTH-1:0] bias_final_format            [DATA_IN_0_PARALLELISM_DIM_0-1:0];
    logic signed [FP_FINAL_WIDTH-1:0] final_res                    [DATA_IN_0_PARALLELISM_DIM_0-1:0];
    fixed_cast #(
        .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0),
        .IN_WIDTH(FP_MULT_WIDTH),
        .IN_FRAC_WIDTH(FP_MULT_FRAC_WIDTH),
        .OUT_WIDTH(FP_FINAL_WIDTH),
        .OUT_FRAC_WIDTH(FP_FINAL_FRAC_WIDTH)
    ) cast_mult_res_to_final (
        .data_in(mult_res),
        .data_out(mult_res_final_format)
    );
    fixed_cast #(
        .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_0),
        .IN_WIDTH(BIAS_PRECISION_0),
        .IN_FRAC_WIDTH(BIAS_PRECISION_1),
        .OUT_WIDTH(FP_FINAL_WIDTH),
        .OUT_FRAC_WIDTH(FP_FINAL_FRAC_WIDTH)
    ) cast_bias_to_final (
        .data_in(bias),
        .data_out(bias_final_format)
    );

    always_comb begin
        for (int i = 0; i < IN_DEPTH; i++)
        begin
            data_out_0[i] = (final_res[i] >>> (FP_FINAL_FRAC_WIDTH - DATA_OUT_0_PRECISION_1));
        end
    end
    
    assign data_in_0_ready     = 1'b1;
    // Delay line for valid out.
    localparam NUM_STAGES = 3;
    typedef enum int { SUB_STAGE, MULT_STAGE, FINAL_STAGE} NORM_STAGES; 
    logic stage_valid_b [NUM_STAGES-1:0];
    logic stage_valid_r [NUM_STAGES-1:0];
    
    always_comb
    begin
        stage_valid_b[SUB_STAGE] = data_in_0_valid && mean_valid;
        stage_valid_b[MULT_STAGE] = stage_valid_r[MULT_STAGE-1] && weight_valid;
        stage_valid_b[FINAL_STAGE] = stage_valid_r[FINAL_STAGE-1] && bias_valid;
        data_out_0_valid = stage_valid_r[FINAL_STAGE];
    end

    // Stage FSM
    always_ff @(posedge clk)
    begin
        for (int s = 0; s < NUM_STAGES; s++)
        begin
            stage_valid_r[s] = stage_valid_b[s];
        end
    end

    always_ff @(posedge clk)
    begin
        for (int i = 0; i < IN_DEPTH; i++) 
        begin
            sub_res[i] <= sub_res[i];
            mult_res[i] <= mult_res[i];
            final_res[i] <= final_res[i];
            data_out_0[i] <= data_out_0[i];

            if (stage_valid_b[SUB_STAGE])
                sub_res[i] <= data_sub_format[i] - mean_sub_format[i];

            if (stage_valid_b[MULT_STAGE])
                mult_res[i] <= sub_res[i] * weight[i];
            
            if (stage_valid_b[FINAL_STAGE])
                final_res[i] <= mult_res_final_format[i] + bias_final_format[i];
        end
        // valid_out_r     <= valid_out_b;
        //delay line as expect the TB requires small delay (+ real designs will have it so best to check)
    end

    
endmodule
