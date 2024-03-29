`timescale 1ns / 1ps
module fixed_batch_norm1d #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 16,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,

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
    
    input  [DATA_IN_0_PRECISION_0-1:0] data_in_0        [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
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
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0      [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output data_out_0_valid,
    input data_out_0_ready
);
    
    
    localparam IN_BLOCK_SIZE = DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1; 
    localparam BLOCK_SIZE    = DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1; 

    
    // Convert all inputs to the output parallelism. 
    logic [DATA_IN_0_PRECISION_0-1:0] data_in_0_para [BLOCK_SIZE-1:0];
    logic data_in_0_para_valid;
    logic data_in_0_para_ready;
    convert_parallelism #(
        .DATA_WIDTH(DATA_IN_0_PRECISION_0),
        .DATA_IN_PARALLELISM(IN_BLOCK_SIZE),
        .DATA_OUT_PARALLELISM(BLOCK_SIZE)
    ) conv_data_para (
        .clk(clk),
        .rst(rst),
    
        .data_in(data_in_0),
        .data_in_valid(data_in_0_valid),
        .data_in_ready(data_in_0_ready),

        .data_out(data_in_0_para),
        .data_out_valid(data_in_0_para_valid),
        .data_out_ready(data_in_0_para_ready)
    );
    
    logic [WEIGHT_PRECISION_0-1:0] weight_para [BLOCK_SIZE-1:0];
    logic weight_para_valid;
    logic weight_para_ready;
    convert_parallelism #(
        .DATA_WIDTH(WEIGHT_PRECISION_0),
        .DATA_IN_PARALLELISM(WEIGHT_PARALLELISM_DIM_0),
        .DATA_OUT_PARALLELISM(BLOCK_SIZE)
    ) conv_weight_para (
        .clk(clk),
        .rst(rst),

        .data_in(weight),
        .data_in_valid(weight_valid),
        .data_in_ready(weight_ready),

        .data_out(weight_para),
        .data_out_valid(weight_para_valid),
        .data_out_ready(weight_para_ready)
    );
    
    logic [MEAN_PRECISION_0-1:0] mean_para [BLOCK_SIZE-1:0];
    logic mean_para_valid;
    logic mean_para_ready;
    convert_parallelism #(
        .DATA_WIDTH(MEAN_PRECISION_0),
        .DATA_IN_PARALLELISM(MEAN_PARALLELISM_DIM_0),
        .DATA_OUT_PARALLELISM(BLOCK_SIZE)
    ) conv_mean_para (
        .clk(clk),
        .rst(rst),

        .data_in(mean),
        .data_in_valid(mean_valid),
        .data_in_ready(mean_ready),

        .data_out(mean_para),
        .data_out_valid(mean_para_valid),
        .data_out_ready(mean_para_ready)
    );

    logic [BIAS_PRECISION_0-1:0] bias_para [BLOCK_SIZE-1:0];
    logic bias_para_valid;
    logic bias_para_ready;
    convert_parallelism #(
        .DATA_WIDTH(BIAS_PRECISION_0),
        .DATA_IN_PARALLELISM(BIAS_PARALLELISM_DIM_0),
        .DATA_OUT_PARALLELISM(BLOCK_SIZE)
    ) conv_bias_para (
        .clk(clk),
        .rst(rst),
    
        .data_in(bias),
        .data_in_valid(bias_valid),
        .data_in_ready(bias_ready),

        .data_out(bias_para),
        .data_out_valid(bias_para_valid),
        .data_out_ready(bias_para_ready)
    );
    
    let max2(v1, v2) = (v1 > v2) ? v1 : v2;
    // Intermediate FP format for the result of the data - mean subtraction
    localparam FP_SUB_FRAC_WIDTH = max2(DATA_IN_0_PRECISION_1, MEAN_PRECISION_1);
    localparam FP_SUB_WIDTH = max2(DATA_IN_0_PRECISION_0 - DATA_IN_0_PRECISION_1, MEAN_PRECISION_0 - MEAN_PRECISION_1) + FP_SUB_FRAC_WIDTH;
    logic signed [FP_SUB_WIDTH-1:0] data_sub_format         [BLOCK_SIZE-1:0];
    logic signed [FP_SUB_WIDTH-1:0] mean_sub_format         [BLOCK_SIZE-1:0];
    logic signed [FP_SUB_WIDTH-1:0] sub_res                 [BLOCK_SIZE-1:0];
    fixed_cast #(
        .IN_SIZE(BLOCK_SIZE),
        .IN_WIDTH(DATA_IN_0_PRECISION_0),
        .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
        .OUT_WIDTH(FP_SUB_WIDTH),
        .OUT_FRAC_WIDTH(FP_SUB_FRAC_WIDTH)
    ) cast_data_to_sub (
        .data_in(data_in_0_para),
        .data_out(data_sub_format)
    );
    fixed_cast #(
        .IN_SIZE(BLOCK_SIZE),
        .IN_WIDTH(MEAN_PRECISION_0),
        .IN_FRAC_WIDTH(MEAN_PRECISION_1),
        .OUT_WIDTH(FP_SUB_WIDTH),
        .OUT_FRAC_WIDTH(FP_SUB_FRAC_WIDTH)
    ) cast_mean_to_sub (
        .data_in(mean_para),
        .data_out(mean_sub_format)
    );

    // Intermediate FP format for result (sumres) * weight multiplication
    localparam FP_MULT_FRAC_WIDTH = FP_SUB_FRAC_WIDTH + WEIGHT_PRECISION_1;
    localparam FP_MULT_WIDTH = FP_SUB_WIDTH + WEIGHT_PRECISION_0;
    logic signed [FP_MULT_WIDTH-1:0] mult_res               [BLOCK_SIZE-1:0];

    // Intermediate FP format for the result of the bias addition
    parameter FP_ADD_FRAC_WIDTH = max2(FP_MULT_FRAC_WIDTH, BIAS_PRECISION_1);
    parameter FP_ADD_WIDTH = max2(FP_MULT_WIDTH - FP_MULT_FRAC_WIDTH, BIAS_PRECISION_0 - BIAS_PRECISION_1) + FP_ADD_FRAC_WIDTH;
    logic signed [FP_ADD_WIDTH-1:0] mult_res_add_format        [BLOCK_SIZE-1:0];
    logic signed [FP_ADD_WIDTH-1:0] bias_add_format            [BLOCK_SIZE-1:0];
    logic signed [FP_ADD_WIDTH-1:0] final_res                  [BLOCK_SIZE-1:0];
    fixed_cast #(
        .IN_SIZE(BLOCK_SIZE),
        .IN_WIDTH(FP_MULT_WIDTH),
        .IN_FRAC_WIDTH(FP_MULT_FRAC_WIDTH),
        .OUT_WIDTH(FP_ADD_WIDTH),
        .OUT_FRAC_WIDTH(FP_ADD_FRAC_WIDTH)
    ) cast_mult_res_to_final (
        .data_in(mult_res),
        .data_out(mult_res_add_format)
    );
    fixed_cast #(
        .IN_SIZE(BLOCK_SIZE),
        .IN_WIDTH(BIAS_PRECISION_0),
        .IN_FRAC_WIDTH(BIAS_PRECISION_1),
        .OUT_WIDTH(FP_ADD_WIDTH),
        .OUT_FRAC_WIDTH(FP_ADD_FRAC_WIDTH)
    ) cast_bias_to_final (
        .data_in(bias_para),
        .data_out(bias_add_format)
    );


    // Format for the output.
    logic signed [DATA_OUT_0_PRECISION_0-1:0] final_res_out_format [BLOCK_SIZE-1:0];
    always_comb begin
        for (int i = 0; i < BLOCK_SIZE; i++)
        begin
            final_res_out_format[i] = (final_res[i] >>> (FP_ADD_FRAC_WIDTH - DATA_OUT_0_PRECISION_1));
        end
    end

    // FP Conversions done, now for the main logic of the module: 
    // Batch norm is calculated in 3 seperate stages, one for each arithmetic operation:
    // 1. Subtracting the mean.
    // 2. Multiplying the weight (which is inlined gamma / stdv)
    // 3. Adding the bias.
    logic sub_join_valid, sub_join_ready, sub_out_valid;
    join2 #() sub_join (
        .data_in_valid ({data_in_0_para_valid, mean_para_valid}),
        .data_in_ready ({data_in_0_para_ready, mean_para_ready}),
        .data_out_valid(sub_join_valid),
        .data_out_ready(sub_join_ready)
    );

    logic mult_join_valid, mult_join_ready, mult_out_valid;
    join2 #() mult_join (
        .data_in_valid ({sub_out_valid, weight_para_valid}),
        .data_in_ready ({sub_join_ready, weight_para_ready}),
        .data_out_valid(mult_join_valid),
        .data_out_ready(mult_join_ready)
    );

    logic add_join_valid, add_out_valid;
    join2 #() add_join (
        .data_in_valid ({mult_out_valid, bias_para_valid}),
        .data_in_ready ({mult_join_ready, bias_para_ready}),
        .data_out_valid(add_join_valid),
        .data_out_ready(&skid_reg_ready)
    );

    always_ff @(posedge clk)
    begin
        for (int i = 0; i < BLOCK_SIZE; i++) 
        begin
            if (rst) begin
                sub_res[i]     <= 0;
                mult_res[i]    <= 0;
                final_res[i]   <= 0;
                sub_out_valid  <= 0;
                mult_out_valid <= 0;
                add_out_valid  <= 0;
            end else begin

                sub_res[i] <= sub_res[i];
                mult_res[i] <= mult_res[i];
                final_res[i] <= final_res[i];

                if (sub_join_valid && sub_join_ready)
                    sub_res[i] <= data_sub_format[i] - mean_sub_format[i];

                if (mult_join_valid && mult_join_ready)
                    mult_res[i] <= sub_res[i] * weight_para[i];
            
                if (add_join_valid && (&skid_reg_ready))
                    final_res[i] <= mult_res_add_format[i] + bias_add_format[i];

                sub_out_valid <= sub_join_valid && sub_join_ready;
                mult_out_valid <= mult_join_valid && mult_join_ready;
                add_out_valid <= add_join_valid && (&skid_reg_ready);
            end
        end
    end

    logic [DATA_OUT_0_PARALLELISM_DIM_0-1:0] skid_reg_ready;
    for (genvar i = 0; i < BLOCK_SIZE; i = i + 1) begin : skid_buf 
        logic dout_valid;
        skid_buffer #(
            .DATA_WIDTH(DATA_OUT_0_PRECISION_0)
        ) skid_buf_out (
            .clk           (clk),
            .rst           (rst),

            .data_in       (final_res_out_format[i]),
            .data_in_valid (add_out_valid),
            .data_in_ready (skid_reg_ready[i]),

            .data_out      (data_out_0[i]),
            .data_out_valid(dout_valid),
            .data_out_ready(data_out_0_ready)
        );
    end
    assign data_out_0_valid = skid_buf[0].dout_valid;

endmodule
