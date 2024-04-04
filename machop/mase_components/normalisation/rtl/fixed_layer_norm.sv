`timescale 1ns / 1ps
module fixed_layer_norm #(


    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 16,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 16,
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,  //sv720: needed by tb
    /* verilator lint_on UNUSEDPARAM */

    parameter WEIGHT_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter WEIGHT_PRECISION_1 = DATA_IN_0_PRECISION_1,
    /* verilator lint_off UNUSEDPARAM */
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    /* verilator lint_on UNUSEDPARAM */
    parameter WEIGHT_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,

    parameter BIAS_PRECISION_0 = DATA_IN_0_PRECISION_0,
    /* verilator lint_off UNUSEDPARAM */
    parameter BIAS_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter BIAS_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    /* verilator lint_on UNUSEDPARAM */
    parameter BIAS_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,

    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    /* verilator lint_on UNUSEDPARAM */
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,

    // ------ We need to use the above parameters a lot, rename some of the most used ----------
    parameter IN_WIDTH      = DATA_IN_0_PRECISION_0,
    parameter IN_FRAC_WIDTH = DATA_IN_0_PRECISION_1,

    // IN_DEPTH describes the number of data points per sample.
    // In image contexts, IN_DEPTH = sum_product of C, H & W.
    parameter IN_DEPTH                = DATA_IN_0_PARALLELISM_DIM_0,
    parameter NUM_NORMALIZATION_ZONES = 1
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
) (
    input clk,
    input rst,

    // Input ports for data
    input  logic signed [IN_WIDTH-1:0] data_in_0      [DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input                              data_in_0_valid,
    output                             data_in_0_ready,

    input  logic signed [BIAS_PRECISION_0-1:0] bias      [BIAS_PARALLELISM_DIM_0-1:0],
    /* verilator lint_off UNUSEDSIGNAL */
    input                                      bias_valid,
    /* verilator lint_on UNUSEDSIGNAL */
    output                                     bias_ready,

    input  logic signed [WEIGHT_PRECISION_0-1:0] weight      [WEIGHT_PARALLELISM_DIM_0-1:0],
    /* verilator lint_off UNUSEDSIGNAL */
    input                                        weight_valid,
    /* verilator lint_on UNUSEDSIGNAL */
    output                                       weight_ready,

    // Output ports for data
    output logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output data_out_0_valid,
    input data_out_0_ready

);

  localparam BLOCK_SIZE = DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1;
  localparam NORMALIZATION_ZONE_PERIOD = BLOCK_SIZE / NUM_NORMALIZATION_ZONES;

  // The max size the sum for calculating the mean is
  // MAX_NUM_ELEMS * MAX_SIZE_PER_ELEM = BLOCK_SIZE * 2^IN_WIDTH
  parameter SUM_MAX_SIZE = BLOCK_SIZE * (2 ** IN_WIDTH);

  // We need larger bitwidth than the inputs.
  parameter SUM_EXTRA_FRAC_WIDTH = $clog2(BLOCK_SIZE);
  parameter SUM_WIDTH = $clog2(SUM_MAX_SIZE) + SUM_EXTRA_FRAC_WIDTH;
  parameter SUM_FRAC_WIDTH = IN_FRAC_WIDTH + SUM_EXTRA_FRAC_WIDTH;

  parameter SUM_SQUARED_BITS = $clog2(SUM_MAX_SIZE ** 2) + SUM_EXTRA_FRAC_WIDTH * 2;
  parameter SUM_SQUARED_FRAC_WIDTH = 2 * (IN_FRAC_WIDTH + SUM_EXTRA_FRAC_WIDTH);

  parameter SUM_OF_SQUARES_BITS = SUM_SQUARED_BITS + $clog2(IN_DEPTH);

  parameter SUM_OF_SQUARES_BITS_PADDED = SUM_OF_SQUARES_BITS + $clog2(IN_DEPTH);
  parameter VAR_BITS_PADDED = SUM_OF_SQUARES_BITS_PADDED;
  parameter VAR_BITS = SUM_OF_SQUARES_BITS;
  parameter VAR_FRAC_WIDTH = SUM_SQUARED_FRAC_WIDTH + $clog2(
      IN_DEPTH
  );  //sv720: division by depth -> less integer bits

  // Before performing the devision by the stdv, extend the
  // fractional part of the divident. Dividing will remove the number of
  // divisor frac bits from the dividend - extending prevents all of
  // our fractional data getting lost. 
  localparam PRE_DIV_WIDTH = SUM_WIDTH + SUM_FRAC_WIDTH;
  localparam PRE_DIV_FRAC_WIDTH = SUM_FRAC_WIDTH * 2;

  // Post-division FP format is same as the divior except
  // with dividend fractional bits removed.
  localparam DIV_WIDTH = PRE_DIV_WIDTH;
  localparam DIV_FRAC_WIDTH = PRE_DIV_FRAC_WIDTH - SUM_FRAC_WIDTH;

  // We multiply by gamma/weight, which adds adds WEIGHT_PRECISION_0
  // more bits to the representation. However, we only add the fractional bits
  // since the non-fractional bits won't overflow from this operation. 
  localparam FINAL_WIDTH = DIV_WIDTH;
  localparam FINAL_FRAC_WIDTH = DIV_FRAC_WIDTH + WEIGHT_PRECISION_1;

  parameter EPSILON = 0;

  parameter NUM_STATE_BITS = 32;
  parameter VALID_IN_DELAY_DELAY_LINE_SIZE = 6;
  parameter DATA_DELAY_LINE_SIZE = 12;

  typedef enum logic [NUM_STATE_BITS-1:0] {
    RST_STATE        = '0,
    MEAN_SUM_STATE   = 1,
    MEAN_DIV_STATE   = 2,
    SUB_STATE        = 3,
    SQUARING_STATE   = 4,
    SUM_SQU_STATE    = 5,
    VAR_DIV_STATE    = 6,   //if right shift -> drop this
    NORM_DIFF_STATE  = 7,
    WAITING_FOR_SQRT = 8,
    NORM_DIV_STATE   = 9,
    NORM_MULT_STATE  = 10,
    NORM_ADD_STATE   = 11,

    READY_STATE = 12,
    UNASSIGNED  = 13,
    DONE        = '1
  } state_t;

  state_t state_b;
  state_t state_r;
  // logic rst = ~reset_n;

  logic [IN_WIDTH-1:0] data_b[BLOCK_SIZE-1:0];
  logic [IN_WIDTH-1:0] data_r[BLOCK_SIZE-1:0];
  logic [IN_WIDTH-1:0] data_r3[BLOCK_SIZE-1:0];
  logic [IN_WIDTH-1:0] data_r7[BLOCK_SIZE-1:0];

  logic [SUM_WIDTH - 1:0] data_r_sum_format[BLOCK_SIZE-1:0];
  logic [SUM_WIDTH - 1:0] data_r3_sum_format[BLOCK_SIZE-1:0];
  logic [SUM_WIDTH- 1:0] data_r7_sum_format[BLOCK_SIZE-1:0];

  fixed_cast #(
      .IN_SIZE(BLOCK_SIZE),
      .IN_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
      .OUT_WIDTH(SUM_WIDTH),
      .OUT_FRAC_WIDTH(SUM_FRAC_WIDTH)
  ) cast_data_r_to_sum (
      .data_in (data_r),
      .data_out(data_r_sum_format)
  );

  fixed_cast #(
      .IN_SIZE(BLOCK_SIZE),
      .IN_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
      .OUT_WIDTH(SUM_WIDTH),
      .OUT_FRAC_WIDTH(SUM_FRAC_WIDTH)
  ) cast_data_r3_to_sum (
      .data_in (data_r3),
      .data_out(data_r3_sum_format)
  );

  fixed_cast #(
      .IN_SIZE(BLOCK_SIZE),
      .IN_WIDTH(IN_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .OUT_WIDTH(SUM_WIDTH),
      .OUT_FRAC_WIDTH(SUM_FRAC_WIDTH)
  ) cast_data_r7_to_sum (
      .data_in (data_r7),
      .data_out(data_r7_sum_format)
  );

  logic signed [SUM_WIDTH-1:0] data_minus_mean_r11[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r10[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r9[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r8[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r7[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r6[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r5[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r4[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r3[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r2[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_r[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH-1:0] data_minus_mean_b[BLOCK_SIZE-1:0];

  logic signed [PRE_DIV_WIDTH-1:0] data_minus_mean_r11_pre_div[BLOCK_SIZE-1:0];

  fixed_cast #(
      .IN_SIZE(BLOCK_SIZE),
      .IN_WIDTH(SUM_WIDTH),
      .IN_FRAC_WIDTH(SUM_FRAC_WIDTH),
      .OUT_WIDTH(PRE_DIV_WIDTH),
      .OUT_FRAC_WIDTH(PRE_DIV_FRAC_WIDTH)
  ) cast_data_minus_mean_r11_to_pre_div (
      .data_in (data_minus_mean_r11),
      .data_out(data_minus_mean_r11_pre_div)
  );


  logic signed [IN_WIDTH-1:0] beta_b[BLOCK_SIZE-1:0];
  logic signed [IN_WIDTH-1:0] beta_r[BLOCK_SIZE-1:0];
  logic signed [FINAL_WIDTH-1:0] beta_r_fin_format[BLOCK_SIZE-1:0];
  fixed_cast #(
      .IN_SIZE(BLOCK_SIZE),
      .IN_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
      .OUT_WIDTH(FINAL_WIDTH),
      .OUT_FRAC_WIDTH(FINAL_FRAC_WIDTH)
  ) cast_beta_to_sum (
      .data_in (beta_r),
      .data_out(beta_r_fin_format)
  );

  logic signed [IN_WIDTH-1:0] gamma_b[BLOCK_SIZE-1:0];
  logic signed [IN_WIDTH-1:0] gamma_r[BLOCK_SIZE-1:0];
  // logic signed [DIV_WIDTH-1:0] gamma_r_div_format[BLOCK_SIZE-1:0];
  // fixed_cast #(
  //     .IN_SIZE(BLOCK_SIZE),
  //     .IN_WIDTH(DATA_IN_0_PRECISION_0),
  //     .IN_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
  //     .OUT_WIDTH(DIV_WIDTH),
  //     .OUT_FRAC_WIDTH(DIV_FRAC_WIDTH)
  // ) cast_gamma_to_sum (
  //     .data_in (gamma_r),
  //     .data_out(gamma_r_div_format)
  // );

  logic signed [SUM_WIDTH - 1:0] sum_b[NUM_NORMALIZATION_ZONES-1:0];
  logic signed [SUM_WIDTH - 1:0] sum_r[NUM_NORMALIZATION_ZONES-1:0];

  logic signed [SUM_WIDTH - 1:0] mean_b[NUM_NORMALIZATION_ZONES-1:0];
  logic signed [SUM_WIDTH - 1:0] mean_r[NUM_NORMALIZATION_ZONES-1:0];
  logic signed [SUM_WIDTH - 1:0] mean_r2[NUM_NORMALIZATION_ZONES-1:0];
  logic signed [SUM_WIDTH - 1:0] mean_r3[NUM_NORMALIZATION_ZONES-1:0];
  logic signed [SUM_WIDTH - 1:0] mean_r4[NUM_NORMALIZATION_ZONES-1:0];
  logic signed [SUM_WIDTH - 1:0] mean_r5[NUM_NORMALIZATION_ZONES-1:0];

  logic signed [SUM_WIDTH - 1:0] data_in_minus_mean_b[BLOCK_SIZE-1:0];
  logic signed [SUM_WIDTH - 1:0] data_in_minus_mean_r[BLOCK_SIZE-1:0];
  logic [SUM_SQUARED_BITS - 1:0] data_in_minus_mean_squared_b[BLOCK_SIZE-1:0];
  logic [SUM_SQUARED_BITS - 1:0] data_in_minus_mean_squared_r[BLOCK_SIZE-1:0];

  logic [SUM_OF_SQUARES_BITS - 1:0] sum_of_squared_differences_b[NUM_NORMALIZATION_ZONES-1:0];
  logic [SUM_OF_SQUARES_BITS - 1:0] sum_of_squared_differences_r[NUM_NORMALIZATION_ZONES-1:0];

  logic [SUM_OF_SQUARES_BITS - 1:0] sum_of_squared_differences_tmp[NUM_NORMALIZATION_ZONES-1:0];
  logic [SUM_OF_SQUARES_BITS_PADDED - 1:0]  sum_of_squared_differences_padded   [NUM_NORMALIZATION_ZONES-1:0];
  logic [VAR_BITS - 1:0] variance[NUM_NORMALIZATION_ZONES-1:0];
  /* verilator lint_off UNOPTFLAT */ //sv720: don't understand why event model (not synthesised) circular logic
  logic [VAR_BITS_PADDED - 1:0] variance_padded[NUM_NORMALIZATION_ZONES-1:0];
  /* verilator lint_on UNOPTFLAT */
  logic signed [IN_WIDTH - 1:0] variance_in_width[NUM_NORMALIZATION_ZONES-1:0];
  logic [IN_WIDTH - 1:0] sqrt_out[NUM_NORMALIZATION_ZONES-1:0];
  logic [SUM_WIDTH - 1:0] sqrt_out_sum_format[NUM_NORMALIZATION_ZONES-1:0];
  logic signed [SUM_WIDTH - 1:0] standard_deviation_b[NUM_NORMALIZATION_ZONES-1:0];
  logic signed [SUM_WIDTH - 1:0] standard_deviation_r[NUM_NORMALIZATION_ZONES-1:0];

  logic signed [DIV_WIDTH-1:0] data_minus_mean_div_by_std_b[BLOCK_SIZE-1:0];
  logic signed [DIV_WIDTH-1:0] data_minus_mean_div_by_std_r[BLOCK_SIZE-1:0];

  logic signed [DIV_WIDTH-1:0] data_minus_mean_div_by_std_times_gamma_b[BLOCK_SIZE-1:0];
  logic signed [DIV_WIDTH-1:0] data_minus_mean_div_by_std_times_gamma_r[BLOCK_SIZE-1:0];


  logic signed [DIV_WIDTH-1:0] normalised_data_b[BLOCK_SIZE-1:0];
  logic signed [DIV_WIDTH-1:0] normalised_data_r[BLOCK_SIZE-1:0];

  fixed_cast #(
      .IN_SIZE(BLOCK_SIZE),
      .IN_WIDTH(FINAL_WIDTH),
      .IN_FRAC_WIDTH(FINAL_FRAC_WIDTH),
      .OUT_WIDTH(IN_WIDTH),
      .OUT_FRAC_WIDTH(IN_FRAC_WIDTH)
  ) cast_norm_data_to_out (
      .data_in (normalised_data_r),
      .data_out(data_out_0)
  );

  /* verilator lint_off UNUSEDSIGNAL */
  logic sqrt_v_in_ready;  //TODO: use this
  /* verilator lint_on UNUSEDSIGNAL */
  logic [NUM_NORMALIZATION_ZONES-1:0] sqrt_v_out_valid;  //TODO: use this

  logic sqrt_valid_out_b;
  logic sqrt_valid_out_r;
  logic sqrt_valid_out_r2;
  logic sqrt_valid_out_r3;
  logic sqrt_valid_out_r4;
  logic sqrt_valid_out_r5;

  // logic valid_in_sqrt_b;
  // logic valid_in_sqrt_r;
  logic valid_in_sqrt;

  logic [VALID_IN_DELAY_DELAY_LINE_SIZE - 1:0] data_in_valid_delay_line_b;
  logic [VALID_IN_DELAY_DELAY_LINE_SIZE - 1:0] data_in_valid_delay_line_r;

  logic [IN_WIDTH-1:0] data_r_delay_line_b[DATA_DELAY_LINE_SIZE-1:0][IN_DEPTH-1:0];
  logic [IN_WIDTH-1:0] data_r_delay_line_r[DATA_DELAY_LINE_SIZE-1:0][IN_DEPTH-1:0];

  assign valid_in_sqrt = data_in_valid_delay_line_r[VALID_IN_DELAY_DELAY_LINE_SIZE-1];


  assign data_r7 = data_r_delay_line_r[6][BLOCK_SIZE-1:0];
  assign data_r3 = data_r_delay_line_r[2][BLOCK_SIZE-1:0];


  always_comb begin

    // Convert the sqrt output to our internal FP representation.
    // Due to verilator bugs, this can not be done with the fixed_cast module, unfortunately.
    // if (NUM_NORMALIZATION_ZONES == 1) begin
    //   sqrt_out_sum_format[SUM_WIDTH-1:0] = 0;
    //   sqrt_out_sum_format[SUM_WIDTH-1:SUM_WIDTH-IN_WIDTH] = sqrt_out;
    //   sqrt_out_sum_format = sqrt_out_sum_format >>> (SUM_WIDTH - IN_WIDTH - SUM_EXTRA_FRAC_WIDTH);
    // end 
    // else begin
    //   for (int i = 0; i < NUM_NORMALIZATION_ZONES; i++) begin
    //     sqrt_out_sum_format[i][SUM_WIDTH-1:0] = 0;
    //     sqrt_out_sum_format[i][SUM_WIDTH-1:SUM_WIDTH-IN_WIDTH] = sqrt_out[i];
    //     sqrt_out_sum_format[i] = sqrt_out_sum_format[i] >>> (SUM_WIDTH - IN_WIDTH - SUM_EXTRA_FRAC_WIDTH);
    //   end
    // end

    for (int i = 0; i < NUM_NORMALIZATION_ZONES; i++) begin
      sqrt_out_sum_format[i][SUM_WIDTH-1:0] = 0;
      sqrt_out_sum_format[i][SUM_WIDTH-1:SUM_WIDTH-IN_WIDTH] = sqrt_out[i];
      sqrt_out_sum_format[i] = sqrt_out_sum_format[i] >>> (SUM_WIDTH - IN_WIDTH - SUM_EXTRA_FRAC_WIDTH);
    end
    for (int i = 0; i < BLOCK_SIZE; i++) begin
      data_in_valid_delay_line_b[i] = '0;
    end


    state_b                      = state_r;

    // valid_in_sqrt_b     = '0; 
    sqrt_valid_out_b             = '0;

    normalised_data_b            = normalised_data_r;

    data_b                       = data_r;
    beta_b                       = beta_r;
    gamma_b                      = gamma_r;
    sum_b                        = sum_r;
    mean_b                       = mean_r;



    sum_of_squared_differences_b = sum_of_squared_differences_r;

    for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++) begin
      sum_of_squared_differences_tmp[j] = '0;
    end

    data_minus_mean_b = data_minus_mean_r;

    standard_deviation_b = standard_deviation_r;
    data_minus_mean_div_by_std_b = data_minus_mean_div_by_std_r;

    data_minus_mean_div_by_std_times_gamma_b = data_minus_mean_div_by_std_times_gamma_r;

    if (data_in_0_valid) begin
      data_in_valid_delay_line_b[0] = 1'b1;
      state_b   = MEAN_SUM_STATE;
      data_b    = data_in_0;
      beta_b    = bias;
      gamma_b   = weight;
    end else begin
      data_in_valid_delay_line_b = data_in_valid_delay_line_r << 1;
    end

    data_r_delay_line_b[0] = data_b;

    for (int i = 1; i < DATA_DELAY_LINE_SIZE; i++) begin
      data_r_delay_line_b[i] = data_r_delay_line_r[i-1];
    end

    for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++) begin
      sum_of_squared_differences_padded[j] = (SUM_OF_SQUARES_BITS_PADDED'(sum_of_squared_differences_r[j]) << $clog2(
          BLOCK_SIZE));
      variance_padded[j] = sum_of_squared_differences_padded[j] / NORMALIZATION_ZONE_PERIOD;
      variance[j] = variance_padded[j][VAR_BITS-1:0];
      variance_in_width[j] = variance[j][ IN_WIDTH + VAR_FRAC_WIDTH - IN_FRAC_WIDTH -1 : VAR_FRAC_WIDTH - IN_FRAC_WIDTH ];

      if (&sqrt_v_out_valid) begin
        standard_deviation_b = sqrt_out_sum_format;
        sqrt_valid_out_b     = '1;
      end
    end

    for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++) begin
      // Sum over the widened inputs.
      sum_b[j] = 0;
      for (int i = 0; i < NORMALIZATION_ZONE_PERIOD; i++) begin
        sum_b[j] += data_r_sum_format[i+j*NORMALIZATION_ZONE_PERIOD];
      end

      mean_b[j] = sum_r[j] / NORMALIZATION_ZONE_PERIOD;

      for (int i = 0; i < NORMALIZATION_ZONE_PERIOD; i++) begin
        data_in_minus_mean_b[i+j*NORMALIZATION_ZONE_PERIOD] = data_r3_sum_format[i+ j*NORMALIZATION_ZONE_PERIOD] - mean_r[j];
      end
    end

    for (int i = 0; i < BLOCK_SIZE; i++) begin
      data_in_minus_mean_squared_b[i] = SUM_SQUARED_BITS'(data_in_minus_mean_r[i]) ** 2;
    end

    for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++) begin
      sum_of_squared_differences_tmp[j] = '0;

      for (int i = 0; i < NORMALIZATION_ZONE_PERIOD; i++) begin
        sum_of_squared_differences_tmp[j] +=   SUM_OF_SQUARES_BITS'(data_in_minus_mean_squared_r[i+j*NORMALIZATION_ZONE_PERIOD]);
      end
    end

    sum_of_squared_differences_b = sum_of_squared_differences_tmp;


    for (int j = 0; j < NUM_NORMALIZATION_ZONES; j++) begin
      for (int i = 0; i < NORMALIZATION_ZONE_PERIOD; i++) begin
        data_minus_mean_b[i+j*NORMALIZATION_ZONE_PERIOD] = (data_r7_sum_format[i+j*NORMALIZATION_ZONE_PERIOD] - mean_r5[j]);
        // data_minus_mean_b[i+j*NORMALIZATION_ZONE_PERIOD] = (data_r[i+j*NORMALIZATION_ZONE_PERIOD] - mean_r[j]);
      end
    end

    for (int i = 0; i < BLOCK_SIZE; i++) begin
      data_minus_mean_div_by_std_b[i] = data_minus_mean_r11_pre_div[i]/(standard_deviation_r[int'(i/NORMALIZATION_ZONE_PERIOD)] + EPSILON);
    end

    for (int i = 0; i < BLOCK_SIZE; i++) begin
      data_minus_mean_div_by_std_times_gamma_b[i] = data_minus_mean_div_by_std_r[i] * gamma_r[i];
    end

    for (int i = 0; i < BLOCK_SIZE; i++) begin
      normalised_data_b[i] = data_minus_mean_div_by_std_times_gamma_r[i] + beta_r_fin_format[i];
    end



    //TODO: remove the state machine below - left for debugging only

    if (state_r == MEAN_SUM_STATE) begin
      state_b = MEAN_DIV_STATE;
    end else if (state_r == MEAN_DIV_STATE) begin
      state_b = SUB_STATE;
    end else if (state_r == SUB_STATE) begin
      state_b = SQUARING_STATE;
    end else if (state_r == SQUARING_STATE) begin
      state_b = SUM_SQU_STATE;
    end else if (state_r == SUM_SQU_STATE) begin
      state_b = VAR_DIV_STATE;
    end else if (state_r == VAR_DIV_STATE) begin
      state_b = NORM_DIFF_STATE;
    end else if (state_r == NORM_DIFF_STATE) begin
      if (&sqrt_v_out_valid) begin
        state_b = NORM_DIV_STATE;
      end else begin
        state_b = WAITING_FOR_SQRT;
      end
    end else if (state_r == WAITING_FOR_SQRT) begin
      if (&sqrt_v_out_valid) begin
        state_b = NORM_DIV_STATE;
      end else begin
        state_b = WAITING_FOR_SQRT;
      end
    end else if (state_r == NORM_DIV_STATE) begin

      state_b = NORM_MULT_STATE;
    end else if (state_r == NORM_MULT_STATE) begin

      state_b = NORM_ADD_STATE;
    end else if (state_r == NORM_ADD_STATE) begin

      state_b = DONE;

    end else if (state_r == DONE) begin
      state_b = READY_STATE;

    end
  end

  genvar j;
  generate
    for (j = 0; j < NUM_NORMALIZATION_ZONES; j++) begin : a_sqrt_module
      sqrt #(
          .IN_WIDTH(IN_WIDTH),
          .NUM_ITERATION(10)
      ) sqrt_cordic (
          .clk(clk),
          .rst(rst),
          .v_in(variance_in_width[j]),
          .v_in_valid(valid_in_sqrt),  //TODO: set meaningful value
          .v_in_ready(sqrt_v_in_ready),

          .v_out(sqrt_out[j]),
          .v_out_valid(sqrt_v_out_valid[j]),
          .v_out_ready('1)  //TODO: assign this and check in module
      );

    end
  endgenerate



  // Data outputs.
  assign data_in_0_ready  = 1'b1;
  assign bias_ready       = 1'b1;
  assign weight_ready     = 1'b1;


  assign data_out_0_valid = sqrt_valid_out_r5 && data_out_0_ready;




  always_ff @(posedge clk) //TODO: add asynchronous reset behaviour
    begin
    state_r                                  <= state_b;
    data_r                                   <= data_b;
    sqrt_valid_out_r                         <= sqrt_valid_out_b;
    sqrt_valid_out_r2                        <= sqrt_valid_out_r;
    sqrt_valid_out_r3                        <= sqrt_valid_out_r2;
    sqrt_valid_out_r4                        <= sqrt_valid_out_r3;
    sqrt_valid_out_r5                        <= sqrt_valid_out_r4;

    // valid_in_sqrt_r                             <= valid_in_sqrt_b;
    beta_r                                   <= beta_b;
    gamma_r                                  <= gamma_b;
    normalised_data_r                        <= normalised_data_b;
    sum_r                                    <= sum_b;
    mean_r                                   <= mean_b;
    mean_r2                                  <= mean_r;
    mean_r3                                  <= mean_r2;
    mean_r4                                  <= mean_r3;
    mean_r5                                  <= mean_r4;
    data_in_minus_mean_r                     <= data_in_minus_mean_b;
    data_in_minus_mean_squared_r             <= data_in_minus_mean_squared_b;
    sum_of_squared_differences_r             <= sum_of_squared_differences_b;
    data_minus_mean_r                        <= data_minus_mean_b;
    data_minus_mean_r2                       <= data_minus_mean_r;
    data_minus_mean_r3                       <= data_minus_mean_r2;
    data_minus_mean_r4                       <= data_minus_mean_r3;
    data_minus_mean_r5                       <= data_minus_mean_r4;
    data_minus_mean_r6                       <= data_minus_mean_r5;
    data_minus_mean_r7                       <= data_minus_mean_r6;
    data_minus_mean_r8                       <= data_minus_mean_r7;
    data_minus_mean_r9                       <= data_minus_mean_r8;
    data_minus_mean_r10                      <= data_minus_mean_r9;
    data_minus_mean_r11                      <= data_minus_mean_r10;
    standard_deviation_r                     <= standard_deviation_b;
    data_minus_mean_div_by_std_r             <= data_minus_mean_div_by_std_b;
    data_minus_mean_div_by_std_times_gamma_r <= data_minus_mean_div_by_std_times_gamma_b;
    data_in_valid_delay_line_r               <= data_in_valid_delay_line_b;
    data_r_delay_line_r                      <= data_r_delay_line_b;

  end



endmodule
