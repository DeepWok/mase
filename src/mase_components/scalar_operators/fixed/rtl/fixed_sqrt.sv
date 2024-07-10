//This is a cordic square-root module that takes inpiration from the localFixedPointCORDICSQRT MATLAB function

`timescale 1ns / 1ps
module fixed_sqrt #(
    parameter IN_WIDTH      = 8,
    parameter NUM_ITERATION = 10

) (
    input                       clk,
    input                       rst,
    input        [IN_WIDTH-1:0] v_in,
    input                       v_in_valid,
    output logic                v_in_ready,
    output logic [IN_WIDTH-1:0] v_out,
    output logic                v_out_valid,
    input                       v_out_ready   //TODO: assign to this
);

  parameter NUM_STATES = NUM_ITERATION + 2;  // a rst and final state
  parameter NUM_STATE_BITS = $clog2(NUM_STATES) + 1;
  parameter K_WORDSIZE = 32;

  parameter VALID_IN_DELAY_LINE_SIZE = NUM_ITERATION;

  // Define an enum for states (one hot)
  typedef enum logic [NUM_STATE_BITS-1:0] {
    RST         = '0,
    //Didn't find a way to change this automatically
    // w/ for loop so just gonna have to change by hand
    STATE_1     = 1,
    STATE_2     = 2,
    STATE_3     = 3,
    STATE_4     = 4,
    STATE_5     = 5,
    STATE_6     = 6,
    STATE_7     = 7,
    STATE_8     = 8,
    STATE_9     = 9,
    STATE_10    = 10,
    READY_STATE = 11,
    UNASSIGNED  = 12,
    DONE        = '1
  } state_t;


  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s1_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s1_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s1_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s1_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s2_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s2_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s2_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s2_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s3_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s3_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s3_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s3_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s4_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s4_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s4_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s4_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s5_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s5_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s5_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s5_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s6_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s6_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s6_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s6_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s7_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s7_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s7_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s7_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s8_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s8_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s8_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s8_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s9_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s9_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s9_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s9_r;  //here will work with 14 factional bits

  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s10_b;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] x_s10_r;  //here will work with 14 factional bits
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s10_b;  //here will work with 14 factional bits  
  logic signed [IN_WIDTH+IN_WIDTH-1:0] y_s10_r;  //here will work with 14 factional bits


  logic signed [IN_WIDTH+IN_WIDTH-1:0] igc;

  parameter X_IGC_WIDTH = (IN_WIDTH + IN_WIDTH) * 2;

  /* verilator lint_off UNUSEDSIGNAL */
  logic signed  [X_IGC_WIDTH-1:0]   x_igc; //implicitly: we move multiply by 4: this is done only be working with 12 fract bits instead of 14
  /* verilator lint_on UNUSEDSIGNAL */
  assign igc   = 16'h136F;  //~ 1.2144775390625 //N.B. using 12 fractional bits
  assign x_igc = x_s10_r * igc;




  state_t                                     state_b;
  state_t                                     state_r;


  logic        [VALID_IN_DELAY_LINE_SIZE-1:0] valid_in_delay_line_r;
  logic signed [                  IN_WIDTH:0] zero_point_25;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s1;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s1;
  logic        [              K_WORDSIZE-1:0] k_s1_b;
  logic        [              K_WORDSIZE-1:0] k_s1_r;
  logic        [              K_WORDSIZE-1:0] idx_s1_b;
  logic        [              K_WORDSIZE-1:0] idx_s1_r;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s2;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s2;
  logic        [              K_WORDSIZE-1:0] k_s2_b;
  logic        [              K_WORDSIZE-1:0] k_s2_r;
  logic        [              K_WORDSIZE-1:0] idx_s2_b;
  logic        [              K_WORDSIZE-1:0] idx_s2_r;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s3;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s3;
  logic        [              K_WORDSIZE-1:0] k_s3_b;
  logic        [              K_WORDSIZE-1:0] k_s3_r;
  logic        [              K_WORDSIZE-1:0] idx_s3_b;
  logic        [              K_WORDSIZE-1:0] idx_s3_r;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s4;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s4;
  logic        [              K_WORDSIZE-1:0] k_s4_b;
  logic        [              K_WORDSIZE-1:0] k_s4_r;
  logic        [              K_WORDSIZE-1:0] idx_s4_b;
  logic        [              K_WORDSIZE-1:0] idx_s4_r;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s5;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s5;
  logic        [              K_WORDSIZE-1:0] k_s5_b;
  logic        [              K_WORDSIZE-1:0] k_s5_r;
  logic        [              K_WORDSIZE-1:0] idx_s5_b;
  logic        [              K_WORDSIZE-1:0] idx_s5_r;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s6;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s6;
  logic        [              K_WORDSIZE-1:0] k_s6_b;
  logic        [              K_WORDSIZE-1:0] k_s6_r;
  logic        [              K_WORDSIZE-1:0] idx_s6_b;
  logic        [              K_WORDSIZE-1:0] idx_s6_r;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s7;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s7;
  logic        [              K_WORDSIZE-1:0] k_s7_b;
  logic        [              K_WORDSIZE-1:0] k_s7_r;
  logic        [              K_WORDSIZE-1:0] idx_s7_b;
  logic        [              K_WORDSIZE-1:0] idx_s7_r;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s8;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s8;
  logic        [              K_WORDSIZE-1:0] k_s8_b;
  logic        [              K_WORDSIZE-1:0] k_s8_r;
  logic        [              K_WORDSIZE-1:0] idx_s8_b;
  logic        [              K_WORDSIZE-1:0] idx_s8_r;

  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s9;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s9;
  logic        [              K_WORDSIZE-1:0] k_s9_b;
  logic        [              K_WORDSIZE-1:0] k_s9_r;
  logic        [              K_WORDSIZE-1:0] idx_s9_b;
  logic        [              K_WORDSIZE-1:0] idx_s9_r;


  // verilator lint_off UNUSED
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] xtmp_s10;
  logic signed [       IN_WIDTH+IN_WIDTH-1:0] ytmp_s10;
  // verilator lint_on UNUSED

  logic        [              K_WORDSIZE-1:0] k_s10_b;

  // verilator lint_off UNUSED
  logic        [              K_WORDSIZE-1:0] k_s10_r;
  // verilator lint_on UNUSED

  logic        [              K_WORDSIZE-1:0] idx_s10_b;

  // verilator lint_off UNUSED
  logic        [              K_WORDSIZE-1:0] idx_s10_r;
  // verilator lint_on UNUSED

  //assign log2_v_in = ($clog2(v_in) -3); // -3 bc 3 bit decimal precission
  assign xtmp_s1 = (x_s1_r >>> 1);
  assign ytmp_s1 = (y_s1_r >>> 1);
  assign xtmp_s2 = (x_s2_r >>> 2);
  assign ytmp_s2 = (y_s2_r >>> 2);
  assign xtmp_s3 = (x_s3_r >>> 3);
  assign ytmp_s3 = (y_s3_r >>> 3);
  assign xtmp_s4 = (x_s4_r >>> 4);
  assign ytmp_s4 = (y_s4_r >>> 4);
  assign xtmp_s5 = (x_s5_r >>> 5);
  assign ytmp_s5 = (y_s5_r >>> 5);
  assign xtmp_s6 = (x_s6_r >>> 6);
  assign ytmp_s6 = (y_s6_r >>> 6);
  assign xtmp_s7 = (x_s7_r >>> 7);
  assign ytmp_s7 = (y_s7_r >>> 7);
  assign xtmp_s8 = (x_s8_r >>> 8);
  assign ytmp_s8 = (y_s8_r >>> 8);
  assign xtmp_s9 = (x_s9_r >>> 9);
  assign ytmp_s9 = (y_s9_r >>> 9);
  assign xtmp_s10 = (x_s10_r >>> 10);
  assign ytmp_s10 = (y_s10_r >>> 10);


  assign v_in_ready = (state_r == READY_STATE) ? 1 : 0;
  assign v_out_valid = (valid_in_delay_line_r[VALID_IN_DELAY_LINE_SIZE-1] && v_out_ready);

  always_ff @(posedge clk) begin
    if (rst) // sv720 TODO: check if reset is active high or low
        begin
      x_s1_r    <= '0;
      y_s1_r    <= '0;
      idx_s1_r  <= 1;
      x_s2_r    <= '0;
      y_s2_r    <= '0;
      idx_s2_r  <= 1;
      x_s3_r    <= '0;
      y_s3_r    <= '0;
      idx_s3_r  <= 1;
      x_s4_r    <= '0;
      y_s4_r    <= '0;
      idx_s4_r  <= 1;
      x_s5_r    <= '0;
      y_s5_r    <= '0;
      idx_s5_r  <= 1;
      x_s6_r    <= '0;
      y_s6_r    <= '0;
      idx_s6_r  <= 1;
      x_s7_r    <= '0;
      y_s7_r    <= '0;
      idx_s7_r  <= 1;
      x_s8_r    <= '0;
      y_s8_r    <= '0;
      idx_s8_r  <= 1;
      x_s9_r    <= '0;
      y_s9_r    <= '0;
      idx_s9_r  <= 1;
      x_s10_r   <= '0;
      y_s10_r   <= '0;
      idx_s10_r <= 1;
      state_r   <= RST;
      k_s1_r    <= '0;
      k_s2_r    <= '0;
      k_s1_r    <= '0;
      k_s2_r    <= '0;
      k_s3_r    <= '0;
      k_s4_r    <= '0;
      k_s5_r    <= '0;
      k_s6_r    <= '0;
      k_s7_r    <= '0;
      k_s8_r    <= '0;
      k_s9_r    <= '0;
      k_s10_r   <= '0;

      for (int i = 0; i < VALID_IN_DELAY_LINE_SIZE; i++) begin
        valid_in_delay_line_r[i] <= '0;
      end

    end else begin
      x_s1_r                   <= x_s1_b;
      y_s1_r                   <= y_s1_b;
      k_s1_r                   <= k_s1_b;
      idx_s1_r                 <= idx_s1_b;
      x_s2_r                   <= x_s2_b;
      y_s2_r                   <= y_s2_b;
      k_s2_r                   <= k_s2_b;
      idx_s2_r                 <= idx_s2_b;
      x_s3_r                   <= x_s3_b;
      y_s3_r                   <= y_s3_b;
      k_s3_r                   <= k_s3_b;
      idx_s3_r                 <= idx_s3_b;
      x_s4_r                   <= x_s4_b;
      y_s4_r                   <= y_s4_b;
      k_s4_r                   <= k_s4_b;
      idx_s4_r                 <= idx_s4_b;
      x_s5_r                   <= x_s5_b;
      y_s5_r                   <= y_s5_b;
      k_s5_r                   <= k_s5_b;
      idx_s5_r                 <= idx_s5_b;
      x_s6_r                   <= x_s6_b;
      y_s6_r                   <= y_s6_b;
      k_s6_r                   <= k_s6_b;
      idx_s6_r                 <= idx_s6_b;
      x_s7_r                   <= x_s7_b;
      y_s7_r                   <= y_s7_b;
      k_s7_r                   <= k_s7_b;
      idx_s7_r                 <= idx_s7_b;
      x_s8_r                   <= x_s8_b;
      y_s8_r                   <= y_s8_b;
      k_s8_r                   <= k_s8_b;
      idx_s8_r                 <= idx_s8_b;
      x_s9_r                   <= x_s9_b;
      y_s9_r                   <= y_s9_b;
      k_s9_r                   <= k_s9_b;
      idx_s9_r                 <= idx_s9_b;
      x_s10_r                  <= x_s10_b;
      y_s10_r                  <= y_s10_b;
      k_s10_r                  <= k_s10_b;
      idx_s10_r                <= idx_s10_b;
      state_r                  <= state_b;

      valid_in_delay_line_r[0] <= v_in_valid;

      for (int i = 1; i < VALID_IN_DELAY_LINE_SIZE; i++) begin
        valid_in_delay_line_r[i] <= valid_in_delay_line_r[i-1];
      end
    end
  end

  // verilator lint_off UNUSED
  logic dbg_in_if_1;  //TODO: remove 
  logic dbg_in_if_2;  //TODO: remove 
  logic dbg_in_if_3;  //TODO: remove 
  logic dbg_in_if_4;  //TODO: remove 
  logic dbg_in_if_5;  //TODO: remove 
  // verilator lint_on UNUSED

  always_latch begin
    //Set default values of _b wires here
    zero_point_25             = '0;
    zero_point_25[IN_WIDTH-3] = 1'b1;

    state_b                   = RST;

    // v_out_valid             = '0;
    v_out                     = '0;
    dbg_in_if_1               = '0;
    dbg_in_if_2               = '0;
    dbg_in_if_3               = '0;
    dbg_in_if_4               = '0;
    dbg_in_if_5               = '0;

    if (v_in_valid && !rst) begin
      x_s1_b = {v_in + zero_point_25, 7'b0};  //(32'b10 << right_shift_to_apply_b); // + 0.25
      y_s1_b = {v_in - zero_point_25, 7'b0};  //(32'b10 << right_shift_to_apply_b); // - 0.25
      k_s1_b = 4;
    end

    if (v_out_valid) begin
      v_out = (x_igc[(X_IGC_WIDTH-1-3):(X_IGC_WIDTH-1-10)]);
    end


    //STAGE 1: 
    idx_s2_b = idx_s1_r + 1;
    if ($signed(y_s1_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s2_b = $signed(x_s1_r) + $signed(ytmp_s1);
      y_s2_b = $signed(y_s1_r) + $signed(xtmp_s1);
    end else begin
      dbg_in_if_2 = '1;
      x_s2_b = x_s1_r - ytmp_s1;
      y_s2_b = y_s1_r - xtmp_s1;
    end

    if (idx_s1_r == k_s1_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s1_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s2_b = x_s1_r + ytmp_s1;
        y_s2_b = y_s1_r + xtmp_s1;
      end else begin
        dbg_in_if_5 = '1;
        x_s2_b = x_s1_r - ytmp_s1;
        y_s2_b = y_s1_r - xtmp_s1;
      end

      k_s2_b = 3 * k_s1_r + 1;
    end

    //STAGE 2:
    idx_s3_b = idx_s2_r + 1;
    if ($signed(y_s2_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s3_b = $signed(x_s2_r) + $signed(ytmp_s2);
      y_s3_b = $signed(y_s2_r) + $signed(xtmp_s2);
    end else begin
      dbg_in_if_2 = '1;
      x_s3_b = x_s2_r - ytmp_s2;
      y_s3_b = y_s2_r - xtmp_s2;
    end


    if (idx_s2_r == k_s2_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s2_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s3_b = x_s2_r + ytmp_s2;
        y_s3_b = y_s2_r + xtmp_s2;
      end else begin
        dbg_in_if_5 = '1;
        x_s3_b = x_s2_r - ytmp_s2;
        y_s3_b = y_s2_r - xtmp_s2;
      end

      k_s3_b = 3 * k_s2_r + 1;
    end

    //STAGE 3:
    idx_s4_b = idx_s3_r + 1;
    if ($signed(y_s3_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s4_b = $signed(x_s3_r) + $signed(ytmp_s3);
      y_s4_b = $signed(y_s3_r) + $signed(xtmp_s3);
    end else begin
      dbg_in_if_2 = '1;
      x_s4_b = x_s3_r - ytmp_s3;
      y_s4_b = y_s3_r - xtmp_s3;
    end


    if (idx_s3_r == k_s3_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s3_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s4_b = x_s3_r + ytmp_s3;
        y_s4_b = y_s3_r + xtmp_s3;
      end else begin
        dbg_in_if_5 = '1;
        x_s4_b = x_s3_r - ytmp_s3;
        y_s4_b = y_s3_r - xtmp_s3;
      end

      k_s4_b = 3 * k_s3_r + 1;
    end

    //STAGE 4: 
    idx_s5_b = idx_s4_r + 1;
    if ($signed(y_s4_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s5_b = $signed(x_s4_r) + $signed(ytmp_s4);
      y_s5_b = $signed(y_s4_r) + $signed(xtmp_s4);
    end else begin
      dbg_in_if_2 = '1;
      x_s5_b = x_s4_r - ytmp_s4;
      y_s5_b = y_s4_r - xtmp_s4;
    end

    if (idx_s4_r == k_s4_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s4_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s5_b = x_s4_r + ytmp_s4;
        y_s5_b = y_s4_r + xtmp_s4;
      end else begin
        dbg_in_if_5 = '1;
        x_s5_b = x_s4_r - ytmp_s4;
        y_s5_b = y_s4_r - xtmp_s4;
      end

      k_s5_b = 3 * k_s4_r + 1;
    end

    //STAGE 5: 
    idx_s6_b = idx_s5_r + 1;
    if ($signed(y_s5_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s6_b = $signed(x_s5_r) + $signed(ytmp_s5);
      y_s6_b = $signed(y_s5_r) + $signed(xtmp_s5);
    end else begin
      dbg_in_if_2 = '1;
      x_s6_b = x_s5_r - ytmp_s5;
      y_s6_b = y_s5_r - xtmp_s5;
    end

    if (idx_s5_r == k_s5_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s5_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s6_b = x_s5_r + ytmp_s5;
        y_s6_b = y_s5_r + xtmp_s5;
      end else begin
        dbg_in_if_5 = '1;
        x_s6_b = x_s5_r - ytmp_s5;
        y_s6_b = y_s5_r - xtmp_s5;
      end

      k_s6_b = 3 * k_s5_r + 1;
    end

    //STAGE 6: 
    idx_s7_b = idx_s6_r + 1;
    if ($signed(y_s6_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s7_b = $signed(x_s6_r) + $signed(ytmp_s6);
      y_s7_b = $signed(y_s6_r) + $signed(xtmp_s6);
    end else begin
      dbg_in_if_2 = '1;
      x_s7_b = x_s6_r - ytmp_s6;
      y_s7_b = y_s6_r - xtmp_s6;
    end

    if (idx_s6_r == k_s6_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s6_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s7_b = x_s6_r + ytmp_s6;
        y_s7_b = y_s6_r + xtmp_s6;
      end else begin
        dbg_in_if_5 = '1;
        x_s7_b = x_s6_r - ytmp_s6;
        y_s7_b = y_s6_r - xtmp_s6;
      end

      k_s7_b = 3 * k_s6_r + 1;
    end

    //STAGE 7: 
    idx_s8_b = idx_s7_r + 1;
    if ($signed(y_s7_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s8_b = $signed(x_s7_r) + $signed(ytmp_s7);
      y_s8_b = $signed(y_s7_r) + $signed(xtmp_s7);
    end else begin
      dbg_in_if_2 = '1;
      x_s8_b = x_s7_r - ytmp_s7;
      y_s8_b = y_s7_r - xtmp_s7;
    end

    if (idx_s7_r == k_s7_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s7_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s8_b = x_s7_r + ytmp_s7;
        y_s8_b = y_s7_r + xtmp_s7;
      end else begin
        dbg_in_if_5 = '1;
        x_s8_b = x_s7_r - ytmp_s7;
        y_s8_b = y_s7_r - xtmp_s7;
      end

      k_s8_b = 3 * k_s7_r + 1;
    end

    //STAGE 8: 
    idx_s9_b = idx_s8_r + 1;

    if ($signed(y_s8_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s9_b = $signed(x_s8_r) + $signed(ytmp_s8);
      y_s9_b = $signed(y_s8_r) + $signed(xtmp_s8);
    end else begin
      dbg_in_if_2 = '1;
      x_s9_b = x_s8_r - ytmp_s8;
      y_s9_b = y_s8_r - xtmp_s8;
    end

    if (idx_s8_r == k_s8_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s8_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s9_b = x_s8_r + ytmp_s8;
        y_s9_b = y_s8_r + xtmp_s8;
      end else begin
        dbg_in_if_5 = '1;
        x_s9_b = x_s8_r - ytmp_s8;
        y_s9_b = y_s8_r - xtmp_s8;
      end

      k_s9_b = 3 * k_s8_r + 1;
    end

    //STAGE 9: 
    idx_s10_b = idx_s9_r + 1;
    if ($signed(y_s9_r) < 0) begin
      dbg_in_if_1 = '1;
      x_s10_b = $signed(x_s9_r) + $signed(ytmp_s9);
      y_s10_b = $signed(y_s9_r) + $signed(xtmp_s9);
    end else begin
      dbg_in_if_2 = '1;
      x_s10_b = x_s9_r - ytmp_s9;
      y_s10_b = y_s9_r - xtmp_s9;
    end

    if (idx_s9_r == k_s9_r) //if state is k: do it again
        begin
      dbg_in_if_3 = '1;

      if ($signed(y_s9_r) < 0) begin
        dbg_in_if_4 = '1;
        x_s10_b = x_s9_r + ytmp_s9;
        y_s10_b = y_s9_r + xtmp_s9;
      end else begin
        dbg_in_if_5 = '1;
        x_s10_b = x_s9_r - ytmp_s9;
        y_s10_b = y_s9_r - xtmp_s9;
      end

      k_s10_b = 3 * k_s9_r + 1;
    end

    //END



    if (state_r == RST) begin
      state_b = READY_STATE;
    end
        else if ((state_r == READY_STATE) && v_in_valid) //N.B. 1 cycle delay (potential for optimization)
        begin

      state_b = STATE_1;

    end else if (state_r == READY_STATE) begin
      state_b = READY_STATE;

    end
        else if (state_r == DONE) //TODO: take out of state machine
        begin


      state_b  = READY_STATE;
      idx_s1_b = 1;
    end else if (state_r == STATE_10) begin
      state_b = DONE;
    end else if (state_r == STATE_1) begin
      state_b = STATE_2;
    end else if (state_r == STATE_2) begin
      state_b = STATE_3;
    end else if (state_r == STATE_3) begin
      state_b = STATE_4;
    end else if (state_r == STATE_4) begin
      state_b = STATE_5;
    end else if (state_r == STATE_5) begin
      state_b = STATE_6;

    end else if (state_r == STATE_6) begin
      state_b = STATE_7;

    end else if (state_r == STATE_7) begin
      state_b = STATE_8;
    end  //=============================================
    else if (state_r == STATE_8) begin
      state_b = STATE_9;

    end  //=============================================

         //=============================================
    else if (state_r == STATE_9) begin
      state_b = STATE_10;
    end
  end

endmodule


// MATLAB code:
// function x = localCORDICSQRT(v,n)

// % Initialize and run CORDIC kernel for N iterations

// x = v + cast(0.25, 'like', v); % v + 0.25 in same data type
// y = v - cast(0.25, 'like', v); % v - 0.25 in same data type

// k = 4; % Used for the repeated (3*k + 1) iteration steps

// for idx = 1:n
//     xtmp = bitsra(x, idx); % multiply by 2^(-idx)
//     ytmp = bitsra(y, idx); % multiply by 2^(-idx)
//     if y < 0
//         x(:) = x + ytmp;
//         y(:) = y + xtmp;
//     else
//         x(:) = x - ytmp;
//         y(:) = y - xtmp;
//     end

//     if idx==k
//         xtmp = bitsra(x, idx); % multiply by 2^(-idx)
//         ytmp = bitsra(y, idx); % multiply by 2^(-idx)
//         if y < 0
//             x(:) = x + ytmp;
//             y(:) = y + xtmp;
//         else
//             x(:) = x - ytmp;
//             y(:) = y - xtmp;
//         end
//         k = 3*k + 1;
//     end
// end % idx loop

// end % function
