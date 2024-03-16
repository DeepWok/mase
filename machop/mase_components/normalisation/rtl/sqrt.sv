//This is a cordic square-root module that takes inpiration for the localFixedPointCORDICSQRT MATLAB function
module sqrt #(
    parameter IN_WIDTH      = 8,
    parameter IN_FRAC_WIDTH = 3, 
    parameter NUM_ITERATION = 10

)(
    input                           clk,
    input                           rst, 
    input           [IN_WIDTH-1:0]  v_in, 
    input                           v_in_valid, 
    output logic                    v_in_ready,

    
    

    output logic    [IN_WIDTH-1:0]  v_out, 
    output logic                    v_out_valid,
    input                           v_out_ready //TODO: assign to this
    
);  

    parameter NUM_STATES = NUM_ITERATION + 2; // a rst and final state
    parameter NUM_STATE_BITS = $clog2(NUM_STATES) + 1;
    parameter K_WORDSIZE = 32;

    parameter LOG2_IN_WIDTH = $clog2(IN_WIDTH);

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


    logic signed  [IN_WIDTH+IN_WIDTH-1:0]       x_b; //here will work with 14 factional bits
    logic signed  [IN_WIDTH+IN_WIDTH-1:0]       x_r; //here will work with 14 factional bits
    logic signed  [IN_WIDTH+IN_WIDTH-1:0]       y_b; //here will work with 14 factional bits  
    logic signed  [IN_WIDTH+IN_WIDTH-1:0]       y_r; //here will work with 14 factional bits
    logic signed  [IN_WIDTH+IN_WIDTH-1:0]       igc;

    parameter X_IGC_WIDTH = (IN_WIDTH+IN_WIDTH)*2;
    
    logic signed  [X_IGC_WIDTH-1:0]   x_igc; //implicitly: we move multiply by 4: this is done only be working with 12 fract bits instead of 14

    assign igc = 16'h136F; //~ 1.2144775390625 //N.B. using 12 fractional bits
    assign x_igc = x_r*igc;

    
                 

    state_t state_b; 
    state_t state_r; 

    // logic signed  [IN_WIDTH:0]        v_more_fractional;
    logic signed  [IN_WIDTH:0]        zero_point_25; 

    logic   [LOG2_IN_WIDTH-1:0] right_shift_to_apply_b;
    logic   [LOG2_IN_WIDTH-1:0] right_shift_to_apply_r;

    logic signed  [IN_WIDTH+IN_WIDTH-1:0]      xtmp;  
    logic signed  [IN_WIDTH+IN_WIDTH-1:0]      ytmp;

    logic   [K_WORDSIZE-1:0]    k_b;
    logic   [K_WORDSIZE-1:0]    k_r; 

    logic   [K_WORDSIZE-1:0]    idx_b; 
    logic   [K_WORDSIZE-1:0]    idx_r;  

    //assign log2_v_in = ($clog2(v_in) -3); // -3 bc 3 bit decimal precission
    assign xtmp = (x_r >>> state_r);
    assign ytmp = (y_r >>> state_r);
    assign v_in_ready = (state_r == READY_STATE) ? 1 : 0; 

    assign zero_point_25 = 8'h20;//(32'b10 << right_shift_to_apply_b);


    always_ff @(posedge clk)
    begin
        if (rst) // sv720 TODO: check if reset is active high or low
        begin 
            x_r                     <= '0; 
            y_r                     <= '0; 
            state_r                 <=  RST;
            k_r                     <=  '0;
            right_shift_to_apply_r  <= '0; 
            idx_r                   <= 1;

        end
        else
        begin
            x_r                     <= x_b;
            y_r                     <= y_b;
            state_r                 <= state_b;
            k_r                     <= k_b;
            idx_r                   <= idx_b;
            right_shift_to_apply_r  <= right_shift_to_apply_b;
        end
    end 

    logic dbg_in_if_1; //TODO: remove 
    logic dbg_in_if_2; //TODO: remove 
    logic dbg_in_if_3; //TODO: remove 
    logic dbg_in_if_4; //TODO: remove 
    logic dbg_in_if_5; //TODO: remove 

    always_comb
    begin 
        //Set default values of _b wires here
        k_b                     = k_r; 
        idx_b                   = idx_r;
        state_b                 = RST;
        x_b                     = x_r; 
        y_b                     = y_r; 
        v_out_valid             = '0;
        v_out                   = '0; 
        right_shift_to_apply_b  = right_shift_to_apply_r;
        dbg_in_if_1 = '0;
        dbg_in_if_2 = '0;
        dbg_in_if_3 = '0;
        dbg_in_if_4 = '0;
        dbg_in_if_5 = '0;

        if (state_r == RST)
        begin 
            state_b = READY_STATE;
        end
        else if ((state_r == READY_STATE) && v_in_valid) //N.B. 1 cycle delay (potential for optimization)
        begin
            //shift the values such that we have as many fractional bits to work with
            //find MSB 
            // for (int i=0; i<IN_WIDTH; i++)
            // begin
            //     if (v_in[i] == 1)
            //     begin
            //         right_shift_to_apply_b  = (IN_WIDTH-1-i);

            //         //N.B. if odd we need to *2 //TODO: check if this is needed
            //         v_more_fractional       = ((IN_WIDTH-1-i)%2 ==0) ? (v_in << (IN_WIDTH-1-i)) : (v_in << (IN_WIDTH-i));
                    
            //     end
            // end
            x_b     = {v_in + zero_point_25, 7'b0}; //(32'b10 << right_shift_to_apply_b); // + 0.25
            y_b     = {v_in - zero_point_25, 7'b0}; //(32'b10 << right_shift_to_apply_b); // - 0.25
            state_b = STATE_1;
            k_b     = 4;
        end
        else if (state_r == READY_STATE)
        begin
            state_b = READY_STATE;

        end
        else if (state_r == DONE)
        begin 
            v_out_valid = 1;
            v_out       = ((x_igc[(X_IGC_WIDTH-1-3): (X_IGC_WIDTH-1-10)]) >>> right_shift_to_apply_r);
            state_b     = READY_STATE;
            idx_b       = 1;
        end
        else if (state_r == STATE_10)
        begin
            state_b     = DONE;               
        end
        else
        begin
            state_b =  state_t'(state_r + 1);
            idx_b   = idx_r + 1;
            if ($signed(y_r) < 0)
            begin 
                dbg_in_if_1 = '1;
                x_b = $signed(x_r) + $signed(ytmp); 
                y_b = $signed(y_r) + $signed(xtmp); 
            end
            else
            begin
                dbg_in_if_2 = '1;
                x_b = x_r - ytmp;
                y_b = y_r - xtmp;
            end 

            
            if (idx_r == k_r) //if state is k: do it again
            begin
                dbg_in_if_3 = '1;
                
                if ($signed(y_r) < 0)
                begin 
                    dbg_in_if_4 = '1;
                    x_b = x_r + ytmp; 
                    y_b = y_r + xtmp; 
                end
                else
                begin
                    dbg_in_if_5 = '1;
                    x_b = x_r - ytmp;
                    y_b = y_r - xtmp;
                end 

                k_b = 3*k_r + 1;
            end
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



//COSINE cordic implementation

//Q format is 2.30
/* 
`define K 32'h26dd3b6a  // = 0.6072529350088814

`define BETA_0  32'h3243f6a9  // = atan 2^0     = 0.7853981633974483
`define BETA_1  32'h1dac6705  // = atan 2^(-1)  = 0.4636476090008061
`define BETA_2  32'h0fadbafd  // = atan 2^(-2)  = 0.24497866312686414
`define BETA_3  32'h07f56ea7  // = atan 2^(-3)  = 0.12435499454676144
`define BETA_4  32'h03feab77  // = atan 2^(-4)  = 0.06241880999595735
`define BETA_5  32'h01ffd55c  // = atan 2^(-5)  = 0.031239833430268277
`define BETA_6  32'h00fffaab  // = atan 2^(-6)  = 0.015623728620476831
`define BETA_7  32'h007fff55  // = atan 2^(-7)  = 0.007812341060101111
`define BETA_8  32'h003fffeb  // = atan 2^(-8)  = 0.0039062301319669718
`define BETA_9  32'h001ffffd  // = atan 2^(-9)  = 0.0019531225164788188
`define BETA_10 32'h00100000  // = atan 2^(-10) = 0.0009765621895593195
`define BETA_11 32'h00080000  // = atan 2^(-11) = 0.0004882812111948983
`define BETA_12 32'h00040000  // = atan 2^(-12) = 0.00024414062014936177
`define BETA_13 32'h00020000  // = atan 2^(-13) = 0.00012207031189367021
`define BETA_14 32'h00010000  // = atan 2^(-14) = 6.103515617420877e-05
`define BETA_15 32'h00008000  // = atan 2^(-15) = 3.0517578115526096e-05
`define BETA_16 32'h00004000  // = atan 2^(-16) = 1.5258789061315762e-05
`define BETA_17 32'h00002000  // = atan 2^(-17) = 7.62939453110197e-06
`define BETA_18 32'h00001000  // = atan 2^(-18) = 3.814697265606496e-06
`define BETA_19 32'h00000800  // = atan 2^(-19) = 1.907348632810187e-06
`define BETA_20 32'h00000400  // = atan 2^(-20) = 9.536743164059608e-07
`define BETA_21 32'h00000200  // = atan 2^(-21) = 4.7683715820308884e-07
`define BETA_22 32'h00000100  // = atan 2^(-22) = 2.3841857910155797e-07
`define BETA_23 32'h00000080  // = atan 2^(-23) = 1.1920928955078068e-07
`define BETA_24 32'h00000040  // = atan 2^(-24) = 5.960464477539055e-08
`define BETA_25 32'h00000020  // = atan 2^(-25) = 2.9802322387695303e-08
`define BETA_26 32'h00000010  // = atan 2^(-26) = 1.4901161193847655e-08
`define BETA_27 32'h00000008  // = atan 2^(-27) = 7.450580596923828e-09
`define BETA_28 32'h00000004  // = atan 2^(-28) = 3.725290298461914e-09
`define BETA_29 32'h00000002  // = atan 2^(-29) = 1.862645149230957e-09
`define BETA_30 32'h00000001  // = atan 2^(-30) = 9.313225746154785e-10
`define BETA_31 32'h00000000  // = atan 2^(-31) = 4.656612873077393e-10

module cordic(
    //angle,

    done, //asserted for 1 cycle when result is ready
    clock,    // Master clock
    reset,    // Master asynchronous reset (active-high)
    start,    // An input signal that the user of this module should set to high when computation should begin
    angle_in, // Input angle
    cos_out,  // Output value for cosine of angle
    //sin_out   // Output value for sine of angle
);


input clock;
input reset;
input start;
input [31:0] angle_in;
output [31:0] cos_out;
output done; 
//output [31:0] sin_out;
//output [31:0]angle;

reg [31:0] cos;
reg [31:0] sin;

wire [31:0] cos_out = cos;
//wire [31:0] sin_out = sin;


reg [31:0] angle;
reg [4:0] count;
reg state;
reg done; //this is a wire as just a flag
reg done_next; 

reg [31:0] cos_next;
reg [31:0] sin_next;
reg [31:0] angle_next;
reg [4:0] count_next;
reg state_next;
wire direction_negative = angle[31];
wire [31:0] cos_signbits = {32{cos[31]}};
wire [31:0] sin_signbits = {32{sin[31]}};
wire [31:0] cos_shr = {cos_signbits, cos} >> count;
wire [31:0] sin_shr = {sin_signbits, sin} >> count;




wire [31:0] beta_lut [0:31];
assign beta_lut[0] = `BETA_0;
assign beta_lut[1] = `BETA_1;
assign beta_lut[2] = `BETA_2;
assign beta_lut[3] = `BETA_3;
assign beta_lut[4] = `BETA_4;
assign beta_lut[5] = `BETA_5;
assign beta_lut[6] = `BETA_6;
assign beta_lut[7] = `BETA_7;
assign beta_lut[8] = `BETA_8;
assign beta_lut[9] = `BETA_9;
assign beta_lut[10] = `BETA_10;
assign beta_lut[11] = `BETA_11;
assign beta_lut[12] = `BETA_12;
assign beta_lut[13] = `BETA_13;
assign beta_lut[14] = `BETA_14;
assign beta_lut[15] = `BETA_15;
assign beta_lut[16] = `BETA_16;
assign beta_lut[17] = `BETA_17;
assign beta_lut[18] = `BETA_18;
assign beta_lut[19] = `BETA_19;
assign beta_lut[20] = `BETA_20;
assign beta_lut[21] = `BETA_21;
assign beta_lut[22] = `BETA_22;
assign beta_lut[23] = `BETA_23;
assign beta_lut[24] = `BETA_24;
assign beta_lut[25] = `BETA_25;
assign beta_lut[26] = `BETA_26;
assign beta_lut[27] = `BETA_27;
assign beta_lut[28] = `BETA_28;
assign beta_lut[29] = `BETA_29;
assign beta_lut[30] = `BETA_30;
assign beta_lut[31] = `BETA_31;

wire [31:0] beta = beta_lut[count];

always @(posedge clock or posedge reset) begin
    if (reset) begin
		  done <= 0;
		  //done_next <=0; 
        cos <= 0;
        sin <= 0;
        angle <= 0;
        count <= 0;
        state <= 0;
    end else begin
		  done <= done_next; 
        cos <= cos_next;
        sin <= sin_next;
        angle <= angle_next;
        count <= count_next;
        state <= state_next;
    end
end

always @* begin
    // Set all logic regs to a value to prevent any of them holding the value
    // from last tick and hence being misinterpreted as hardware registers.
    cos_next = cos;
    sin_next = sin;
    angle_next = angle;
    count_next = count;
    state_next = state;
	 
	 if (count == 29) begin
            done_next = 1; 
	  end
	  else begin 
            done_next = 0;
	  end
	  
    
	 
    if (state) begin
        // Compute mode.
        cos_next = cos + (direction_negative ? sin_shr : -sin_shr);
        sin_next = sin + (direction_negative ? -cos_shr : cos_shr);
        angle_next = angle + (direction_negative ? beta : -beta);
        count_next = count + 1;
        
        if (count == 31) begin
            // If this is the last iteration, go back to the idle state.
            //done_next = 1; 
            state_next = 0;
        end
    end
    
    else begin
        // Idle mode.
        if (start) begin
            cos_next = `K;         // Set up initial value for cos.
            sin_next = 0;          // Set up initial value for sin.
            angle_next = angle_in; // Latch input angle into the angle register.
            count_next = 0;        // Set up counter.
            state_next = 1;        // Go to compute mode.
        end
    end
end

endmodule
*/