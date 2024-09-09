module swin_controller 
#(

    parameter LAYER_NORM_PL0_0_TOTAL_MAX_DIM0 = 16,
    parameter LAYER_NORM_PL0_0_TOTAL_MAX_DIM1 = 16,
    parameter LAYER_NORM_PL0_0_PARALLELISM_DIM0 = 2,
    parameter LAYER_NORM_PL0_0_PARALLELISM_DIM1 = 2,
    parameter LAYER_NORM_PL0_0_PRECISION_0 = 16,
    parameter LAYER_NORM_PL0_0_PRECISION_1 = 8,

    localparam LAYER_NORM_PL0_0_MAX_DEPTH_DIM0 = LAYER_NORM_PL0_0_TOTAL_MAX_DIM0/LAYER_NORM_PL0_0_PARALLELISM_DIM0, 
    localparam LAYER_NORM_PL0_0_MAX_DEPTH_DIM1 = LAYER_NORM_PL0_0_TOTAL_MAX_DIM1/LAYER_NORM_PL0_0_PARALLELISM_DIM1,
    localparam LAYER_NORM_PL0_0_NUM_ITERS_MAX = LAYER_NORM_PL0_0_MAX_DEPTH_DIM0 * LAYER_NORM_PL0_0_MAX_DEPTH_DIM1,
    localparam LAYER_NORM_PL0_0_ITER_WIDTH = $clog2(LAYER_NORM_PL0_0_NUM_ITERS_MAX),
    localparam LAYER_NORM_PL0_0_ADDER_TREE_IN_SIZE = LAYER_NORM_PL0_0_PARALLELISM_DIM0 * LAYER_NORM_PL0_0_PARALLELISM_DIM1,
    localparam LAYER_NORM_PL0_0_ADDER_TREE_OUT_WIDTH = $clog2(LAYER_NORM_PL0_0_ADDER_TREE_IN_SIZE) + LAYER_NORM_PL0_0_PRECISION_0,
    localparam LAYER_NORM_PL0_0_ACC_OUT_WIDTH = LAYER_NORM_PL0_0_ITER_WIDTH + LAYER_NORM_PL0_0_ADDER_TREE_OUT_WIDTH,

    localparam LAYER_NORM_PL0_0_DIFF_WIDTH = LAYER_NORM_PL0_0_PRECISION_0 + 1,
    localparam LAYER_NORM_PL0_0_SQUARE_WIDTH = LAYER_NORM_PL0_0_DIFF_WIDTH * 2,
    
    localparam LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_IN_SIZE = LAYER_NORM_PL0_0_PARALLELISM_DIM0 * LAYER_NORM_PL0_0_PARALLELISM_DIM1,
    localparam LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH = $clog2(LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_IN_SIZE) + LAYER_NORM_PL0_0_SQUARE_WIDTH,

    parameter LAYER_NORM_PL0_1_TOTAL_MAX_DIM0 = 16,
    parameter LAYER_NORM_PL0_1_TOTAL_MAX_DIM1 = 16,
    parameter LAYER_NORM_PL0_1_PARALLELISM_DIM0 = 2,
    parameter LAYER_NORM_PL0_1_PARALLELISM_DIM1 = 2,
    parameter LAYER_NORM_PL0_1_PRECISION_0 = 16,
    parameter LAYER_NORM_PL0_1_PRECISION_1 = 8,

    localparam LAYER_NORM_PL0_1_MAX_DEPTH_DIM0 = LAYER_NORM_PL0_1_TOTAL_MAX_DIM0/LAYER_NORM_PL0_1_PARALLELISM_DIM0, 
    localparam LAYER_NORM_PL0_1_MAX_DEPTH_DIM1 = LAYER_NORM_PL0_1_TOTAL_MAX_DIM1/LAYER_NORM_PL0_1_PARALLELISM_DIM1,
    localparam LAYER_NORM_PL0_1_NUM_ITERS_MAX = LAYER_NORM_PL0_1_MAX_DEPTH_DIM0 * LAYER_NORM_PL0_1_MAX_DEPTH_DIM1,
    localparam LAYER_NORM_PL0_1_ITER_WIDTH = $clog2(LAYER_NORM_PL0_1_NUM_ITERS_MAX),
    localparam LAYER_NORM_PL0_1_ADDER_TREE_IN_SIZE = LAYER_NORM_PL0_1_PARALLELISM_DIM0 * LAYER_NORM_PL0_1_PARALLELISM_DIM1,
    localparam LAYER_NORM_PL0_1_ADDER_TREE_OUT_WIDTH = $clog2(LAYER_NORM_PL0_1_ADDER_TREE_IN_SIZE) + LAYER_NORM_PL0_0_PRECISION_0,
    localparam LAYER_NORM_PL0_1_ACC_OUT_WIDTH = LAYER_NORM_PL0_1_ITER_WIDTH + LAYER_NORM_PL0_1_ADDER_TREE_OUT_WIDTH,

    
    localparam LAYER_NORM_PL0_1_DIFF_WIDTH = LAYER_NORM_PL0_1_PRECISION_0 + 1,
    localparam LAYER_NORM_PL0_1_SQUARE_WIDTH = LAYER_NORM_PL0_1_DIFF_WIDTH * 2,


    localparam LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_IN_SIZE = LAYER_NORM_PL0_1_PARALLELISM_DIM0 * LAYER_NORM_PL0_1_PARALLELISM_DIM1,
    localparam LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_OUT_WIDTH = $clog2(LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_IN_SIZE) + LAYER_NORM_PL0_1_SQUARE_WIDTH
)

(

    input logic clk,
    input logic rst,


    output logic [9:0] layer_norm_pl0_0_n_iters,
    output logic [39:0] layer_norm_pl0_0_inv_numvalues_0,
    output logic [39:0] layer_norm_pl0_0_inv_numvalues_1,
    output logic [9:0] mha_pl0_0_data_in_0_depth_dim_1,
    output logic [9:0] mha_pl0_0_weight_tensor_size_dim0,
    output logic [9:0] mha_pl0_0_weight_depth_dim_0,
    output logic [9:0] mha_pl0_0_weight_depth_dim_1,
    output logic [9:0] mha_pl0_0_weight_depth_mult,
    output logic [9:0] mha_pl0_0_block_per_head,
    output logic [9:0] mha_pl0_0_q_depth_dim_0,
    output logic [9:0] mha_pl0_0_q_depth_dim_1,
    output logic [9:0] mha_pl0_0_q_depth_mult,
    output logic [9:0] mha_pl0_0_weight_out_depth_dim_1,
    output logic [9:0] layer_norm_pl0_1_n_iters,
    output logic [39:0] layer_norm_pl0_1_inv_numvalues_0,
    output logic [39:0] layer_norm_pl0_1_inv_numvalues_1,
    output logic [9:0] linear_pl0_0_data_in_0_depth_dim1,
    output logic [9:0] linear_pl0_0_weight_tensor_size_dim0,
    output logic [9:0] linear_pl0_0_weight_depth_dim0,
    output logic [9:0] linear_pl0_0_weight_depth_dim1,
    output logic [9:0] linear_pl0_0_weight_depth_mult,
    output logic [9:0] linear_pl0_1_data_in_0_depth_dim1,
    output logic [9:0] linear_pl0_1_weight_tensor_size_dim0,
    output logic [9:0] linear_pl0_1_weight_depth_dim0,
    output logic [9:0] linear_pl0_1_weight_depth_dim1,
    output logic [9:0] linear_pl0_1_weight_depth_mult,
    output logic [9:0] roll_pl0_0_roll_distance,
    output logic [9:0] roll_pl0_0_buffer_size,
    output logic [9:0] roll_pl0_1_roll_distance,
    output logic [9:0] roll_pl0_1_buffer_size,
    input  logic done,
    output logic load_input_seq,
    output logic input_mux_ctrl,
    output logic output_mux_ctrl,
    output logic [14:0] counter_0_max_input_counter,
    output logic [14:0] counter_0_max_output_counter,
    input  logic counter_0_counter_max,
    output logic [14:0] counter_1_max_input_counter,
    output logic [14:0] counter_1_max_output_counter,
    input  logic counter_1_counter_max,
    output logic [14:0] counter_2_max_input_counter,
    output logic [14:0] counter_2_max_output_counter,
    input  logic counter_2_counter_max,
    output logic [14:0] counter_3_max_input_counter,
    output logic [14:0] counter_3_max_output_counter,
    input  logic counter_3_counter_max,

    output [3:0] counter_0,
    output [3:0] counter_1,

    output [2:0] instr_0,
    output [2:0] instr_1,
    output [2:0] instr_2,
    output [2:0] instr_3

);

//  typedef struct packed {
//     logic load_input_seq;
//     logic input_mux_ctrl;
//     logic output_mux_ctrl;
//  } instr

logic [2:0] instr [3:0] = {
    3'b000,
    3'b001,
    3'b010,
    3'b011
};


logic [2:0]  counter [3:0];
logic [14:0] counters_max [3:0];
logic [2:0]  current_instr [3:0];   

always_ff @(posedge clk)
    begin
        if(rst)
        begin
            for (int i = 0; i < 4; i++) begin
                    counter[i] <= 0;
            end
        end
        else begin
            for (int i = 0; i < 4; i++) begin
                    if(counters_max[i]) counter[i] <= counter[i] + 1;
            end
        end
    end

always_ff @(posedge clk)
begin
    if (rst) input_mux_ctrl <= 1;
    else if (counters_max[3]) input_mux_ctrl <= !input_mux_ctrl;
end

always_ff @(posedge clk)
begin
    if (rst) output_mux_ctrl <= 1;
    else if (counters_max[0] && counter[0]>0) output_mux_ctrl <= !output_mux_ctrl;
end

always_ff@(posedge clk)
begin
    if(rst) load_input_seq <= 1;
    else if (counters_max[0]) load_input_seq <= 0;
end

always_comb
begin
    for (int i = 0; i < 4 ; i++) begin
        current_instr[i] = instr[counter[i]];
    end
end


always_comb
begin
    //1st Layer Norm
    case (current_instr[0])

        //dim0 = 32, dim1 = 8
        3'b000: begin    layer_norm_pl0_0_n_iters = 32/LAYER_NORM_PL0_0_PARALLELISM_DIM0 * 8/LAYER_NORM_PL0_0_PARALLELISM_DIM1;
                    layer_norm_pl0_0_inv_numvalues_0 = (1<<LAYER_NORM_PL0_0_ACC_OUT_WIDTH)/8;
                    layer_norm_pl0_0_inv_numvalues_1 = (1<<LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
                    counter_0_max_input_counter = 32/LAYER_NORM_PL0_0_PARALLELISM_DIM0 * 8/LAYER_NORM_PL0_0_PARALLELISM_DIM1;
                    counter_0_max_output_counter = 32/LAYER_NORM_PL0_0_PARALLELISM_DIM0 * 8/LAYER_NORM_PL0_0_PARALLELISM_DIM1;
        end
        // 3'b001: begin    layer_norm_pl0_0_n_iters = 8;
        //             layer_norm_pl0_0_inv_numvalues_0 = (1<<LAYER_NORM_PL0_0_ACC_OUT_WIDTH)/16;
        //             layer_norm_pl0_0_inv_numvalues_1 = (1<<LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH)/16;
        // end

        // 3'b010: begin    layer_norm_pl0_0_n_iters = 16;
        //             layer_norm_pl0_0_inv_numvalues_0 = (1<<LAYER_NORM_PL0_0_ACC_OUT_WIDTH)/8;
        //             layer_norm_pl0_0_inv_numvalues_1 = (1<<LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
        // end

        // 3'b011: begin    layer_norm_pl0_0_n_iters = 32;
        //             layer_norm_pl0_0_inv_numvalues_0 = (1<<LAYER_NORM_PL0_0_ACC_OUT_WIDTH)/8;
        //             layer_norm_pl0_0_inv_numvalues_1 = (1<<LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
        // end

        // 3'b100: begin    layer_norm_pl0_0_n_iters = 64;
        //             layer_norm_pl0_0_inv_numvalues_0 = (1<<LAYER_NORM_PL0_0_ACC_OUT_WIDTH)/8;
        //             layer_norm_pl0_0_inv_numvalues_1 = (1<<LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
        // end

        default: begin    layer_norm_pl0_0_n_iters = 16;
                    layer_norm_pl0_0_inv_numvalues_0 = (1<<LAYER_NORM_PL0_0_ACC_OUT_WIDTH)/8;
                    layer_norm_pl0_0_inv_numvalues_1 = (1<<LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
                    counter_0_max_input_counter = 4;
                    counter_0_max_output_counter = 4;
        end
    endcase
    //1st MHA
    case (current_instr[1])
        3'b000:     begin mha_pl0_0_data_in_0_depth_dim_1 = 2;
                    mha_pl0_0_weight_tensor_size_dim0 = 4;
                    mha_pl0_0_weight_depth_dim_0 = 2;
                    mha_pl0_0_weight_depth_dim_1 = 2;
                    mha_pl0_0_weight_depth_mult = 4;
                    mha_pl0_0_block_per_head = 2;
                    mha_pl0_0_q_depth_dim_0 = 2;
                    mha_pl0_0_q_depth_dim_1 = 2;
                    mha_pl0_0_q_depth_mult = 4;
                    mha_pl0_0_weight_out_depth_dim_1 = 2;
                    counter_1_max_input_counter = 4;
                    counter_1_max_output_counter = 4;

        end

        // 3'b001:     begin mha_pl0_0_data_in_0_depth_dim_1 = 4;
        //             mha_pl0_0_weight_tensor_size_dim0 = 8;
        //             mha_pl0_0_weight_depth_dim_0 = 4;
        //             mha_pl0_0_weight_depth_dim_1 = 4;
        //             mha_pl0_0_weight_depth_mult = 16;
        //             mha_pl0_0_block_per_head = 4;
        //             mha_pl0_0_q_depth_dim_0 = 4;
        //             mha_pl0_0_q_depth_dim_1 = 4;
        //             mha_pl0_0_q_depth_mult = 16;
        //             mha_pl0_0_weight_out_depth_dim_1 = 4;
        // end

        // 3'b010:     begin mha_pl0_0_data_in_0_depth_dim_1 = 8;
        //             mha_pl0_0_weight_tensor_size_dim0 = 16;
        //             mha_pl0_0_weight_depth_dim_0 = 8;
        //             mha_pl0_0_weight_depth_dim_1 = 8;
        //             mha_pl0_0_weight_depth_mult = 64;
        //             mha_pl0_0_block_per_head = 8;
        //             mha_pl0_0_q_depth_dim_0 = 8;
        //             mha_pl0_0_q_depth_dim_1 = 8;
        //             mha_pl0_0_q_depth_mult = 64;
        //             mha_pl0_0_weight_out_depth_dim_1 = 8;
        // end

        // 3'b011:     begin mha_pl0_0_data_in_0_depth_dim_1 = 16;
        //             mha_pl0_0_weight_tensor_size_dim0 = 32;
        //             mha_pl0_0_weight_depth_dim_0 = 16;
        //             mha_pl0_0_weight_depth_dim_1 = 16;
        //             mha_pl0_0_weight_depth_mult = 256;
        //             mha_pl0_0_block_per_head = 16;
        //             mha_pl0_0_q_depth_dim_0 = 16;
        //             mha_pl0_0_q_depth_dim_1 = 16;
        //             mha_pl0_0_q_depth_mult = 256;
        //             mha_pl0_0_weight_out_depth_dim_1 = 8;
        // end

        default: begin mha_pl0_0_data_in_0_depth_dim_1 = 1024;
                    mha_pl0_0_weight_tensor_size_dim0 = 64*8;
                    mha_pl0_0_weight_depth_dim_0 = 64*8/32;
                    mha_pl0_0_weight_depth_dim_1 = 1;
                    mha_pl0_0_weight_depth_mult = 64*8/32;
                    mha_pl0_0_block_per_head = 2;
                    mha_pl0_0_q_depth_dim_0 = 2;
                    mha_pl0_0_q_depth_dim_1 = 2;
                    mha_pl0_0_q_depth_mult = 4;
                    mha_pl0_0_weight_out_depth_dim_1 = 2;
                    counter_1_max_input_counter = 4;
                    counter_1_max_output_counter = 4;
                    
        end

    endcase
        //2nd Layer Norm
        case (current_instr[2])
        3'b000: begin    layer_norm_pl0_1_n_iters = 4;
                    layer_norm_pl0_1_inv_numvalues_0 = (1<<LAYER_NORM_PL0_1_ACC_OUT_WIDTH)/8;
                    layer_norm_pl0_1_inv_numvalues_1 = (1<<LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
                    counter_2_max_input_counter = 4;
                    counter_2_max_output_counter = 4;
                    
        end
        // 3'b001: begin    layer_norm_pl0_1_n_iters = 8;
        //             layer_norm_pl0_1_inv_numvalues_0 = (1<<LAYER_NORM_PL0_1_ACC_OUT_WIDTH)/16;
        //             layer_norm_pl0_1_inv_numvalues_1 = (1<<LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_OUT_WIDTH)/16;
        // end

        // 3'b010: begin    layer_norm_pl0_1_n_iters = 12;
        //             layer_norm_pl0_1_inv_numvalues_0 = (1<<LAYER_NORM_PL0_1_ACC_OUT_WIDTH)/8;
        //             layer_norm_pl0_1_inv_numvalues_1 = (1<<LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
        // end

        // 3'b011: begin    layer_norm_pl0_1_n_iters = 16;
        //             layer_norm_pl0_1_inv_numvalues_0 = (1<<LAYER_NORM_PL0_1_ACC_OUT_WIDTH)/8;
        //             layer_norm_pl0_1_inv_numvalues_1 = (1<<LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
        // end

        // 3'b100: begin    layer_norm_pl0_1_n_iters = 20;
        //             layer_norm_pl0_1_inv_numvalues_0 = (1<<LAYER_NORM_PL0_1_ACC_OUT_WIDTH)/8;
        //             layer_norm_pl0_1_inv_numvalues_1 = (1<<LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
        // end
        default: begin    layer_norm_pl0_1_n_iters = 4;
                    layer_norm_pl0_1_inv_numvalues_0 = (1<<LAYER_NORM_PL0_1_ACC_OUT_WIDTH)/8;
                    layer_norm_pl0_1_inv_numvalues_1 = (1<<LAYER_NORM_PL0_1_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
                    counter_2_max_input_counter = 4;
                    counter_2_max_output_counter = 4;
        end
    endcase

    // Feed Forward
    case (current_instr[3])
        3'b000:     begin linear_pl0_0_data_in_0_depth_dim1 = 2;
                    linear_pl0_0_weight_tensor_size_dim0 = 2;
                    linear_pl0_0_weight_depth_dim0 = 2;
                    linear_pl0_0_weight_depth_dim1 = 2;
                    linear_pl0_0_weight_depth_mult = 4;
                    linear_pl0_1_data_in_0_depth_dim1 = 2;
                    linear_pl0_1_weight_tensor_size_dim0 = 4;
                    linear_pl0_1_weight_depth_dim0 = 2;
                    linear_pl0_1_weight_depth_dim1 = 2;
                    linear_pl0_1_weight_depth_mult = 4;
        end

        3'b001:     begin linear_pl0_0_data_in_0_depth_dim1 = 2;
                    linear_pl0_0_weight_tensor_size_dim0 = 2;
                    linear_pl0_0_weight_depth_dim0 = 2;
                    linear_pl0_0_weight_depth_dim1 = 2;
                    linear_pl0_0_weight_depth_mult = 4;
                    linear_pl0_1_data_in_0_depth_dim1 = 2;
                    linear_pl0_1_weight_tensor_size_dim0 = 4;
                    linear_pl0_1_weight_depth_dim0 = 2;
                    linear_pl0_1_weight_depth_dim1 = 2;
                    linear_pl0_1_weight_depth_mult = 4;
        end
        default:  begin linear_pl0_0_data_in_0_depth_dim1 = 2;
                    linear_pl0_0_weight_tensor_size_dim0 = 2;
                    linear_pl0_0_weight_depth_dim0 = 2;
                    linear_pl0_0_weight_depth_dim1 = 2;
                    linear_pl0_0_weight_depth_mult = 4;
                    linear_pl0_1_data_in_0_depth_dim1 = 2;
                    linear_pl0_1_weight_tensor_size_dim0 = 4;
                    linear_pl0_1_weight_depth_dim0 = 2;
                    linear_pl0_1_weight_depth_dim1 = 2;
                    linear_pl0_1_weight_depth_mult = 4;
                    counter_3_max_input_counter = 4;
                    counter_3_max_output_counter = 4;
        end

    endcase

    //1st Buffer
    case(current_instr[4])
        3'b000: begin    
        roll_pl0_0_roll_distance = 2;
        roll_pl0_0_buffer_size = 4;
        end
        3'b001: begin    
        roll_pl0_0_roll_distance = 2;
        roll_pl0_0_buffer_size = 4;
        end

        3'b010: begin    
        roll_pl0_0_roll_distance = 2;
        roll_pl0_0_buffer_size = 4;
        end


        default: begin   
        roll_pl0_0_roll_distance = 2;
        roll_pl0_0_buffer_size = 4; 
        end

    endcase

        //2st Buffer
    case(current_instr[5])
        3'b000: begin    
        roll_pl0_1_roll_distance = 2;
        roll_pl0_1_buffer_size = 4;
                    
        end
        3'b001: begin    
        roll_pl0_1_roll_distance = 2;
        roll_pl0_1_buffer_size = 4;
        end

        3'b010: begin    
        roll_pl0_1_roll_distance = 2;
        roll_pl0_1_buffer_size = 4;
        end

        // 3'b011: begin   
        // end

        // 3'b100: begin    
        // end
        default: begin
        roll_pl0_1_roll_distance = 2;
        roll_pl0_1_buffer_size = 4;    
        end

    endcase
end


assign counters_max[0] = counter_0_counter_max;
assign counters_max[1] = counter_1_counter_max;
assign counters_max[2] = counter_2_counter_max;
assign counters_max[3] = counter_3_counter_max;

assign counter_0 = counter[0];
assign counter_1 = counter[1];
assign instr_0 = instr[0];
assign instr_1 = instr[1];
assign instr_2 = instr[2];
assign instr_3 = instr[3];


endmodule