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
    output logic [14:0] max_input_counter,
    output logic [14:0] max_output_counter
);

assign layer_norm_pl0_0_n_iters = 4; 
assign layer_norm_pl0_0_inv_numvalues_0 = (1<<LAYER_NORM_PL0_0_ACC_OUT_WIDTH)/8; 
assign layer_norm_pl0_0_inv_numvalues_1 = (1<<LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH)/8;
assign mha_pl0_0_data_in_0_depth_dim_1 = 2;
assign mha_pl0_0_weight_tensor_size_dim0 = 4; 
assign mha_pl0_0_weight_depth_dim_0 = 2;
assign mha_pl0_0_weight_depth_dim_1 = 2;
assign mha_pl0_0_weight_depth_mult = 4;
assign mha_pl0_0_block_per_head = 2;
assign mha_pl0_0_q_depth_dim_0 = 2;
assign mha_pl0_0_q_depth_dim_1 = 2;
assign mha_pl0_0_q_depth_mult = 4;
assign mha_pl0_0_weight_out_depth_dim_1 = 2;


assign layer_norm_pl0_1_n_iters = 4; 
assign layer_norm_pl0_1_inv_numvalues_0 = (1<<LAYER_NORM_PL0_0_ACC_OUT_WIDTH)/8; 
assign layer_norm_pl0_1_inv_numvalues_1 = (1<<LAYER_NORM_PL0_0_SQUARES_ADDER_TREE_OUT_WIDTH)/8; 
assign linear_pl0_0_data_in_0_depth_dim1 = 2;
assign linear_pl0_0_weight_tensor_size_dim0 = 2;
assign linear_pl0_0_weight_depth_dim0 = 2;
assign linear_pl0_0_weight_depth_dim1 = 2;
assign linear_pl0_0_weight_depth_mult = 4;
assign linear_pl0_1_data_in_0_depth_dim1 = 2;
assign linear_pl0_1_weight_tensor_size_dim0 = 4;
assign linear_pl0_1_weight_depth_dim0 = 2;
assign linear_pl0_1_weight_depth_dim1 = 2;
assign linear_pl0_1_weight_depth_mult = 4;
assign roll_pl0_0_roll_distance = 2;
assign roll_pl0_0_buffer_size = 4;
assign roll_pl0_1_roll_distance = 2;
assign roll_pl0_1_buffer_size = 4;

assign load_input_seq = 1;
assign input_mux_ctrl = 1;
assign output_mux_ctrl = 1;

assign max_input_counter = 15;
assign max_output_counter = 1;



endmodule