`timescale 1ns / 1ps
module matrix_to_vector #(
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_MAX_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_MAX_TENSOR_SIZE_DIM_1 = 8,
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 8,

    localparam DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    localparam DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1,

    localparam DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    localparam DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,

    localparam DATA_IN_0_MAX_DEPTH_DIM_0 = DATA_IN_0_MAX_TENSOR_SIZE_DIM_0/DATA_IN_0_PARALLELISM_DIM_0,
    localparam DATA_IN_0_MAX_DEPTH_DIM_1 = DATA_IN_0_MAX_TENSOR_SIZE_DIM_1/DATA_IN_0_PARALLELISM_DIM_1,
    localparam DATA_IN_0_MAX_DEPTH_DIM_0_WIDTH = $clog2(DATA_IN_0_MAX_DEPTH_DIM_0),
    localparam DATA_IN_0_MAX_DEPTH_DIM_1_WIDTH = $clog2(DATA_IN_0_MAX_DEPTH_DIM_1),

    localparam SLICED_INPUT_SIZE = DATA_IN_0_PARALLELISM_DIM_0,
    localparam FIFO_INPUT_SIZE = (DATA_IN_0_PARALLELISM_DIM_0-1) * DATA_IN_0_PARALLELISM_DIM_1,
    localparam COUNTER_WIDTH = $clog2(DATA_IN_0_MAX_DEPTH_DIM_0*DATA_IN_0_MAX_TENSOR_SIZE_DIM_1)
)
(

    input clk,
    input rst,

    input logic [DATA_IN_0_MAX_DEPTH_DIM_0_WIDTH:0] data_in_0_depth_dim0,
    
    //should be equal to depth0 * parallelism1
    input logic [COUNTER_WIDTH:0] counter_max,

    //probably depth or size will be required
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

enum int unsigned{
    IDLE,
    LOAD,
    PROCESS,
    LAST,
    FINISHED
} 
state, next;

logic [COUNTER_WIDTH:0] counter;
logic [DATA_OUT_0_PRECISION_0-1:0] fifo_out [DATA_IN_0_PARALLELISM_DIM_0 * (DATA_IN_0_PARALLELISM_DIM_1-1)-1:0];
//logic [DATA_OUT_0_PRECISION_0-1:0] fifo_in [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1-1:0];

logic [DATA_OUT_0_PRECISION_0-1:0] data_in_0_out [DATA_IN_0_PARALLELISM_DIM_0-1:0];
logic [DATA_OUT_0_PRECISION_0-1:0] data_in_0_fifo [DATA_IN_0_PARALLELISM_DIM_0 * (DATA_IN_0_PARALLELISM_DIM_1-1)-1:0];
logic [DATA_OUT_0_PRECISION_0-1:0] fifo_in [DATA_IN_0_PARALLELISM_DIM_0 * (DATA_IN_0_PARALLELISM_DIM_1-1)-1:0];

logic fifo_in_ready;
logic fifo_in_valid;
logic fifo_out_valid;
logic fifo_out_ready;
logic ctrl_output_fifo;
logic ctrl_run_counter;
logic ctrl_input_fifo;
logic ctrl_data_out_valid;
logic data_in_0_valid_delayed; 
logic data_in_0_ready_delayed;
logic fifo_out_valid_delayed;


matrix_fifo #(
    .DATA_WIDTH(DATA_IN_0_PRECISION_0),
    .DIM0(DATA_IN_0_PARALLELISM_DIM_0),
    .DIM1(DATA_IN_0_PARALLELISM_DIM_1-1),
    .FIFO_SIZE(DATA_IN_0_MAX_DEPTH_DIM_0+1)
) m_fifo
(
    .clk(clk),
    .rst(rst),
    .in_data(fifo_in),
    .in_valid(fifo_in_valid),
    .in_ready(fifo_in_ready),
    .out_data(fifo_out),
    .out_valid(fifo_out_valid),
    .out_ready(fifo_out_ready)
);


//split incoming data into fifo input, data_output
always_comb begin
    data_in_0_out = data_in_0 [DATA_IN_0_PARALLELISM_DIM_0-1:0];
    data_in_0_fifo = data_in_0 [DATA_IN_0_PARALLELISM_DIM_0* DATA_IN_0_PARALLELISM_DIM_1-1:DATA_IN_0_PARALLELISM_DIM_0];
end

//direct the correct signal to the output
always_ff @(posedge clk) begin
    if (ctrl_output_fifo)
        data_out_0 <= fifo_out[DATA_IN_0_PARALLELISM_DIM_0-1:0];
    else 
        data_out_0 <= data_in_0_out;
end

//direct the correct signal to fifo input
always_comb begin 
    if (ctrl_input_fifo) begin
        fifo_in = data_in_0_fifo;
    end
    else
        for (int i=0; i<DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1; i++) begin
                fifo_in[i] = fifo_out[i+DATA_IN_0_PARALLELISM_DIM_0];
        end
end


always_ff@(posedge clk) begin
    if (rst)
        counter <= 0;
    else if (counter == counter_max) 
        counter <= 0;
    else if (fifo_activity)
        counter <= counter + 1;
    else 
        counter <= counter;
end

always_ff@(posedge clk) begin
    fifo_out_valid_delayed <= fifo_out_valid;
end

always_ff@(posedge clk) begin
    data_in_0_valid_delayed <= data_in_0_valid;
    data_in_0_ready_delayed <= data_in_0_ready;
end


always_comb begin
    if(data_in_0_ready_delayed && data_in_0_valid_delayed)
        data_out_0_valid = 1;
    else if(ctrl_data_out_valid)
        data_out_0_valid = fifo_out_valid_delayed;
    else
        data_out_0_valid = 0;
end


//state machine blocks

always_comb begin
    next = state;
    case(state)
        IDLE:     if (data_in_0_valid && data_in_0_ready) begin
                        if (data_in_0_depth_dim0 == 1)                                                         next = PROCESS;
                        else                                                                                   next = LOAD; 
                  end
        LOAD:     if (counter == (counter_max - data_in_0_depth_dim0 - 1) && fifo_activity)                    next = LAST;     
                  else if (counter >= (data_in_0_depth_dim0-1) && fifo_activity)                               next = PROCESS;
                  
        PROCESS:  if (counter == counter_max - data_in_0_depth_dim0)                                           next = LAST;
        LAST:     if (counter == counter_max)                                                                  next = FINISHED;
        FINISHED: if (data_in_0_valid && data_in_0_ready) begin
                        if (data_in_0_depth_dim0 == 1)                                                         next = PROCESS;
                        else                                                                                   next = LOAD;
                  end
                  else                                                                                         next = IDLE;                                                                               
    endcase
end


always_comb  begin
    case(state)
        IDLE: begin       data_in_0_ready  = fifo_in_ready;
                          fifo_out_ready      = 0;
                          ctrl_output_fifo    = 0;
                          ctrl_run_counter    = 1;
                          ctrl_input_fifo     = 1;
                          ctrl_data_out_valid = 0;
                          fifo_in_valid       = data_in_0_valid;
        end
        LOAD: begin       data_in_0_ready     = fifo_in_ready;
                          fifo_out_ready      = 0;
                          ctrl_output_fifo    = 0;
                          ctrl_run_counter    = 1;
                          ctrl_input_fifo     = 1;
                          ctrl_data_out_valid = 0;
                          fifo_in_valid       = data_in_0_valid;
                        
        end
        PROCESS: begin    data_in_0_ready     = 0;
                          fifo_out_ready      = fifo_in_ready;
                          ctrl_output_fifo    = 1;
                          ctrl_run_counter    = 1;
                          ctrl_input_fifo     = 0;
                          ctrl_data_out_valid = 1;
                          fifo_in_valid       = fifo_out_valid;
        end
        LAST: begin       data_in_0_ready     = 0;
                          fifo_out_ready      = fifo_in_ready;
                          ctrl_output_fifo    = 1;
                          ctrl_run_counter    = 1;
                          ctrl_input_fifo     = 1;
                          ctrl_data_out_valid = 1;
                          fifo_in_valid       = 0;
        end
        FINISHED: begin   data_in_0_ready     = 1;
                          fifo_out_ready      = 1;
                          ctrl_output_fifo    = 0;
                          ctrl_run_counter    = 1;
                          ctrl_input_fifo     = 1;
                          ctrl_data_out_valid = 1;
                          fifo_in_valid       = data_in_0_valid;
                          
        end
    endcase
end

always_ff @(posedge clk) begin
    if(rst) state <= IDLE;
    else    state <= next;
end



assign fifo_activity = (fifo_in_ready && fifo_in_valid) || (fifo_out_ready && fifo_out_valid);

endmodule