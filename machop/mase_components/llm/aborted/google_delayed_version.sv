module data_synchronizer #(
    parameter DATA1_WIDTH = 32,  // data1的宽度
    parameter DATA2_WIDTH = 16,  // data2的宽度
    parameter FIFO_DEPTH = 4     // FIFO缓冲区的深度
)(
    input logic clk,             // 时钟信号
    input logic reset_n,         // 低电平复位
    input logic valid_data1,     // data1有效信号
    input logic [DATA1_WIDTH-1:0] data1, // 32位输入data1
    input logic valid_data2,     // data2有效信号
    input logic [DATA2_WIDTH-1:0] data2, // 16位输入data2
    output logic ready,          // 准备好的信号，表明data1和data2已同步
    output logic [DATA1_WIDTH-1:0] out_data1, // 输出同步的data1
    output logic [DATA2_WIDTH-1:0] out_data2  // 输出同步的data2
);

// 用于缓冲data2的FIFO
logic [DATA2_WIDTH-1:0] fifo[FIFO_DEPTH-1:0];
logic [$clog2(FIFO_DEPTH)-1:0] write_ptr, read_ptr;
logic fifo_full, fifo_empty;

always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        write_ptr <= 0;
        read_ptr <= 0;
        fifo_full <= 0;
        fifo_empty <= 1;
    end else begin
        if (valid_data2 && !fifo_full) begin
            fifo[write_ptr] <= data2;
            write_ptr <= write_ptr + 1;
            fifo_empty <= 0;
        end
        
        if (valid_data1 && !fifo_empty) begin
            out_data2 <= fifo[read_ptr];
            read_ptr <= read_ptr + 1;
            fifo_full <= 0;
            out_data1 <= data1;
            ready <= 1;
        end else begin
            ready <= 0;
        end
        
        // 更新FIFO满/空状态
        if (write_ptr == FIFO_DEPTH-1) fifo_full <= 1;
        if (read_ptr == write_ptr) fifo_empty <= 1;
    end
end

endmodule
