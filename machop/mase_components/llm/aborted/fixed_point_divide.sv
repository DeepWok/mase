`timescale 1ns / 1ps


    module divide_fixed_point(
        parameter IN_WIDTH = 16

        )(

        input [IN_WIDTH:0] data_in_1, // 被除数
        input [IN_WIDTH:0] data_in_2, // 除数
        output [IN_WIDTH:0] data_out // 结果
);
    // 假设这里的a和b都是16位的固定点数，其中有8位小数。
    // 实际的取整方式可能需要根据你的小数点位置进行调整。
    // 这里只是一个向最近整数取整的简单示例。

    // 临时变量，用于存储扩展的中间结果以便取整
    logic [2*IN_WIDTH:0] temp_result;

    // 扩展运算以保持精度，添加0.5的偏移量实现四舍五入
    temp_result = (a * 256 + (b / 2)) / b;
    // 将结果转换回16位，假设小数部分在扩展结果的低位
    result = temp_result[23:8]; // 根据实际小数点位置调整


endmodule