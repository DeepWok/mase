module swin_counter
(
    input logic clk,
    input logic rst, 

    input logic [14:0] max_input_counter,
    input logic [14:0] max_output_counter,

    input logic data_in_0_valid,
    input logic data_in_0_ready,
    input logic data_out_0_ready,
    input logic data_out_0_valid,

    output logic input_ready,
    output logic counter_max
);

logic [14:0] input_counter;
logic [14:0] output_counter;

always_ff @(posedge clk)
begin
    if(rst)
        input_counter <= 0; 
    else if (input_counter == max_input_counter && output_counter == max_output_counter)
        input_counter <= 0;
    else if (data_in_0_valid && data_in_0_ready)
        input_counter <= input_counter+1;
        
end

always_ff @(posedge clk)
begin
    if(rst)    
        output_counter <= 0;
    else if (input_counter == max_input_counter && output_counter == max_output_counter)
        output_counter <= 0;
    else if (data_out_0_valid && data_out_0_ready)
        output_counter <= output_counter +1;

end


always_comb begin
    if(input_counter == max_input_counter)
        input_ready = 0;
    else
        input_ready = 1;
end

always_comb begin
    if(output_counter == max_output_counter)
        counter_max = 1;
    else
        counter_max = 0;
end

endmodule