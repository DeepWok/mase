
module priority_encoder #(
    parameter NUM_INPUT_CHANNELS = 2,
    parameter NUM_OUPUT_CHANNELS = 1,
    parameter NO_INDICIES = 1

)(
    input [NUM_INPUT_CHANNELS-1:0] input_channels,
    output logic [$clog2(NUM_OUPUT_CHANNELS)-1:0] output_channels [NO_INDICIES-1:0]
);
    integer i;
    always @* begin
    output_channels[0] = 0; // default value if 'in' is all 0's
    //LSB Priority
    for (i=2**NUM_INPUT_CHANNELS-1; i>=0; i=i-1)
        if (input_channels[i]) output_channels[0] = i;
    end
endmodule


module index_to_mask #(
    parameter NUM_INPUT_CHANNELS = 4,
    parameter NUM_OUPUT_CHANNELS = 1,
    parameter NO_INDICIES = 1,
    parameter OUTPUT_WIDTH = 4

)(
    input [$clog2(NUM_INPUT_CHANNELS)-1:0] indicies [NO_INDICIES-1:0],
    output logic [OUTPUT_WIDTH-1:0] output_mask
);
    integer i;
    always @* begin
        output_mask = 0;
        for (i=0; i< NO_INDICIES; i=i+1)
            output_mask[indicies[i]] = 1'b1;
    end


endmodule
