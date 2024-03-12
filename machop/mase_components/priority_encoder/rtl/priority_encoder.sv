
module priority_encoder #(
    parameter no_input_channels = 2,
    parameter no_output_channels = 1

)(
    input [$clog2(no_input_channels)-1:0] input_channels,
    output logic [$clog2(no_output_channels)-1:0] output_channels
);
    integer i;
    always @* begin
    output_channels = 0; // default value if 'in' is all 0's
    //LSB Priority
    for (i=2**no_input_channels-1; i>=0; i=i-1)
        if (input_channels[i]) output_channels = i;
    end
endmodule
