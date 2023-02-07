module accumulator #(
    parameter NUM = 2,
    parameter IN_WIDTH = 32,
    parameter OUT_WIDTH = $clog2(NUM) + IN_WIDTH
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic [ IN_WIDTH-1:0] in,
    input  logic                 in_valid,
    output logic                 in_ready,
    output logic [OUT_WIDTH-1:0] out,
    output logic                 out_valid,
    input  logic                 out_ready
);

  localparam COUNTER_WIDTH = $clog2(NUM);

  // Sign extension before feeding into the accumulator
  logic [OUT_WIDTH-1:0] in_sext;
  assign in_sext = {{(OUT_WIDTH - IN_WIDTH) {in[IN_WIDTH-1]}}, in};

  // 1-bit wider so NUM also fits.
  logic [COUNTER_WIDTH:0] counter;
  assign in_ready  = counter != NUM || out_ready;
  assign out_valid = counter == NUM;

  always_ff @(posedge clk)
    if (rst) begin
      counter <= 0;
      out <= '0;
    end else begin
      if (out_valid) begin
        if (out_ready) begin
          if (in_valid) begin
            out <= in_sext;
            counter <= 1;
          end else begin
            out <= '0;
            counter <= 0;
          end
        end
      end else if (in_valid && in_ready) begin
        out <= out + in_sext;
        counter <= counter + 1;
      end
    end

endmodule
