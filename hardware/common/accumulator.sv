module accumulator #(
    parameter NUM = 4,
    parameter IN_WIDTH = 32,
    parameter OUT_WIDTH = $clog2(NUM) + IN_WIDTH
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic [ IN_WIDTH-1:0] ind,
    input  logic                 ind_valid,
    output logic                 ind_ready,
    output logic [OUT_WIDTH-1:0] outd,
    output logic                 outd_valid,
    input  logic                 outd_ready
);

  // 1-bit wider so NUM also fits.
  localparam COUNTER_WIDTH = $clog2(NUM);
  logic [COUNTER_WIDTH:0] counter;

  // Sign extension before feeding into the accumulator
  logic [  OUT_WIDTH-1:0] ind_sext;
  assign ind_sext   = {{(OUT_WIDTH - IN_WIDTH) {ind[IN_WIDTH-1]}}, ind};

  /* verilator lint_off WIDTH */
  assign ind_ready  = (counter != NUM) || outd_ready;
  assign outd_valid = (counter == NUM);
  /* verilator lint_on WIDTH */

  // counter
  always_ff @(posedge clk)
    if (rst) counter <= 0;
    else begin
      if (outd_valid) begin
        if (outd_ready) begin
          if (ind_valid) counter <= 1;
          else counter <= 0;
        end
      end else if (ind_valid && ind_ready) counter <= counter + 1;
    end

  // outd 
  always_ff @(posedge clk)
    if (rst) outd <= '0;
    else begin
      if (outd_valid) begin
        if (outd_ready) begin
          if (ind_valid) outd <= ind_sext;
          else outd <= '0;
        end
      end else if (ind_valid && ind_ready) outd <= outd + ind_sext;
    end

endmodule
