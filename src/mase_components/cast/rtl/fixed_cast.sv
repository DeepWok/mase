`timescale 1ns / 1ps
module fixed_cast #(
    parameter IN_SIZE = 8,
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 4,
    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 4
) (
    input  logic [ IN_WIDTH-1:0] data_in [IN_SIZE-1:0],
    output logic [OUT_WIDTH-1:0] data_out[IN_SIZE-1:0]
);

  // TODO: Negative frac_width is not supported

  localparam IN_INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH;
  localparam OUT_INT_WIDTH = OUT_WIDTH - OUT_FRAC_WIDTH;

  // Sign
  for (genvar i = 0; i < IN_SIZE; i++) begin : out_sign
    logic data;
    assign data = data_in[i][IN_WIDTH-1];
  end

  // Fraction part
  for (genvar i = 0; i < IN_SIZE; i++) begin : out_frac
    logic [OUT_FRAC_WIDTH-1:0] data;
    if (IN_FRAC_WIDTH > OUT_FRAC_WIDTH)
      assign data = data_in[i][IN_FRAC_WIDTH-1:IN_FRAC_WIDTH-OUT_FRAC_WIDTH];
    /* verilator lint_off WIDTH */
    else
      assign data = data_in[i][IN_FRAC_WIDTH-1:0] << (OUT_FRAC_WIDTH - IN_FRAC_WIDTH);
    /* verilator lint_on WIDTH */
  end

  // Integer part
  for (genvar i = 0; i < IN_SIZE; i++) begin : out_int
    logic [OUT_INT_WIDTH-2:0] data;
    if (IN_INT_WIDTH > OUT_INT_WIDTH)
      assign data = data_in[i][OUT_INT_WIDTH-2+IN_FRAC_WIDTH:IN_FRAC_WIDTH];
    else
      assign data = {
        {(OUT_INT_WIDTH - IN_INT_WIDTH) {data_in[i][IN_WIDTH-1]}},
        data_in[i][IN_WIDTH-2:IN_FRAC_WIDTH]
      };
  end

  for (genvar i = 0; i < IN_SIZE; i++) begin

    if (IN_INT_WIDTH > OUT_INT_WIDTH) begin
      always_comb begin
        // Saturation check
        if (|({(IN_WIDTH-OUT_INT_WIDTH-IN_FRAC_WIDTH){data_in[i][IN_WIDTH-1]}} ^ data_in[i][IN_WIDTH-2:OUT_INT_WIDTH-1+IN_FRAC_WIDTH])) begin
          /* saturate to b'100...001 or b' 011..111*/
          data_out[i] = {out_sign[i].data, {(OUT_WIDTH - 2) {~data_in[i][IN_WIDTH-1]}}, 1'b1};
        end else begin
          data_out[i] = {out_sign[i].data, out_int[i].data, out_frac[i].data};
        end
      end
    end else begin
      assign data_out[i] = {out_sign[i].data, out_int[i].data, out_frac[i].data};
    end
  end


endmodule

