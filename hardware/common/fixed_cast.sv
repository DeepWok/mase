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

  // Split variable
  for (genvar i = 0; i < IN_SIZE; i++) begin : cast_vars
    assign data_out[i][OUT_WIDTH-1] = data_in[i][IN_WIDTH-1];
  end

  // Saturation check
  if (IN_INT_WIDTH > OUT_INT_WIDTH) begin
    for (genvar i = 0; i < IN_SIZE; i++) begin : int_trunc
      always_comb begin
        if (|data_in[i][IN_WIDTH-2:OUT_INT_WIDTH-2+IN_FRAC_WIDTH]) data_out[i][OUT_WIDTH-2:0] = '1;
        else begin
          data_out[i][OUT_WIDTH-2:OUT_FRAC_WIDTH] = data_in[i][OUT_INT_WIDTH-2+IN_FRAC_WIDTH:IN_FRAC_WIDTH];
          /* verilator lint_off WIDTH */
          if (IN_FRAC_WIDTH > OUT_FRAC_WIDTH)
            data_out[i][OUT_FRAC_WIDTH-1:0] = data_in[i][IN_FRAC_WIDTH-1:IN_FRAC_WIDTH-OUT_FRAC_WIDTH];
          else
            data_out[i][OUT_FRAC_WIDTH-1:0]  = data_in[i][IN_FRAC_WIDTH-1:0] << (OUT_FRAC_WIDTH-IN_FRAC_WIDTH);
          /* verilator lint_on WIDTH */
        end
      end
    end
  end else begin
    for (genvar i = 0; i < IN_SIZE; i++) begin : int_trunc
      always_comb begin
        /* verilator lint_off SELRANGE */
        /* verilator lint_off WIDTH */
        data_out[i][OUT_WIDTH-2:OUT_FRAC_WIDTH] = data_in[i][IN_WIDTH-2:IN_FRAC_WIDTH];
        if (IN_FRAC_WIDTH > OUT_FRAC_WIDTH)
          data_out[i][OUT_FRAC_WIDTH-1:0] = data_in[i][IN_FRAC_WIDTH-1:IN_FRAC_WIDTH-OUT_FRAC_WIDTH];
        else
          data_out[i][OUT_FRAC_WIDTH-1:0]  = data_in[i][IN_FRAC_WIDTH-1:0] << (OUT_FRAC_WIDTH-IN_FRAC_WIDTH);
        /* verilator lint_on WIDTH */
        /* verilator lint_on SELRANGE */
      end
    end
  end

endmodule

