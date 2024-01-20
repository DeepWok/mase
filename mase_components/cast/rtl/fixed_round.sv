`timescale 1ns / 1ps

/* verilator lint_off DECLFILENAME */
module fixed_round #(
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 3,
    parameter OUT_WIDTH = 3,
    parameter OUT_FRAC_WIDTH = 1
) (
    input  logic [ IN_WIDTH - 1:0] data_in,
    output logic [OUT_WIDTH - 1:0] data_out
);
  /* verilator lint_on DECLFILENAME */
  localparam IN_INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH;
  localparam OUT_INT_WIDTH = OUT_WIDTH - OUT_FRAC_WIDTH;
  localparam MAX_POS = (1 << (OUT_WIDTH - 1)) - 1;
  localparam MAX_NEG = (1 << (OUT_WIDTH - 1));

  logic [2:0] lsb_below;
  logic [IN_WIDTH - 2:0] input_data;
  logic carry_in, input_sign;
  assign input_sign = data_in[IN_WIDTH-1];
  assign input_data = (input_sign) ? ~(data_in[IN_WIDTH-2:0] - 1) : data_in[IN_WIDTH-2:0];
  /* verilator lint_off SELRANGE */
  always_comb begin
    lsb_below[2] = (IN_FRAC_WIDTH >= OUT_FRAC_WIDTH) ? input_data[IN_FRAC_WIDTH-OUT_FRAC_WIDTH] : 0;
    lsb_below[1] = (IN_FRAC_WIDTH-1 >= OUT_FRAC_WIDTH)  ? input_data[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-1]     : 0;
    lsb_below[0] = (IN_FRAC_WIDTH-2 >= OUT_FRAC_WIDTH)  ? |(input_data[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-2:0]): 0;
  end
  always_comb begin
    if ((IN_FRAC_WIDTH - OUT_FRAC_WIDTH) >= 0)
      casez (lsb_below)
        // positives
        3'b?00:  carry_in = 1'b0;
        3'b?01:  carry_in = 1'b0;
        3'b010:  carry_in = 1'b0;
        3'b110:  carry_in = 1'b1;
        3'b?11:  carry_in = 1'b1;
        default: carry_in = 1'b0;
      endcase
    else carry_in = 1'b0;
  end
  // this data is the rounded data without sign
  // Basically the data is
  // from [IN_WIDTH-1][IN_WIDTH-2 : IN_FRAC_WIDTH-OUT_FRAC_WIDTH][IN_FRAC_WIDTH-OUT_FRAC_WIDTH-1 : 0]
  // To   [IN_WIDTH-2 : IN_FRAC_WIDTH-OUT_FRAC_WIDTH]
  // Then add one bit to get the total number with carry in
  //      [IN_WIDTH-1 : IN_FRAC_WIDTH-OUT_FRAC_WIDTH]

  // enough bit to store rounded_out_data;
  /* verilator lint_off WIDTH */
  logic [OUT_WIDTH + IN_WIDTH:0] rounded_out_data;

  always_comb begin
    if (IN_FRAC_WIDTH >= OUT_FRAC_WIDTH)
      rounded_out_data = input_data[IN_WIDTH-2:IN_FRAC_WIDTH-OUT_FRAC_WIDTH] + carry_in;
    else rounded_out_data = input_data[IN_WIDTH-2:0] << (OUT_FRAC_WIDTH - IN_FRAC_WIDTH);
  end
  /* verilator lint_off UNUSEDSIGNAL */
  logic [OUT_WIDTH + IN_WIDTH:0] comp_rouded_out;
  assign comp_rouded_out = (~rounded_out_data[OUT_WIDTH-2:0] + 1);
  /* verilator lint_on UNUSEDSIGNAL */
  /* verilator lint_on WIDTH */


  // Saturation check
  always_comb begin
    if (input_sign == 0)
      /* verilator lint_off UNSIGNED */
      if (rounded_out_data >= MAX_POS)
        data_out = {input_sign, {(OUT_WIDTH - 1) {1'b1}}};
      else data_out = {input_sign, rounded_out_data[OUT_WIDTH-2:0]};
    else if (rounded_out_data >= MAX_NEG) data_out = {input_sign, {(OUT_WIDTH - 1) {1'b0}}};
    else if (input_data == 0)
      if (IN_INT_WIDTH >= OUT_INT_WIDTH) data_out = {input_sign, {(OUT_WIDTH - 1) {1'b0}}};
      else data_out = {{(OUT_WIDTH - 1) {1'b1}}, 1'b0} << (OUT_FRAC_WIDTH + IN_INT_WIDTH - 2);
    else if (rounded_out_data == 0) data_out = 0;
    else data_out = {input_sign, comp_rouded_out[OUT_WIDTH-2:0]};
  end
  /* verilator lint_on UNSIGNED */
  /* verilator lint_on SELRANGE */
endmodule
