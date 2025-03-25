`timescale 1ns / 1ps
module fixed_difflogic_logic_neuron #(
    parameter OP_CODE
) (
    input  wire clk,
    input  wire rst,
    input  wire a,
    input  wire b,
    output reg  res
);

  generate

    case (OP_CODE)
      (4'd0): begin : GEN_ZERO
        always @* begin
          res = 0;
        end
      end
      (4'd1): begin : GEN_AND
        always @* begin
          res = a & b;
        end
      end
      (4'd2): begin : GEN_NOT_IMPLY
        always @* begin
          res = a & ~b;
        end
      end
      (4'd3): begin : GEN_A
        always @* begin
          res = a;
        end
      end
      (4'd4): begin : GEN_NOT_IMPLY_BY
        always @* begin
          res = ~a & b;
        end
      end
      (4'd5): begin : GEN_B
        always @* begin
          res = b;
        end
      end
      (4'd6): begin : GEN_XOR
        always @* begin
          res = a ^ b;
        end
      end
      (4'd7): begin : GEN_OR
        always @* begin
          res = a | b;
        end
      end
      (4'd8): begin : GEN_NOT_OR
        always @* begin
          res = ~(a | b);
        end
      end
      (4'd9): begin : GEN_NOT_XOR
        always @* begin
          res = ~(a ^ b);
        end
      end
      (4'd10): begin : GEN_NOT_B
        always @* begin
          res = ~b;
        end
      end
      (4'd11): begin : GEN_IMPLY_BY
        always @* begin
          res = a | ~b;
        end
      end
      (4'd12): begin : GEN_NOT_A
        always @* begin
          res = ~a;
        end
      end
      (4'd13): begin : GEN_IMPLY
        always @* begin
          res = ~a | b;
        end
      end
      (4'd14): begin : GEN_NOT_AND
        always @* begin
          res = ~(a & b);
        end
      end
      (4'd15): begin : GEN_ONE
        always @* begin
          res = 1;
        end
      end
    endcase

  endgenerate

endmodule
