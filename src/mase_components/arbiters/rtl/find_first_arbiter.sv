`timescale 1ns / 1ps
module find_first_arbiter #(
    parameter NUM_REQUESTERS = 4
) (
    input        [      NUM_REQUESTERS - 1:0] request,
    output logic [      NUM_REQUESTERS - 1:0] grant_oh,
    output logic [$clog2(NUM_REQUESTERS)-1:0] grant_bin
);

  always_comb begin
    grant_oh  = '0;
    grant_bin = '0;

    for (int i = 0; i < NUM_REQUESTERS; i++) begin
      if (request[i]) begin
        grant_oh  = ({{(NUM_REQUESTERS - 1) {1'b0}}, 1'b1} << i);
        grant_bin = i;
        break;
      end
    end
  end

endmodule
