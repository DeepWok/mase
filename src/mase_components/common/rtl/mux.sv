/*
Module      : mux
Description : This module multiplexes N inputs
*/

`timescale 1ns / 1ps

module mux #(
    parameter  NUM_INPUTS   = 4,
    parameter  DATA_WIDTH   = 32,
    localparam SELECT_WIDTH = $clog2(NUM_INPUTS)
) (
    input  logic [  DATA_WIDTH-1:0] data_in [NUM_INPUTS-1:0],
    input  logic [SELECT_WIDTH-1:0] select,
    output logic [  DATA_WIDTH-1:0] data_out
);

  assign data_out = data_in[select];

endmodule
