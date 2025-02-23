/*
Module      : channel_selection
Description : This module a double counter which outputs which channel the
              current input compute window belongs to. It is used in the
              normalization hardware.
*/

`timescale 1ns / 1ps

module channel_selection #(
    parameter NUM_CHANNELS       = 2,
    // Number of blocks in spatial dimensions (usually = depth_dim0 * depth_dim1)
    parameter NUM_SPATIAL_BLOCKS = 4,

    // Channel and spatial state widths
    localparam C_STATE_WIDTH = (NUM_CHANNELS == 1) ? 1 : $clog2(NUM_CHANNELS),
    localparam S_STATE_WIDTH = (NUM_SPATIAL_BLOCKS == 1) ? 1 : $clog2(NUM_SPATIAL_BLOCKS)
) (
    input  logic                     clk,
    input  logic                     rst,
    input  logic                     inc,
    output logic [C_STATE_WIDTH-1:0] channel
);

  generate
    if (NUM_CHANNELS == 1) begin
      assign channel = 0;
    end else begin
      logic [C_STATE_WIDTH-1:0] channel_counter;
      logic [S_STATE_WIDTH-1:0] spatial_counter;

      always_ff @(posedge clk) begin
        if (rst) begin
          channel_counter <= 0;
          spatial_counter <= 0;
        end else if (inc) begin
          if (channel_counter == NUM_CHANNELS-1 && spatial_counter == NUM_SPATIAL_BLOCKS-1) begin
            channel_counter <= 0;
            spatial_counter <= 0;
          end else if (spatial_counter == NUM_SPATIAL_BLOCKS - 1) begin
            channel_counter <= channel_counter + 1;
            spatial_counter <= 0;
          end else begin
            spatial_counter <= spatial_counter + 1;
          end
        end
      end

      assign channel = channel_counter;
    end
  endgenerate

endmodule
