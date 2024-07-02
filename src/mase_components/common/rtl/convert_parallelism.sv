`timescale 1ns / 1ps
module convert_parallelism #(
    parameter DATA_WIDTH = 8,
    parameter DATA_IN_PARALLELISM = 4,
    parameter DATA_OUT_PARALLELISM = 4
) (
    input clk,
    input rst,

    input  logic [DATA_WIDTH-1:0] data_in      [DATA_IN_PARALLELISM-1:0],
    input  logic                  data_in_valid,
    output logic                  data_in_ready,

    output logic [DATA_WIDTH-1:0] data_out      [DATA_OUT_PARALLELISM-1:0],
    output logic                  data_out_valid,
    input  logic                  data_out_ready
);
  // if (DATA_OUT_PARALLELISM == DATA_IN_PARALLELISM) begin    
  //     always_comb begin
  //         for (int i = 0; i < DATA_OUT_PARALLELISM; i++)
  //         begin
  //             data_out[i] = data_in[i];
  //         end
  //         data_out_valid = data_in_valid;
  //         data_in_ready = data_out_ready;
  //     end
  // end else 
  if (DATA_OUT_PARALLELISM > DATA_IN_PARALLELISM) begin

    // How many cycles we need to transfer the input to the output module.
    // If the output parallelism is larger than the input, we must spend several
    // cycles feeding input into the out.
    // Conversely, if the input parallelism is larger, we must feed the input
    // to the output slowly over several cycles.
    localparam TRANSFER_CYCLES = DATA_OUT_PARALLELISM / DATA_IN_PARALLELISM;
    logic [$clog2(TRANSFER_CYCLES):0] count;
    // assign data_in_ready = data_out_ready; 

    always_ff @(posedge clk) begin
      if (rst) begin
        count <= TRANSFER_CYCLES;
        data_out_valid <= 0;
        data_in_ready <= 1;
      end else begin
        if (data_out_ready && data_in_valid) begin

          for (int s = 0; s < TRANSFER_CYCLES; s++) begin
            for (int i = 0; i < DATA_IN_PARALLELISM; i++) begin
              if ((count - 1) == s) begin
                data_out[i+s*DATA_IN_PARALLELISM] <= data_in[i];
              end else begin
                data_out[i+s*DATA_IN_PARALLELISM] <= data_out[i+s*DATA_IN_PARALLELISM];
              end
            end
          end
          data_in_ready <= 1;

          if (count == 1) begin
            data_out_valid <= 1;
            count <= TRANSFER_CYCLES;
          end else begin
            data_out_valid <= 0;
            count <= count - 1;
          end
        end else begin
          data_in_ready <= data_out_ready;
          if (data_out_ready && data_out_valid) begin
            data_out_valid <= 0;
          end else begin
            data_out_valid <= data_out_valid;
          end
          count <= count;
        end
      end
    end

  end else if (DATA_OUT_PARALLELISM <= DATA_IN_PARALLELISM) begin

    logic [DATA_WIDTH-1:0] store[DATA_IN_PARALLELISM-1:0];
    logic store_valid;
    // How many cycles we need to transfer the input to the output module.
    // If the output parallelism is larger than the input, we must spend several
    // cycles feeding input into the out.
    // Conversely, if the input parallelism is larger, we must feed the input
    // to the output slowly over several cycles.
    localparam TRANSFER_CYCLES = DATA_IN_PARALLELISM / DATA_OUT_PARALLELISM;
    logic [$clog2(TRANSFER_CYCLES):0] count;

    // We can only accept new input when:
    // 1. The input is valid.
    // 2. Our store will be empty next cycle.
    assign data_in_ready = !store_valid;

    always_ff @(posedge clk) begin
      if (rst) begin
        count <= TRANSFER_CYCLES;
        data_out_valid <= 0;
        store_valid <= 0;
      end else begin

        // Update store when new input comes.  
        if (data_out_ready && data_in_valid && (!store_valid)) begin
          for (int i = 0; i < DATA_IN_PARALLELISM; i++) begin
            store[i] <= data_in[i];
          end
          store_valid <= 1;
          data_out_valid <= 0;
        end else if (data_out_ready && store_valid) begin

          for (int i = 0; i < DATA_OUT_PARALLELISM; i++) begin
            data_out[i] <= store[i+(count-1)*DATA_OUT_PARALLELISM];
          end
          data_out_valid <= 1;

          if (count == 1) begin
            count <= TRANSFER_CYCLES;
            store_valid <= 0;
          end else begin
            count <= count - 1;
          end
        end else if (!store_valid) begin

          data_out_valid <= 0;
        end

        // if (data_out_ready && (data_in_valid | store_valid)) begin

        //     for (int i = 0; i < DATA_OUT_PARALLELISM; i++)
        //     begin
        //         if (count == TRANSFER_CYCLES) begin
        //             data_out[i] <= data_in[i + (count - 1) * DATA_OUT_PARALLELISM];
        //         end else begin
        //             data_out[i] <= store[i + (count - 1) * DATA_OUT_PARALLELISM];
        //         end
        //     end

        //     if (count == 1) begin
        //         count <= TRANSFER_CYCLES;
        //         store_valid <= 0;
        //     end else begin
        //         count <= count - 1;
        //     end
        // end else begin
        //     count <= count;
        // end
      end
    end
  end
endmodule
