// Join2 synchronises two sets of input handshake signals with a set of output handshaked signals 
module join2 #(
) (
    input logic [1:0] in_valid,
    output logic [1:0] in_ready,
    output logic out_valid,
    input logic out_ready
);

  // If only one of the inputs is valid - we need to stall that input and wait
  // for the other input by setting one of the ready bit to 0.
  // +-----------+-----------+------------+------------+------------+
  // | out_ready | invalid_0 | in_valid_1 | in_ready_0 | in_ready_1 |
  // +-----------+-----------+------------+------------+------------+
  // |         0 |         0 |          0 |          0 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         0 |         0 |          1 |          0 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         0 |         1 |          0 |          0 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         0 |         1 |          1 |          0 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         1 |         0 |          0 |          1 |          1 |
  // +-----------+-----------+------------+------------+------------+
  // |         1 |         0 |          1 |          1 |          0 |
  // +-----------+-----------+------------+------------+------------+
  // |         1 |         1 |          0 |          0 |          1 |
  // +-----------+-----------+------------+------------+------------+
  // |         1 |         1 |          1 |          1 |          1 |
  // +-----------+-----------+------------+------------+------------+
  assign in_ready[0] = out_ready & (!in_valid[0] | in_valid[1]);
  assign in_ready[1] = out_ready & (!in_valid[1] | in_valid[0]);
  assign out_valid   = &in_valid;

endmodule
