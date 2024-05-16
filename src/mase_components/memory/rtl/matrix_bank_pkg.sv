package matrix_bank_pkg;

  parameter AXI_ADDRESS_WIDTH = 32;
  parameter MAX_DIMENSION = 1024;

  typedef struct packed {
    logic [AXI_ADDRESS_WIDTH-1:0]   start_address;
    logic [$clog2(MAX_DIMENSION):0] columns;
    logic [$clog2(MAX_DIMENSION):0] rows;
  } REQ_t;

  typedef struct packed {logic partial;} RESP_t;

  typedef struct packed {
    // Check request payloads match NSB payloads
    logic [$clog2(MAX_FEATURE_COUNT):0] columns;
    logic [$clog2(MAX_FEATURE_COUNT):0] rows;
  } ROW_CHANNEL_REQ_t;

  typedef struct packed {
    logic [MAX_FEATURE_COUNT-1:0][31:0] data;
    logic [MAX_FEATURE_COUNT-1:0]       valid_mask;
    logic                               done;
  } ROW_CHANNEL_RESP_t;

endpackage
