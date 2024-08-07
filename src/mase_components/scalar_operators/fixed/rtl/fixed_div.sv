// Int div 

module ALU_intDiv
(
        clk : input logic,
        reset : input logic,
        pvalid : input logic,
        nready : input logic,
        valid, ready : output logic,
        din0 : input logic [31 : 0],
        din1 : input logic [31 : 0],
        dout : output logic[31 : 0]
);

    localparam st = 28;
    logic [st-2 downto 0] q;
    logic ce;

begin

    fop_fadd_32ns_32ns_32_5_full_dsp_1
    # (
        .ID ( 1),
        .NUM_STAGE ( st),
        .din0_WIDTH ( 32),
        .din1_WIDTH ( 32),
        .dout_WIDTH ( 32))
    fop_fadd_32ns_32ns_32_5_full_dsp_1_U1  (
        .clk ( clk),
        .reset ( reset),
        .din0 ( din0),
        .din1 ( din1),
        .ce ( ce),
        .dout ( dout));

    assign ready = ce;
    assign ce = nReady | (! q(st-2));



always @(posedge clk)
begin
         if (reset = '1') then
             q <= (others => '0');
             q(0) <= pvalid;
         elsif (ce='1') then
             q(0) <= pvalid;
             q(st-2 downto 1) <= q(st-3 downto 0);
          end if;
end
    assign valid = q(st-2);

endmodule

module fadd_op # (
  INPUTS = 2, OUTPUTS= 1, DATA_SIZE_IN = 32, DATA_SIZE_OUT = 32
)
(
        clk, rst : input logic; 
        dataInArray : input logic [2*DATA_SIZE_input-1 : 0 ]; 
        dataOutArray : inout logic [DATA_SIZE_OUT-1 : 0 ];      
        pValidArray : input logic [1 : 0 ];
        nReadyArray : input logic [0 : 0 ];
        validArray : inout logic [0 : 0 ];
        readyArray : inout logic [1 : 0 ] );

logic logic  [DATA_SIZE_OUT-1 downto 0] alu_out;
logic alu_valid, alu_ready;

logic join_valid;
logic [1:0] join_ReadyArray;

logic buff_ready;

    join2 #() 
(
.data_in_valid(pValidArray),
.data_in_ready(alu_ready),
.data_out_valid(join_valid),
.data_out_ready(readyArray));

     ALU_intDiv #()
            (clk, rst,
                     join_valid,
                     buff_ready,
                     alu_valid,
                     alu_ready,
                     dataInArray[DATA_SIZE_IN-1 : 0],
                     dataInArray[2*DATA_SIZE_IN-1 : DATA_SIZE_IN], 
                     alu_out);

    register_slice #(...)
    out_reg (
            clk => clk,
            rst => rst,
            dataInArray(DATA_SIZE_IN-1 downto 0) => alu_out,
            pValidArray(0) => alu_valid,
            readyArray(0) => buff_ready,
            nReadyArray(0) => nReadyArray(0),
            validArray(0) => validArray(0),
            dataOutArray => dataOutArray
    );

endmodule
