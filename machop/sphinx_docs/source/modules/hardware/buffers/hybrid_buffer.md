# Hybrid Buffer

The Hybrid Buffer behaves as an addressable RAM on the write interface but a FIFO on the read interface and contains a parametrizable number of buffer slots. Each buffer slot is implemented using BRAM blocks, to reduce LUT and Flip-Flop usage on the FPGA.

![Hybrid Buffer](https://raw.githubusercontent.com/DeepWok/mase/main/machop/sphinx_docs/source/imgs/hardware/bram_fifo.png)

Additionally, dual-port BRAMs are used, enabling different widths and depths on the write/read interfaces to support different memory bandwidths.