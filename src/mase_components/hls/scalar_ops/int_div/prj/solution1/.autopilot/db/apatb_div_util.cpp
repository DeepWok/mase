#include <iostream>
#include "hls_stream.h"

using namespace std;

struct __cosim_T_2__ {char data[2];};
extern "C" void fpga_fifo_push_2(__cosim_T_2__* val, hls::stream<__cosim_T_2__>* fifo) {
  fifo->write(*val);
}
extern "C" void fpga_fifo_pop_2(__cosim_T_2__* val, hls::stream<__cosim_T_2__>* fifo) {
  *val = fifo->read();
}
extern "C" bool fpga_fifo_not_empty_2(hls::stream<__cosim_T_2__>* fifo) {
  return !fifo->empty();
}
extern "C" bool fpga_fifo_exist_2(hls::stream<__cosim_T_2__>* fifo) {
  return fifo->exist();
}
struct __cosim_T_4__ {char data[4];};
extern "C" void fpga_fifo_push_4(__cosim_T_4__* val, hls::stream<__cosim_T_4__>* fifo) {
  fifo->write(*val);
}
extern "C" void fpga_fifo_pop_4(__cosim_T_4__* val, hls::stream<__cosim_T_4__>* fifo) {
  *val = fifo->read();
}
extern "C" bool fpga_fifo_not_empty_4(hls::stream<__cosim_T_4__>* fifo) {
  return !fifo->empty();
}
extern "C" bool fpga_fifo_exist_4(hls::stream<__cosim_T_4__>* fifo) {
  return fifo->exist();
}
