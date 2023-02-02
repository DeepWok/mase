#include "Vdataflow_linear.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

using std::stoi;

int main(int argc, char **argv, char **env) {
  int i;
  int clk;

  Verilated::commandArgs(argc, argv);
  Vdataflow_linear *top = new Vdataflow_linear;

  // instantiate top level module
  Verilated::traceEverOn(true);
  VerilatedVcdC *tfp = new VerilatedVcdC;
  top->trace(tfp, 99);
  tfp->open("dataflow_linear.vcd");

  // control signals
  top->clk = 1;
  top->rst = 1;

  const int act[4] = {1, 1, 1, 1};
  top->act = {2, 2, 2, 2};
  top->act_valid = 1;

  top->weights = 1;
  top->w_valid = 1;

  // data signal
  top->out_ready = 1;

  for (i = 0; i < 3000; i++) {
    for (clk = 0; clk < 2; clk++) {
      tfp->dump(i * 2 + clk); // unit is is ps
      top->clk = !top->clk;
      top->eval();
    }

    top->rst = (i < 2);

    if (Verilated::gotFinish())
      exit(0);
  }

  tfp->close();
  exit(0);
}
