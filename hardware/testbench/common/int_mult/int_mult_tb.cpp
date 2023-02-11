// This test bench tests the functionality of a fixed-point multiplier
#include "verilated.h"
#include "verilated_vcd_c.h"

#include "Vint_mult.h"

#include <iostream>

using std::stoi;

int main(int argc, char **argv, char **env) {
  Verilated::commandArgs(argc, argv);
  std::srand(9);

  // Instantiate top level module
  Vint_mult *top = new Vint_mult;

  Verilated::traceEverOn(false);

  unsigned int long a = rand(), b = rand();
  for (int i = 0; i < 100; i++) {
    top->data_a = a;
    top->data_b = b;
    top->eval();
    if (Verilated::gotFinish())
      exit(0);
    if (top->product != a * b) {
      std::cout << "Error: a = " << a << ", b = " << b
                << ", product = " << top->product << ", expected = " << a * b
                << std::endl;
      exit(1);
    }
  }

  delete top;
  exit(0);
}
