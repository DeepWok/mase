class HLSWriter:
    def __init__(self):
        """
        HLS Code generator
        The emitted code has the format:
        {template}
        {type_buff}
        {code_buff}
        """

        self.template = """
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;
"""
        self.type_buff = ""
        self.code_buff = ""
        self.types = []
        self.op_id = 0

    def emit(self, file_name=None):
        """
        Emit HLS code
        """
        if file_name is not None:
            with open(file_name, "w", encoding="utf-8") as outf:
                outf.write(self.template)
                outf.write(self.type_buff)
                outf.write(self.code_buff)

        return self.template + self.type_buff + self.code_buff
