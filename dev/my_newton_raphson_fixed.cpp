// LIBRARIES
// ---------
#include <cstdint>
#include <cmath>
#include <bits/stdc++.h>
#include <string>
#include <stdlib.h>

// MACROS
// ------
#define ISQRT_2 0b0101101010000010          // FORMAT: Q1.(WIDTH-1)
#define SQRT_2 0b1011010100000100           // FORMAT: Q1.(WIDTH-1)
#define LUT_POW 5                           // LUT_POW = log2(LUT_SIZE)
#define LUT_SIZE 32                         // NOTE: LUT_SIZE must be a power of 2.
#define LUT_STEP (1.0f / (LUT_SIZE + 1.0f)) // FORMAT: float
#define NUM_ITER 1

int i_width = 3;
int f_width = 8;
int w_width = i_width + f_width;
uint16_t max_num = (0b1 << w_width) - 0b1;

// Signatures for utils.
float qxy_to_float(uint64_t x, uint8_t int_width, uint8_t frac_width);
uint64_t float_to_qxy(float x, uint8_t int_width, uint8_t frac_width);
template<int WWIDTH>
void print_bits(std::string label, uint64_t x){
    std::cout << label << ": " << std::bitset<WWIDTH>(x) << "\n";
}

// ===========================================================================
// Newton Raphson method
// ---------------------------------------------------------------------------

// NOTE: Range reduction
// The input is mapped to the range [1, 2).
// The reason for this range is because 1/sqrt(x) contains values between (1/sqrt(2), 1].
// Both input range and output range can be supported by the format Q1.(WIDTH-1).
// Also the format Q1.(WIDTH-1) supports the 1.5f present in the Newton-Raphson 
// equation.
//
// NOTE: this reduced range does not attempt to squash infinity into a small range.
// The process is performed by dividing the input by 2 until it falls in the 
// range [1, 2) or multiplying by 2 until it falls in the range [1, 2). However
// this operation is the same as moving the fixed point behind the MSB of the 
// input number.

// NOTE: the lut values are in the format Q1.(WIDTH-1).
// The reason for this format is because the range of the input is mapped to
// [1, 2) therefore the possible values for 1/sqrt(x) will be (1/sqrt(2), 1].

// FORMAT: Q1.(WIDTH-1)
uint16_t lut[LUT_SIZE];

// FORMAT: Q1.(WIDTH-1)
void init_lut(){
    float x = LUT_STEP + 1.0f;
    for(int i = 0; i < LUT_SIZE; i++){
        // Calculate look up values.
        float ref = 1.0f / sqrt(x);
        // Convert look up values to Q1.(WIDTH-1) format.
        // NOTE: since we only support 16 bit numbers this can be hard coded.
        lut[i] = float_to_qxy(ref, 1, w_width-1);
        x += LUT_STEP;
    }
}

// Finds the MSB of the input number.
// NOTE: index 0 is assumed to be the left most bit within the number.
// FORMAT: Integer
uint16_t find_msb(uint16_t x){

    for(int i = 1; i < w_width + 1; i++){
        bool msb = (x >> (w_width - i)) & 0b1;
        if(msb){
            return w_width - i;
        }
    }

    // In this case the input is 0 and should be rejected by the algorithm.
    return 0xFFFF;
}

// Maps [0, inf) to [1, 2)
// NOTE: this mapping function does not squish infinity into [1, 2) which would
// be a 1:1 mapping. Instead it does a 1:N mapping where multiple values of the
// original range will map to the same value in the new range.
//
// INPUT FORMAT: Q(i_width).(f_width)
// OUTPUT FORMAT: Q1.(WIDTH-1)
uint32_t range_reduction(uint32_t x, uint16_t msb_index){
    // Shifts the input left until the MSB of x is at the leftmost index.
    if(msb_index < w_width - 1){
        return x << (w_width - 1 - msb_index);
    }
    // X is perfect because MSB is at the leftmost position.
    else{
        return x;
    }
}

// NOTE: this is not the opposite to the range_reduction function.
// IDEA: 
// 1 / sqrt(x) --> 1 / sqrt(x * 2^(k))                                  Through range reduction.
// 1 / sqrt(x) = sqrt(2^(k)) * 1 / sqrt(x * 2^(k))                      Through math.
// From this identity we get 4 cases for the combinations of k being even or odd
// and k being positive or negative.
//      Case 1: k is negative and even
//          1 / sqrt(x) = [1 / sqrt(x * 2^(k))] >> (k/2)                Through math.
//      Case 2: k is positive and even
//          1 / sqrt(x) = [1 / sqrt(x * 2^(k))] << (k/2)                Through math.
//      Case 3: k is negative and odd
//          1 / sqrt(x) = sqrt(-1/2) * [1 / sqrt(x * 2^(k))] >> (k/2)   Through math.
//      Case 4: k is positive and odd
//          1 / sqrt(x) = sqrt(1/2) * [1 / sqrt(x * 2^(k))] << (k/2)    Through math.
//
// INPUT FORMAT: Q1.15
// OUTPUT FORMAT: Q(i_width).(f_width)
uint32_t range_augmentation(uint32_t x_red, uint16_t msb_index){
    // Determine whether shifted right or left and by how much relative to the 
    // position of the fixed point in Q(i_width).(f_width)
    // Left shift = Positive
    // Right shift = Negative
    int16_t shifted = f_width - msb_index;

    // Reduction was through multiplication.
    // Therefore augmentation is through multiplication.
    if(shifted > 0){
        if(shifted % 2 == 0){
            int16_t shift = shifted >> 1; // k / 2
            // FORMAT: Q1.15
            int32_t res = x_red << shift;
            // FORMAT Q(i_width).(f_width)
            res = res >> ((w_width - 1) - f_width);
            return res;
        }
        else{
            int16_t shift = (shifted - 1) >> 1; // (k - 1) / 2
            // FORMAT: Q1.15
            int32_t res = (x_red * SQRT_2) >> 15;
            res = res << shift;
            // FORMAT Q(i_width).(f_width)
            res = res >> ((w_width - 1) - f_width);
            return res;
        }
    }
    // Reduction was through division.
    // Therefore augementation is through division.
    else if(shifted < 0){
        // Even shifting.
        if(shifted % 2 == 0){
            int16_t shift = (-shifted) >> 1;
            // FORMAT Q.15
            int32_t res = x_red >> shift;
            // FORMAT Q(i_width).(f_width)
            res = res >> ((w_width - 1) - f_width);
            return res;
        }
        else{
            int16_t shift = (-shifted - 1) >> 1;
            // FORMAT: Q1.15
            int32_t res = (x_red * ISQRT_2) >> 15;
            res = res >> shift;
            // FORMAT Q(i_width).(f_width)
            res = res >> ((w_width - 1) - f_width);
            return res;
        }
    }
    // Reduction was not done.
    // Therefore just convert format.
    else{
        // FORMAT Q(i_width).(f_width)
        return x_red >> ((w_width - 1) - f_width);
    }
}

// Newton Raphson's method.
//
// INPUT FORMAT: Q(i_width).(f_width)
// OUTPUT FORMAT: Q(i_width).(f_width)
uint16_t isqrt(uint16_t x){
    // X = 0 is invalid.
    // TODO: how to handle this for actual implementation? Return 0? Return MAX_NUM?
    if(x == 0){
        std::cout << "[ERROR] The input number x is invalid." << "\n";
        std::cout << "[X] " << x << "\n";
        return max_num;
    }

    // FORMAT: Integer
    uint16_t msb_index = find_msb(x);

    // FORMAT Q1.(WIDTH-1)
    uint32_t x_red = range_reduction(x, msb_index);

    // If X gets mapped to 1 then 1/sqrt(x) is 1. Then augment range.
    if(x_red == (0b1 << (w_width - 1))){
        uint32_t out = range_augmentation(x_red, msb_index);

        // Check that overflow does not occur.
        // TODO: how to deal with overflow in actual implementation? Return MAX_NUM?
        // Have an output wire OVERFLOW?
        if(out > max_num){
            return max_num;
        }
        return out;
    }

    // FORMAT Q17.15
    uint32_t intermediate;
    // Shift the number to match the Q1.(WIDTH-1) format.
    intermediate = x << (w_width - 1 - msb_index);

    // Get rid of the 1 from the format for index calculation.
    // This is easier in SystemVerilog, just turn the bit to a 0.
    uint32_t temp = intermediate - (0b1 << (w_width - 1));
    temp = temp << LUT_POW;
    // Going from Q1.(WIDTH-1) to Q(WIDTH).0 in order to index the lut.
    // TODO: it will be easier to choose the first LUT_POW bits and use them 
    // to index the LUT.
    temp = temp >> (w_width - 1);
    uint16_t lut_index = temp;
    //std::cout << "LUT index " << lut_index << "\n";

    // FORMAT: Q1.(WIDTH-1)
    uint32_t initial_guess;
    if(lut_index == 0){
        initial_guess = lut[0];
    }
    else{
        initial_guess = lut[lut_index - 1];
    }
    
    uint32_t y = initial_guess;                 // FORMAT: Q1.(WIDTH-1)
    uint32_t mult;                              // FORMAT: Q1.(WIDTH-1)
    uint32_t yy;                                // FORMAT: Q1.(WIDTH-1)
    uint32_t sub;                               // FORMAT: Q1.(WIDTH-1)
    // NOTE: if the input is only 1 bit wide then this gets ignored because of the 
    // if statement checking whether x_red is 1 or 0.
    uint32_t threehalfs = 0x3 << (w_width - 2);   // FORMAT: Q1.(WIDTH-1)

    // Divide mapped input by 2 as stated in Newton-Raphson formula.
    x_red = x_red >> 1;

    // TODO: vary the number of iterations and evaluate the error.
    for (int i = 0; i < NUM_ITER; ++i) {
        // FORMAT: Q1.(WIDTH-1) x Q1.(WIDTH-1) = Q2.(2*WIDTH - 2) >> (WIDTH - 1) = Q1.(WIDTH-1)
        // NOTE: the format calculations may be off.
        yy = (y * y) >> (w_width - 1);
        // FORMAT: Q1.(WIDTH-1) x Q1.(WIDTH-1) = Q2.(2*WIDTH - 2) >> (WIDTH - 1) = Q1.(WIDTH-1)
        mult = (yy * x_red) >> (w_width - 1);
        // FORMAT: Q1.(WIDTH-1)
        sub = threehalfs - mult;
        // FORMAT: Q1.(WIDTH-1) x Q1.(WIDTH-1) = Q2.(2*WIDTH - 2) >> (WIDTH - 1) = Q1.(WIDTH-1)
        y = (y * sub) >> (w_width - 1);
    }

    // FORMAT: Q(i_width).(f_width)
    y = range_augmentation(y, msb_index);

    // If overflow then return max number possible.
    // Overflow is detected by checking if the MSB of Q1.15 format is asserted.
    // This is because the output range of y is [0.707, 1).
    // Can we get overflow? If the algorithm does not converge then yes.
    if(y > max_num){
        return max_num;
    }

    // FORMAT: Q(i_width).(f_width)
    return y;
}

// ===========================================================================
// Main
// ---------------------------------------------------------------------------

float test(uint16_t val, bool verbose){
    uint64_t mask = (0b1 << (i_width + f_width)) - 0b1;
    val = val & mask;
    // Convert to float for reference calculation.
    float val_f = qxy_to_float(val, i_width, f_width);

    if(val_f == 0){
        return 0;
    }

    // Reference calculation.
    float expected_f = 1.0f / sqrt(val_f);
    // Quantise the value for fair comparison.
    uint16_t expected = float_to_qxy(expected_f, i_width, f_width);
    // Update the reference value.
    expected_f = qxy_to_float(expected, i_width, f_width);
    if(expected > max_num){
        expected_f = qxy_to_float(max_num, i_width, f_width);
    }

    // Apply model.
    uint16_t output = isqrt(val);
    float output_f = qxy_to_float(output, i_width, f_width);

    // Calculate error in floating point.
    float error = abs(output_f - expected_f);
    if(verbose){
        std::cout << "sqrt(" << val_f << ") = " << expected_f << " |  " << output_f << " | Error: " << error << "\n";
    }
    return error;
}

// NOTE: for the Q8.8 format the max error achieved is 2^-8.
int main()
{
    for(i_width = 1; i_width < 9; i_width++){
        for(f_width = 1; f_width < 9; f_width++){
            float step = 0.001;
            float x_f = step;
            float max_error = 0.0f;
            w_width = i_width + f_width;
            max_num = (0b1 << w_width) - 0b1;
            init_lut();
            std::cout << "INT WIDTH: " << i_width << " ";
            std::cout << "FRAC WIDTH: " << f_width << " ";
            for(int i = 0; i < 100000; i++){
                int16_t x = float_to_qxy(x_f, i_width, f_width);
                float error = test(x, false);
                max_error = std::max(max_error, error);
                //if(error > 0.011f){
                //    print_bits<16>("IN", x);
                //    break;
                //}
                x_f += step;
            }
            std::cout << "Max error: " << max_error << "\n";
        }
    }

	return 0;
}

// ===========================================================================
// Utils
// ---------------------------------------------------------------------------

float qxy_to_float(uint64_t x, uint8_t int_width, uint8_t frac_width){
    float output = 0.0f;
    // Integer part
    uint64_t mask = (0b1 << int_width) - 0b1;
    uint64_t integer = (x >> frac_width) & mask;
    output = static_cast<float>(integer);

    // Fractional part
    mask = (0b1 << frac_width) - 0b1;
    uint64_t fraction = ((x << int_width) >> int_width) & mask;
    for(int i = 1; i < frac_width+1; i++){
        bool current_bit = (fraction >> (frac_width - i)) & 0b1;
        if(current_bit){
            float bin = pow(2, -i);
            output += bin;
        }
    }
    return output;
}

uint64_t float_to_qxy(float x, uint8_t int_width, uint8_t frac_width){
    uint64_t integer = static_cast<uint64_t>(x);
    float integer_float = static_cast<float>(integer);
    x -= integer_float;
    uint64_t output = integer << frac_width;

    for(int i = 1; i < frac_width+1; i++){
        float power = pow(2, -i);
        if(power <= x){
            uint64_t bin = 0b1 << (frac_width - i);
            output += bin;
            x-= power;
        }
    }
    return output;
}
