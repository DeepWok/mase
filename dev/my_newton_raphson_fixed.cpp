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
#define I_WIDTH 8                           // FORMAT: integer
#define FRAC_WIDTH 8                        // FORMAT: integer
#define WIDTH (I_WIDTH + FRAC_WIDTH)        // FORMAT: integer

// Signatures for utils.
void print_float(std::string label, float x);
void print_int(std::string label, uint16_t x);
void print_int32(std::string label, uint32_t x);
void print_int64(std::string label, uint64_t x);
void print_bit(std::string label, bool bit);
void print_bit(int label, bool bit);
uint16_t float_to_q88(float x);
uint16_t float_to_q115(float x);
uint16_t float_to_q016(float x);
float q115_to_float(uint16_t x);
float q016_to_float(uint16_t x);
float q1616_to_float(uint32_t x);
float q3232_to_float(uint64_t x);
float q88_to_float(uint16_t x);

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
        // TODO: figure out how to support multiple WIDTH values.
        lut[i] += float_to_q115(ref);
        x += LUT_STEP;
    }
}

// Finds the MSB of the input number.
// NOTE: index 0 is assumed to be the left most bit within the number.
// FORMAT: Integer
uint16_t find_msb(uint16_t x){

    for(int i = 1; i < WIDTH + 1; i++){
        bool msb = (x >> (WIDTH - i)) & 0b1;
        if(msb){
            return WIDTH - i;
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
// INPUT FORMAT: Q(I_WIDTH).(FRAC_WIDTH)
// OUTPUT FORMAT: Q1.(WIDTH-1)
uint32_t range_reduction(uint32_t x, uint16_t msb_index){
    // Shifts the input left until the MSB of x is at the leftmost index.
    if(msb_index < WIDTH - 1){
        return x << (WIDTH - 1 - msb_index);
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
// OUTPUT FORMAT: Q(I_WIDTH).(FRAC_WIDTH)
uint32_t range_augmentation(uint32_t x_red, uint16_t msb_index){
    // Determine whether shifted right or left and by how much relative to the 
    // position of the fixed point in Q(I_WIDTH).(FRAC_WIDTH)
    // Left shift = Positive
    // Right shift = Negative
    int16_t shifted = FRAC_WIDTH - msb_index;

    // Reduction was through multiplication.
    // Therefore augmentation is through multiplication.
    if(shifted > 0){
        if(shifted % 2 == 0){
            int16_t shift = shifted >> 1; // k / 2
            // FORMAT: Q1.15
            int32_t res = x_red << shift;
            // FORMAT Q(I_WIDTH).(FRAC_WIDTH)
            res = res >> ((WIDTH - 1) - FRAC_WIDTH);
            return res;
        }
        else{
            int16_t shift = (shifted - 1) >> 1; // (k - 1) / 2
            // FORMAT: Q1.15
            int32_t res = (x_red * SQRT_2) >> 15;
            res = res << shift;
            // FORMAT Q(I_WIDTH).(FRAC_WIDTH)
            res = res >> ((WIDTH - 1) - FRAC_WIDTH);
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
            // FORMAT Q(I_WIDTH).(FRAC_WIDTH)
            res = res >> ((WIDTH - 1) - FRAC_WIDTH);
            return res;
        }
        else{
            int16_t shift = (-shifted - 1) >> 1;
            // FORMAT: Q1.15
            int32_t res = (x_red * ISQRT_2) >> 15;
            res = res >> shift;
            // FORMAT Q(I_WIDTH).(FRAC_WIDTH)
            res = res >> ((WIDTH - 1) - FRAC_WIDTH);
            return res;
        }
    }
    // Reduction was not done.
    // Therefore just convert format.
    else{
        // FORMAT Q(I_WIDTH).(FRAC_WIDTH)
        return x_red >> ((WIDTH - 1) - FRAC_WIDTH);
    }
}

// Newton Raphson's method.
//
// INPUT FORMAT: Q(I_WIDTH).(FRAC_WIDTH)
// OUTPUT FORMAT: Q(I_WIDTH).(FRAC_WIDTH)
uint16_t isqrt(uint16_t x){
    // X = 0 is invalid.
    // TODO: how to handle this for actual implementation? Return 0? Return MAX_NUM?
    if(x == 0){
        std::cout << "[ERROR] The input number x is invalid." << "\n";
        std::cout << "[X] " << x << "\n";
        return 0xFFFF;
    }

    // FORMAT: Integer
    uint16_t msb_index = find_msb(x);

    // FORMAT Q1.(WIDTH-1)
    uint32_t x_red = range_reduction(x, msb_index);

    // If X gets mapped to 1 then 1/sqrt(x) is 1. Then augment range.
    if(x_red == 0x8000){
        uint16_t out = range_augmentation(x_red, msb_index);
        bool msb_bit = (out >> 15) & 0b1;

        // Check that overflow does not occur.
        // TODO: how to deal with overflow in actual implementation? Return MAX_NUM?
        // Have an output wire OVERFLOW?
        if(msb_bit){
            std::cout << "[OVERFLOW]" << "\n";
            return 0xFFFF;
        }
        return out;
    }

    // FORMAT Q17.15
    uint32_t intermediate;
    // Shift the number to match the Q1.(WIDTH-1) format.
    intermediate = x << (WIDTH - 1 - msb_index);

    // Get rid of the 1 from the format for index calculation.
    // This is easier in SystemVerilog, just turn the bit to a 0.
    uint32_t temp = intermediate - (0b1 << (WIDTH - 1));
    temp = temp << LUT_POW;
    // Going from Q1.(WIDTH-1) to Q(WIDTH).0 in order to index the lut.
    // TODO: it will be easier to choose the first LUT_POW bits and use them 
    // to index the LUT.
    temp = temp >> (WIDTH - 1);
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
    uint32_t threehalfs = 0x3 << (WIDTH - 2);   // FORMAT: Q1.(WIDTH-1)

    // Divide mapped input by 2 as stated in Newton-Raphson formula.
    x_red = x_red >> 1;

    // TODO: vary the number of iterations and evaluate the error.
    for (int i = 0; i < 5; ++i) {
        // FORMAT: Q1.(WIDTH-1) x Q1.(WIDTH-1) = Q2.(2*WIDTH - 2) >> (WIDTH - 1) = Q1.(WIDTH-1)
        // NOTE: the format calculations may be off.
        yy = (y * y) >> (WIDTH - 1);
        // FORMAT: Q1.(WIDTH-1) x Q1.(WIDTH-1) = Q2.(2*WIDTH - 2) >> (WIDTH - 1) = Q1.(WIDTH-1)
        mult = (yy * x_red) >> (WIDTH - 1);
        // FORMAT: Q1.(WIDTH-1)
        sub = threehalfs - mult;
        // FORMAT: Q1.(WIDTH-1) x Q1.(WIDTH-1) = Q2.(2*WIDTH - 2) >> (WIDTH - 1) = Q1.(WIDTH-1)
        y = (y * sub) >> (WIDTH - 1);
    }

    // FORMAT: Q(I_WIDTH).(FRAC_WIDTH)
    y = range_augmentation(y, msb_index);

    // If overflow then return max number possible.
    // Overflow is detected by checking if the MSB of Q1.15 format is asserted.
    // This is because the output range of y is [0.707, 1).
    // Can we get overflow? If the algorithm does not converge then yes.
    bool msb_bit = (y >> (WIDTH - 1)) & 0b1;
    if(msb_bit){
        std::cout << "[OVERFLOW]" << "\n";
        return 0xFFFF;
    }

    // FORMAT: Q(I_WIDTH).(FRAC_WIDTH)
    return y;
}

// ===========================================================================
// Main
// ---------------------------------------------------------------------------

uint16_t test(uint16_t val){
    if(val == 0){
        return 0;
    }

    // Convert to float for reference calculation.
    float val_f = q88_to_float(val);
    // Reference calculation.
    float expected_f = 1.0f / sqrt(val_f);
    // Quantise the value for fair comparison.
    uint16_t expected = float_to_q88(expected_f);
    // Update the reference value.
    expected_f = q88_to_float(expected);

    // Apply model.
    uint16_t output = isqrt(val);
    float output_f = q88_to_float(output);

    // Calculate error in floating point.
    float error = abs(output_f - expected_f);
    std::cout << "Square root " << val_f << ") = " << expected_f << " |  " << output_f << " | Error: " << error << "\n";
    return error;
}

// NOTE: for the Q8.8 format the max error achieved is 2^-8.
int main()
{
    init_lut();

    float step = 1.0f;
    float x_f = step;
    float max_error = 0.0f;
    for(int i = 0; i < 1000; i++){
        int16_t x = float_to_q88(x_f);
        float error = test(x);
        max_error = std::max(max_error, error);
        if(error > 0.0f){
            break;
        }
        x_f += step;
    }
    std::cout << "Max error: " << max_error << "\n";

	return 0;
}

// ===========================================================================
// Utils
// ---------------------------------------------------------------------------

void print_float(std::string label, float x){
    std::cout << label << ": " << x << "\n";
}

void print_int(std::string label, uint16_t x){
    std::cout << label << ": " << std::bitset<WIDTH>(x) << "\n";
}

void print_int32(std::string label, uint32_t x){
    std::cout << label << ": " << std::bitset<32>(x) << "\n";
}

void print_int64(std::string label, uint64_t x){
    std::cout << label << ": " << std::bitset<64>(x) << "\n";
}

void print_bit(std::string label, bool bit){
    std::cout << label << ": "<< bit << "\n";
}

void print_bit(int label, bool bit){
    std::cout << label << ": "<< bit << "\n";
}

uint16_t float_to_q88(float x){
    uint16_t integer = static_cast<uint16_t>(x);
    float integer_float = static_cast<float>(integer);
    x -= integer_float;
    uint16_t output = integer << FRAC_WIDTH;

    for(int i = 1; i < FRAC_WIDTH+1; i++){
        float power = pow(2, -i);
        if(power < x){
            uint16_t bin = 0b1 << (FRAC_WIDTH - i);
            output += bin;
            x-= power;
        }
    }
    return output;
}

float q016_to_float(uint16_t x){

    float output = 0.0f;

    for(int i = 1; i < WIDTH + 1; i++){
        bool current_bit = (x >> (WIDTH - i)) & 0b1;
        if(current_bit){
            float bin = pow(2, -i);
            output += bin;
        }
    }
    return output;
}

uint16_t float_to_q016(float x){
    if(x >= 1){
        std::cout << "[error] input to float_to_q016 is larger than or equal to 1" << "\n";
        std::cout << "[error] " << x << "\n";
    }
    else if(x < 0){
        std::cout << "[error] input to float_to_q116 is smaller than 0" << "\n";
        std::cout << "[error] " << x << "\n";
    }

    uint16_t output = 0;

    for(int i = 1; i < WIDTH + 1; i++){
        float power = pow(2, -(i));
        if(power <= x){
            uint16_t bin = 0b1 << (WIDTH - i);
            output += bin;
            x-= power;
        }
    }
    return output;
}

uint16_t float_to_q115(float x){
    // Get integer part.
    uint16_t integer = static_cast<uint16_t>(x);
    float integer_float = static_cast<float>(integer);
    x -= integer_float;
    uint16_t output = integer << (WIDTH-1);

    for(int i = 2; i < WIDTH + 1; i++){
        float power = pow(2, -(i-1));
        if(power <= x){
            uint16_t bin = 0b1 << (WIDTH - i);
            output += bin;
            x-= power;
        }
    }
    return output;
}

float q115_to_float(uint16_t x){
    float output = 0.0f;
    bool bit1 = (x >> (WIDTH-1)) & 0b1;
    output += bit1 ? 1.0f : 0.0f;

    uint16_t fraction = x;
    for(int i = 2; i < WIDTH+1; i++){
        bool current_bit = (fraction >> (WIDTH - i)) & 0b1;
        if(current_bit){
            float bin = pow(2, -(i-1));
            output += bin;
        }
    }
    return output;
}

float q88_to_float(uint16_t x){
    float output = 0.0f;
    // Integer part
    uint16_t mask = (0b1 << FRAC_WIDTH) - 0b1;
    uint16_t integer = (x >> FRAC_WIDTH) & mask;
    output += static_cast<float>(integer);

    // Fractional part
    uint16_t fraction = ((x << I_WIDTH) >> I_WIDTH) & mask;
    for(int i = 1; i < FRAC_WIDTH+1; i++){
        bool current_bit = (fraction >> (FRAC_WIDTH - i)) & 0b1;
        if(current_bit){
            float bin = pow(2, -i);
            output += bin;
        }
    }
    return output;
}

float q1616_to_float(uint32_t x){
    float output = 0.0f;
    // Integer part
    uint16_t mask = (0b1 << 16) - 0b1;
    uint16_t integer = (x >> 16) & mask;
    output += static_cast<float>(integer);

    // Fractional part
    uint16_t fraction = ((x << 16) >> 16) & mask;
    for(int i = 1; i < 16+1; i++){
        bool current_bit = (fraction >> (16 - i)) & 0b1;
        if(current_bit){
            float bin = pow(2, -i);
            output += bin;
        }
    }
    return output;
}


float q3232_to_float(uint64_t x){
    float output = 0.0f;
    // Integer part
    uint32_t mask =  - 0b1;
    uint32_t integer = (x >> 32) & mask;
    output += static_cast<float>(integer);

    // Fractional part
    uint32_t fraction = ((x << 32) >> 32) & mask;
    for(int i = 1; i < 32+1; i++){
        bool current_bit = (fraction >> (32 - i)) & 0b1;
        if(current_bit){
            float bin = pow(2, -i);
            output += bin;
        }
    }
    return output;
}
