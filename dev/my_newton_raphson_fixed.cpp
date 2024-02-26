#include <cstdint>
#include <cmath>
#include <bits/stdc++.h>
#include <string>
#include <stdlib.h>

// FORMAT: Q1.15
#define ISQRT_2 0b0101101010000010
#define SQRT_2 0b1011010100000100
// NOTE: LUT_SIZE must be a power of 2.
#define LUT_POW 5
#define LUT_SIZE 32
#define LUT_STEP (1.0f / (LUT_SIZE + 1.0f))
#define I_WIDTH 8
#define FRAC_WIDTH 8
#define WIDTH (I_WIDTH + FRAC_WIDTH)

void print_float(std::string label, float x);
void print_int(std::string label, uint16_t x);
void print_int32(std::string label, uint32_t x);
void print_int64(std::string label, uint64_t x);
void print_bit(std::string label, bool bit);
void print_bit(int label, bool bit);
uint16_t float_to_q88(float x);
uint16_t float_to_q115(float x); // NOTE: the input is assumed to be in the range [1, 2)
uint16_t float_to_q016(float x); // NOTE: the input is assumed to be in the range [0, 1)
float q115_to_float(uint16_t x);
float q016_to_float(uint16_t x);
float q1616_to_float(uint32_t x);
float q3232_to_float(uint64_t x);
float q88_to_float(uint16_t x);

// ===========================================================================
// Newton Raphson method
// ---------------------------------------------------------------------------

// NOTE: Range reduction
// The input that can be supported by this algorithm is [0, inf). To make the 
// usage of the LUT more effective the values are mapped to a reduced range
// (see below).
// NOTE: this reduced range does not attempt to squash infinity into a small.
// The process is performed by dividing the input by 2 until it falls in the 
// range [1, 2). However, doing this is the same as moving the decimal point 
// up to right before the MSB, as long as the MSB exists in the integer part of the number 
// this equates to just imagining that the format of the number has changed 
// from Q(INT_WIDTH).(FRAC_WIDTH) to Q1.(MSB_POS?).

// NOTE: the lut values are in the format Q0.(WIDTH).
// The reason for this format is because the range of the input is mapped to
// [1, 2) therefore the possible values for 1/sqrt(x) will be (1/sqrt(2), 1].
// If we ignore the 1 in the domain then the mapped range becomes (1/sqrt(2), 1)
// which can be represented with the format Q0.(WIDTH). In this when the input
// is mapped to 1 then this will be handled with separate logic.

// NOTE: had to change the format of the LUT table values to Q1.15 due to the 
// existance of the 1.5f in the Newton Raphson method. For other methods such 
// as interpolation which do not include this then the format Q0.16 will be completely 
// valid.

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

    uint16_t msb_index = find_msb(x);
    //if(msb_index == 0xFFFF){
    //    std::cout << "[ERROR] The input number x is invalid." << "\n";
    //    std::cout << "[X] " << x << "\n";
    //    return 0xFFFF;
    //}
    // FORMAT Q1.15
    uint32_t x_red = range_reduction(x, msb_index);
    if(x_red == 0x8000){
        //std::cout << "X red: " << q115_to_float(x_red) << "\n";
        uint16_t out = range_augmentation(x_red, msb_index);
        //std::cout << "Out: " << q88_to_float(out) << "\n";
        bool msb_bit = (out >> 15) & 0b1;
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

    // TODO: what to do when the intermediate returns as a 1?
    // TODO: what to do when the intermediate is 0?
    
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

    // FORMAT: Q0.(WIDTH)
    // FORMAT: Q16.16
    uint32_t initial_guess;
    if(lut_index == 0){
        initial_guess = lut[0];
    }
    else{
        initial_guess = lut[lut_index - 1];
    }

    // FORMAT: Q1.15
    //uint32_t y = 0x4000;        // Q1.15 format
    // This represents 1.5 in Q1.15 format and 0.75 in Q0.16 format.
    //uint32_t x_red = 0xC000 >> 1;    // Q1.15 format therefore need to divide by 2.
    //std::cout << "X red: " << q115_to_float(x_red) << "\n";
    //std::cout << "Y    : " << q115_to_float(y) << "\n";
    //uint32_t yy = (y * y) >> 15;
    //std::cout << "YY   : " << q115_to_float(yy) << "\n";
    //uint32_t mult = (yy * x_red) >> 15;
    //std::cout << "Mult : " << q115_to_float(mult) << "\n";
    //uint32_t threehalfs = 0x3 << (WIDTH - 2); // FORMAT Q1.15
    //std::cout << "3/2  : " << q115_to_float(threehalfs) << "\n";
    //uint32_t sub = threehalfs - mult;
    //std::cout << "Sub  : " << q115_to_float(sub) << "\n";

    //y = sub; // FORMAT: Q1.16
    //yy = (y * y) >> 15; // FORMAT: Q1.16 * Q1.16 = Q1.34
    //std::cout << "YY   : " << q115_to_float(yy) << "\n";
    //mult = (yy * x_red) >> 15;
    //std::cout << "Mult : " << q115_to_float(mult) << "\n";
    //sub = threehalfs - mult;
    //std::cout << "Sub  : " << q115_to_float(sub) << "\n";

    // NOTE: since the format for intermediate is Q1.(WIDTH-1) and we are using
    // it with Q0.(WIDTH) numbers all that is needed to transform it to Q0.(WIDTH)
    // format is to multiply it by two since the fixed point will need to be
    // moved up by one position to the left.
    // With 32 bits this can be viewed as Q(32-WIDTH).(WIDTH) format.

    // FORMAT: Q16.16
    // TODO: is this correct? Is it meant to be by 2?
    // Mapped to [1, 2).
    //intermediate = intermediate << 1;
    //std::cout << "Inter: " << q1616_to_float(intermediate) << "\n";
    //uint32_t threehalfs = 0x3 << (WIDTH - 1); // FORMAT Q1.16
    //std::cout << "Three: " << q1616_to_float(threehalfs) << "\n";
    
    uint32_t y = initial_guess;     // FORMAT: Q1.15
    //uint32_t x_red = intermediate >> 1;  // FORMAT: Q1.15 and divide by 2 now.
    x_red = x_red >> 1;
    //std::cout << "X red: " << q115_to_float(x_red) << "\n";
    uint32_t mult; // FORMAT Q1.15
    uint32_t yy; // FORMAT Q1.15
    uint32_t sub; // FORMAT Q1.15
    uint32_t threehalfs = 0x3 << (WIDTH - 2); // FORMAT Q1.15
    for (int i = 0; i < 5; ++i) { // Adjust the number of iterations as needed
        //std::cout << "I: " << i << "\n";
        //std::cout << "Y   : " << q115_to_float(y) << "\n";
        yy = (y * y) >> 15; // Multiplication moves format to Q0.32 therefore need to shift by 16 to get Q0.16 format back.
        //std::cout << "YY  : " << q115_to_float(yy) << "\n";
        mult = (yy * x_red) >> 15; // Multiplication moves from to Q0.32 therefore need to shift by 16 to get Q0.16 format back.
        //std::cout << "MULT: " << q115_to_float(mult) << "\n";
        // In this case the format would change to Q1.16 which is ?
        sub = threehalfs - mult;
        //std::cout << "SUB : " << q115_to_float(sub) << "\n";
        y = (y * sub) >> 15;
        //std::cout << "Out : " << q115_to_float(y) << "\n";
        //y = y - (1.0f / (y * y) - intermediate) / (-2.0f / (y * y * y));
        //y = y * (1.5f - 0.5f * intermediate * y * y);
        
        //y = y * (threehalfs - (intermediate >> 1) * y * y);

        //print_int32("Y ", y);
        // TODO: multiplication of 2 32 bit numbers needs a result of 64 bits.
    }
    //print_int32("Y ", y);
    y = range_augmentation(y, msb_index);

    // Range augmentation.
    // FORMAT: from Q(32-WIDTH).(WIDTH) back to Q(INT_WIDTH).(FRAC_WIDTH)
    
    // If overflow then return max number possible.
    // Overflow is detected by checking if the MSB of Q1.15 format is asserted.
    // This is because the output range of y is [0.707, 1).
    // Can we get overflow? If the algorithm does not converge then yes.
    bool msb_bit = (y >> 15) & 0b1;
    if(msb_bit){
        std::cout << "[OVERFLOW]" << "\n";
        return 0xFFFF;
    }

    // If underflow then this should be dealt with ignoring the underflowed bits.
    

    // NOTE: this is just bit selection of the correct integer and fractional
    // bits and should be easier in SystemVerilog.
    //uint32_t mask = (0b1 << WIDTH) - 1; 
    //mask = mask << (16 - FRAC_WIDTH);
    //print_int32("MASK ", mask);
    //uint16_t out = intermediate & mask;
    //print_int("Out ", out);

    // RANGE augmentation is just changing the format back to Q(I_WIDTH).(FRAC_WIDTH)
    // which is just achieved through doing nothing.
    //std::cout << "Output: " << q88_to_float(y) << "\n";
    return y;
}

// ===========================================================================
// Driver 
// ---------------------------------------------------------------------------

uint16_t test(uint16_t val){
    if(val == 0){
        return 0;
    }
    float val_f = q88_to_float(val);
    float expected_f = 1.0f / sqrt(val_f);
    uint16_t expected = float_to_q88(expected_f);
    expected_f = q88_to_float(expected);
    uint16_t output = isqrt(val);
    float output_f = q88_to_float(output);
    float error = abs(output_f - expected_f);

    std::cout << "Square root " << val_f << ") = " << expected_f << " |  " << output_f << " | Error: " << error << "\n";
    return error;
}

int main()
{
    init_lut();

    // X_red = 1.495971875
    // But range_reduction() is not choosing 1.5 as X_red in order to avoid this
    // error the range_reduction() needs to be changed so that it also looks from above.
    // However this will require more logic. Not sure if this is desired.
    //float x_f = 47.8711f;
    //uint16_t x = float_to_q88(x_f);
    //test(x);

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

// RANGE: [0, 1)
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

// RANGE: [0, 1)
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

// NOTE: the input is assumed to be in the range [1, 2)
uint16_t float_to_q115(float x){
    //if(x >= 2){
    //    std::cout << "[ERROR] Input to float_to_q115 is larger than or equal to 2" << "\n";
    //    std::cout << "[ERROR] " << x << "\n";
    //}
    //else if(x < 1){
    //    std::cout << "[ERROR] Input to float_to_q115 is smaller than 1" << "\n";
    //    std::cout << "[ERROR] " << x << "\n";
    //}
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

// TODO: fix this function.
// Also figure out a way to make a function that takes in FRAC and INT widths.
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
