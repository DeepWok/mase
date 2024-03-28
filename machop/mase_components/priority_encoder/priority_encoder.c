
#include <stdio.h>
#include <stdint.h>


unsigned int encoder(unsigned int input_channels,unsigned int NO_INDICIES) {
    if (input_channels== 0) return 0; // No bits are set.
  
    unsigned int input_channels_temp = input_channels;
    unsigned int mask = 0;
    unsigned int channel_mask;
    for (int j = 0; j < NO_INDICIES; j = j + 1){
      channel_mask = input_channels_temp&(~(input_channels_temp-1));
      printf("channel_mask: ");
      printBinary(channel_mask);

      input_channels_temp = input_channels_temp & ~ channel_mask;
      printf("input_channels_temp: ");

      printBinary(input_channels_temp);
      mask = mask | channel_mask;
    }

    
    return mask; // Return the mask that isolates the MSB that is set.
}


void printBinary(unsigned int num) {
    int numBits = sizeof(num) * 8; // Calculate the number of bits in the number
    for (int i = numBits - 1; i >= 0; i--) {
        int bit = (num >> i) & 1; // Shift and isolate each bit
        printf("%d", bit);
    }
    printf("\n"); // Newline for readability
}


int main() {
    unsigned int testNumbers[] = {0b1100, 0b0010, 0b1000, 0b0000, 0b0001,0b1111};
    int numTests = sizeof(testNumbers) / sizeof(testNumbers[0]);
    unsigned int NO_INDICIES = 2;
    for (int i = 0; i < numTests; i++) {
        unsigned int num = testNumbers[i];

        printf("Input: ");
        printBinary(num);
        unsigned int result = encoder(num,NO_INDICIES);
        printf("MSB:   ");
        printBinary(result);
        printf("\n");
    }

    return 0;
}


