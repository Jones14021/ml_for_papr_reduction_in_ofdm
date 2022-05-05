import numpy as np
import os

K = 256 # number of OFDM subcarriers
mu = 4 # bits per symbol (i.e. 16QAM)
payloadBits_per_OFDM = K*mu  # number of payload bits per OFDM symbol

number_of_ofdm_samples_to_generate = 25000

filename_raw_bits = 'raw_bit_sequence_training_50000.npy'
try:
    os.remove(filename_raw_bits)
except:
    pass


totalbits = []

for i in range(number_of_ofdm_samples_to_generate):
    # %%
    # input: integer (start_code)
    # returns:  integer generated with a basic implementation for prbs31 with monic polynomial: x31 + x28 + 1
    def prbs31(start_code):
        for i in range(32):
            next_bit = ~((start_code>>30) ^ (start_code>>27))&0x01
            result_code = ((start_code<<1) | next_bit) & 0xFFFFFFFF
        return result_code

    # converts an integer to a bitfield
    def bitfield(n):
        return [1 if digit=='1' else 0 for digit in bin(n)[2:]] # [2:] to chop off the "0b" part 


    #print("payload bits per OFDM: " + str(payloadBits_per_OFDM))

    # create a bitfield via the prbs31 function by concatenating smaller bitfields
    # start prbs31 with a random integer
    bits_int = prbs31(np.random.randint(0, 2048))
    bits = bitfield(bits_int)
    while (len(bits) < payloadBits_per_OFDM):
        bits_int = prbs31(bits_int)
        bits = bits + bitfield(bits_int)


    # truncate bitfield to the exact length required
    bits = bits[:payloadBits_per_OFDM]

    # convert the list to a numpy.ndarray
    bits = np.asarray(bits)

    ## OTHER METHOD ##
    #bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))

    #print ("Bits count: ", len(bits))
    #print ("First 20 bits: ", bits[:20])
    #print ("Mean of bits (should be around 0.5 for binomial distribution with p=0.5): ", np.mean(bits))

    # append to array
    totalbits = np.append(totalbits, bits)     
    print(i)  

## save to file
myfile = open(filename_raw_bits, "wb")
np.save(myfile, totalbits)   
myfile.close()
print("Total number of bits created: " + str(number_of_ofdm_samples_to_generate * payloadBits_per_OFDM))

with open(filename_raw_bits, 'rb') as myfile:
    bits = np.load(myfile)

print("Total number of bits in file: " + str(len(bits)))