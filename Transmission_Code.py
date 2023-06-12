import numpy as np
import math
from matplotlib import pyplot as plt
import redpitaya_scpi as scpi
import time
import timeit
import scipy as sp
from scipy.io import wavfile
import string
from collections import Counter


# TODO Change the message and IPs

# RP IPs
IP_HACKATHON1 = '169.254.171.225'    #f07163
IP_HACKATHON2 = '169.254.30.94'     #f08734
IP_HACKATHON3 = '169.254.49.139'    #f0aa36


msg_short = " "     # Code
msg_long = " "      # Location of lockbox


msg_1_square = False

# Messages for RPs
message1 = "RED PITAYA"        # FPGA FLASHERS
message2 = "TRITION ROBOTICS IS AWESOME!"        # Triton Robotics: Multi buffer
message3 = "CQ THE COMBINATION IS: 0587 STAR LOCKER"

# "TREASURE"
# "CQ THE COMBINATION IS: 0587 STAR LOCKER"

### Turning on other RPs
ip_1 = True
ip_2 = False
ip_3 = False




# TODO - start of hackathon - test the generation frequency

# 1 period of 56 kHz === 17.857 us, 1 buffer @ 125 Mhz (dec == 1) == 131.072 us;  1 sample == 8 ns 

#######################################################
##################### FUNCTIONS #######################
#######################################################


##### Setup of units #####
# Can be called to change the frequency and number of pulses in a Morse unit# 1 period of 56 kHz === 17.857 us, 1 buffer @ 125 Mhz (dec == 1) == 131.072 us;  1 sample == 8 ns 

def symbol_setup(
    fp: int,
    pulses: int,
    buff_len: int
):
    """ Changes the DOT, DASH, LETTERS and WORDS to match the new pulsing frequency and pulses per Morse unit.
        Changes the pulse_period_len and unit variables according to inputs.
        The changes are only made if the ValueError is not raised.
    Args:
        fp (int): Frequency of a pulsing IR LED
        pulses (int): Number of pulses per Morse unit
        buff_len (int): Number of samples in a buffer

    Raises:
        ValueError: Checks whether there is enough space in the Red Pitaya buffer for such implementation.
    """
    pulse_period_len1 = round(SAMPLING_FREQUENCY/fp)
    print(pulse_period_len1)
    unit1 = pulse_period_len1*pulses
    buff_units = math.floor(buff_len/unit1)
    print(f"Units in buffer: {buff_units}")

    # Verify there are enogh units in a buffer
    try:
        assert buff_units >= max_symb_len
    except AssertionError:
        raise ValueError(f"Not enough units in a buffer ({buff_units}) to accomodate max symbol length ({max_symb_len}). Increase pulsing_frequency or decrease pulses_per_unit.")

    # change global variables
    global unit, pulse_period_len, dot, dash, letters, words
    unit = unit1
    pulse_period_len = pulse_period_len1

    half_pulse_int = math.floor(pulse_period_len1/2)
    if ((pulse_period_len % 2) == 1):
        one = np.ones(half_pulse_int + 1)
        one[0] = 0


    else:
        one = np.ones(half_pulse_int)


    pulse_period = np.append(one, np.zeros(half_pulse_int))    # 1 Pulse period


    print(f"pulse period length {len(pulse_period)}")

    one = pulse_period      # creating a array that represents a 1

    for i in range(0, pulses-1):
        one = np.append(one, pulse_period)

    dot = np.append(one, np.zeros(len(one)))    # redefines the dot
    dash = np.append(np.append(one,one), np.append(one, np.zeros(pulses*2*half_pulse_int)))
    letters = np.zeros(2*(pulses*(2*half_pulse_int)))
    words = np.zeros(6*(pulses*(2*half_pulse_int)))

def add_special(
    message: str,
    ending: str
) -> str:
    """Adds the start and end sequence to the morse message and checks for errors and empty message and returns a warning.

    Args:
        message (str): A string of characters that will be translated to Morse
        ending (str): The ending of Morse message

    Returns:
        str: Array of strings - The Message equipped with start and end signal, errors are added.

    Errors:
        Warning if empty message.
        Warning for wrong end character.

    """
    cypher = [[]]         # 3D list (split the message into letters and add special signs)
    message = [*message.upper()]
    i = 0

    # Checking for empty message
    if len(message) == 0:
        print("No message passed.")
        return cypher


    # Cheking for errors and adding the message
    while i < len(message):
        if MORSE_LEN.get(message[i]) is None:
            message[i]= "%"
        i = i + 1

    # Adding Start signal
    cypher[0:1] = ["start", " "]
    cypher.extend(message)

    # Ending
    cypher.append(" ")

    if ending.upper() == "KN" or ending == "SK":
        cypher.append(ending.upper())
    else:
        cypher.append("K")

    return cypher

def splitmsg(
    message: str,
    buff_len: int
    ) -> str:
    """Splits the string according to the Morse unit length and max buffer lenght and encodes it to Morse. Needs MORSE_LEN dictionary.

    Args:
        message (str): Array of strings - Input message
        buff_len (int): Length of buffer in samples

    Returns:
        str: Array of strings - the original string split into elements
        bool: True if the original string was split or False if it was not
    """
    split = []
    was_split = False

    buff_units = math.floor(buff_len/unit)

    i = 0       # Letter indicator
    j = 0       # Index for splitting

    temp = 0    # Current character length
    length = 0  # Total length of current split


    # Adding message
    while i < len(message):
        temp = MORSE_LEN.get(message[i])
        if temp != None:
            length += int(temp)    # get letter unit length and add it to the total

        if length >= buff_units:
            was_split = True
            split.append(message[j:i])
            j = i
            length = 0
            i = i-1

        i += 1
    
    split.append(message[j:])

    return split, was_split

def encrypt(message):
    cipher = ''
    for letter in message:
        if letter != ' ':
            if MORSE_CODE_DICT.get(letter) != None:
                # Looks up the dictionary and adds the corresponding morse code
                # along with a space to separate morse codes for different characters
                cipher += MORSE_CODE_DICT[letter] + ' '
        else:
            # 1 space indicates different characters and 2 indicates different words
            cipher += ' '
 
    return cipher

def decrypt(message):
 
    # extra space added at the end to access the last morse code
    message += ' '
 
    decipher = ''
    citext = ''
    for letter in message:
 
        # checks for space
        if (letter != ' '):
 
            # counter to keep track of space
            i = 0
 
            # storing morse code of a single character
            citext += letter
 
        # in case of space
        else:
            # if i = 1 that indicates a new character
            i += 1
 
            # if i = 2 that indicates a new word
            if i == 2 :
 
                 # adding space to separate words
                decipher += ' '
            else:
 
                # accessing the keys using their values (reverse of encryption)
                decipher += list(MORSE_CODE_DICT.keys())[list(MORSE_CODE_DICT.values()).index(citext)]
                citext = ''
 
    return decipher

# Special signs except start and endings are added as other characters

def morse2sig(
    morse_msg: str
) -> np.ndarray:
    """
        Transforms a string of Morse code to signal.

    Args:
        morse_msg (str): Message in Morse code

    Returns:
        np.ndarray: Square signal of Morse code
    """
    count = 0               # space counter
    i = 0                   # index counter
    sig = []

    for symbol in morse_msg:

        if symbol == ".":
            if(count >= 2):                         # space between words
                sig = np.concatenate((sig, words))
                count = 0
                i += 1
            elif(count == 1):                       # space between words
                sig = np.concatenate((sig, letters))
                count = 0
                i += 1
            sig = np.concatenate((sig, dot))
            i += 1

        elif symbol == "-":
            if(count >= 2):                         # space between words
                sig = np.concatenate((sig, words))
                count = 0
                i += 1
            elif(count == 1):                       # space between words
                sig = np.concatenate((sig, letters))
                count = 0
                i += 1
            sig = np.concatenate((sig, dash))
            i += 1

        elif symbol == " ":                         # skip empty spaces at the start
            if i != 0:
                count += 1                          # checking the space between letters and words

    return sig


#### NOT USED ###
def create_carrier(
    sample_freq: int,
    carrier_freq: int,
    buff_len: int = 16384,
    test_sig: bool = False
) -> np.ndarray:
    """Generates a sine carrier wave with specified carrier frequency.

    Args:
        sample_freq (int): Sampling frequency in Hz
        carrier_freq (int): Carrier frequency in Hz
        buff_len (int): Maximum buffer length in samples
        test_sig (bool): Creates the carrier regardless of units

    Returns:
        np.ndarray: One buffer-full of carrier wave.
    """
    # Parameters
    samp_period = sample_freq/carrier_freq          # Samples per period
    per_buff = buff_len/samp_period                 # Periods in buffer

    if not test_sig:

        period_min = 3
        print(f"Samples per period: {samp_period}")
        print(f"Periods per buffer: {per_buff}")

        # Check for enough carrier periods in unit
        try:
            assert unit >= period_min*samp_period
        except:
            raise ValueError(f"Carrier frequency too low. Need at least {period_min} carrier periods per message unit (currently {unit/samp_period}.)")

    t = np.linspace(0,1,buff_len)*2*np.pi
    # carrier = np.sin(math.floor(per_buff)*t)
    carrier = np.sin(per_buff*t)    # Approximatly the desired signal
    # print(f"Carrier length: {len(carrier)}")

    return carrier

#carrier = create_carrier(SAMPLING_FREQUENCY, Fc, BUFF_LEN, True)
# ? Plot carrier #

#if plot_carrier:
#    plt.plot(carrier) #,'x-')
#    plt.title(f"{Fc} Hz carrier wave")
#    plt.ylim([-1.1,1.1])

#######################################################
##################### VARIABLES #######################
#######################################################




#### PLOT SETTINGS ###
plot_carrier = False
plot_morsebuff = True
plot_received_data = False
plot_filtered_sig = False
plot_butter_test = False

### GLOBAL VARIABLES ###
BUFF_LEN = 16384
SAMPLING_FREQUENCY = int(125e6)
pulsing_freq = int(7.168e6)         # 56 kHz * 128 (decimation)    # 10e6 - 10 MHz
max_symb_len = 22                   # Morse units

pulse_period_len = round(SAMPLING_FREQUENCY/pulsing_freq)           # number of samples per pulse period
pulses_per_unit = 7                                                 # number of IR LED pulses per unit
unit = pulses_per_unit*pulse_period_len                             # number of samples in Morse unit


print(f"Pulse period length: {pulse_period_len}")       # 17 samples per period == 136 ns
print(f"Unit length: {unit}")

# Calculating set frequency from output frequency
target_output_freq = 57600       # 56 kHz   # 57.6 kHz

input_freq = round(target_output_freq*(pulse_period_len/BUFF_LEN), 1)  # calculation for the ideal set frequency / round to 1 decimal place
print(f"Frequency should be set to: {input_freq}")

dot = []
dash = []
letters = []
words = []


treshold = 0.2      # treshold filtering voltage level 



#######################################################
##################### DICTIONARY ######################
#######################################################

ABC = list(string.ascii_uppercase)
MORSE_ABC = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
NUMBERS = [str(n) for n in range(0,10)]
MORSE_NUM = ["-----",".----","..---","...--","....-",".....","-....","--...","---..","----."]
OTHER = [".",",","?","'","!","/","(",")","&",":",";","=","+","-","_",'"',"$","@"]
MORSE_OTH = [".-.-.-","--..--","..--..",".----.","-.-.--","-..-.","-.--.","-.--.-",".-...","---...","-.-.-.","-...-",".-.-.","-....-","..--.-",".-..-.","...-..-",".--.-."]

PROSIGNS_DECODE = ["start","KN","#","SK","%"]
MORSE_PROS = ["-.-.-","-.--.","...-.","...-.-","........"]

MORSE_PRO = dict(zip(PROSIGNS_DECODE, MORSE_PROS))

MORSE_LEN = {}


MORSE_CODE_DICT = dict(zip(ABC,MORSE_ABC))           # Adding letters to dictionary
MORSE_CODE_DICT.update(zip(NUMBERS, MORSE_NUM))      # Adding numbers to dictionary
MORSE_CODE_DICT.update(zip(OTHER, MORSE_OTH))
MORSE_CODE_DICT.update(MORSE_PRO)

# Creating a dictionary with unit lengths of characters
for key,value in MORSE_CODE_DICT.items():
    i = 0                       # unit counter
    for symbol in value:
        if symbol == ".":
            i += 2              # dot length in units (1 unit High, 1 unit Low)
        else:
            i += 4              # dash length in units (3 units High, 1 unit Low)
    
    i += 2                      # Add space between letters (2 units)
    MORSE_LEN.update({key: str(i)})

MORSE_LEN.update({" ": "4"})    # Add space length

print(MORSE_CODE_DICT)
print(MORSE_LEN)

# Order symbols by their lengths
counter = Counter(MORSE_LEN.values())
print(sorted(counter.items()))

print("\n\n\n\n")





#######################################################
#################### ACTUAL CODE ######################
#######################################################


symbol_setup(pulsing_freq, pulses_per_unit, BUFF_LEN)

print("\n\n")

# Add special symbols
mod_msg1 = add_special(message1, "K")
mod_msg2 = add_special(message2, "K")
mod_msg3 = add_special(message3, "K")

if ip_1:
    print(mod_msg1)
if ip_2:
    print(mod_msg2)
if ip_3:
    print(mod_msg3)

# Split the message into buffers
split_msg1, was_split1 = splitmsg(mod_msg1, BUFF_LEN)
split_msg2, was_split2 = splitmsg(mod_msg2, BUFF_LEN)
split_msg3, was_split3 = splitmsg(mod_msg3, BUFF_LEN)

#if ip_1:
    #print(split_msg1)
if ip_2:
    print(split_msg2)
if ip_3:
    print(split_msg3)

# # Encrypt the message per buffer => Add any special signs
msg_enc1 = []
msg_enc2 = []
msg_enc3 = []


for i in split_msg1:
    msg_enc1.append(encrypt(i))

for i in split_msg2:
    msg_enc2.append(encrypt(i))
    
for i in split_msg3:
    msg_enc3.append(encrypt(i))

if ip_1:
    print(msg_enc1)
if ip_2:
    print(msg_enc2)
if ip_3:
    print(msg_enc3)

print("\n\n")

sig1 = np.zeros((len(split_msg1), BUFF_LEN))
sig2 = np.zeros((len(split_msg2), BUFF_LEN))
sig3 = np.zeros((len(split_msg3), BUFF_LEN))

for i in range(0,len(msg_enc1)):
    temp = morse2sig(msg_enc1[i])
    sig1[i,0:len(temp)] = temp
    
for i in range(0,len(msg_enc2)):
    temp = morse2sig(msg_enc2[i])
    sig2[i,0:len(temp)] = temp
    
for i in range(0,len(msg_enc3)):
    temp = morse2sig(msg_enc3[i])
    sig3[i,0:len(temp)] = temp

if ip_1:
    print(f"Length of split message1: {len(split_msg1)}")
if ip_2:
    print(f"Length of split message2: {len(split_msg2)}")
if ip_3:
    print(f"Length of split message3: {len(split_msg3)}")


# ? Plotting the message#
if plot_morsebuff and ip_1:
    fig_gen, axs_gen = plt.subplots(len(sig1)) #, sharex = True)
    fig_gen.suptitle("Morse buffers")

    if len(sig1) == 1:
        axs_gen.plot(sig1[0])#,'-x')
        axs_gen.set_title("Message")
        axs_gen.set_xlabel("Buffer [samples]")
        axs_gen.set_ylabel("Amplitude [V]")
    else:
        for i in range(0,len(sig1)):
            axs_gen[i].plot(sig1[i]) #, '-x')
            axs_gen[i].set_title("Message")

if plot_morsebuff and ip_2:
    fig_gen, axs_gen = plt.subplots(len(sig2)) #, sharex = True)
    fig_gen.suptitle("Morse buffers")

    if len(sig2) == 1:
        axs_gen.plot(sig2[0],'-x')
        axs_gen.set_title("Message")
    else:
        for i in range(0,len(sig2)):
            axs_gen[i].plot(sig2[i], '-x')
            axs_gen[i].set_title("Message")


if plot_morsebuff and ip_3:
    fig_gen, axs_gen = plt.subplots(len(sig3)) #, sharex = True)
    fig_gen.suptitle("Morse buffers")

    if len(sig3) == 1:
        axs_gen.plot(sig3[0],'-x')
        axs_gen.set_title("Message")
    else:
        for i in range(0,len(sig3)):
            axs_gen[i].plot(sig3[i], '-x')
            axs_gen[i].set_title("Message")



# Transforming data into shape required by RP (string)
msg_sig1 = []
msg_sig2 = []
msg_sig3 = []
msg_sig_row1 = []
msg_sig_row2 = []
msg_sig_row3 = []

for i in range(0, len(sig1)):
    temp = []                       # temporary array of row
    for n in sig1[i]:                # transform into float and append
        temp.append(f"{n:.5f}")
    msg_sig_row1 = ", ".join(map(str, temp)) # convert row to list
    msg_sig1.append(msg_sig_row1)             # append row

for i in range(0, len(sig2)):
    temp = []                       # temporary array of row
    for n in sig2[i]:                # transform into float and append
        temp.append(f"{n:.5f}")
    msg_sig_row2 = ", ".join(map(str, temp)) # convert row to list
    msg_sig2.append(msg_sig_row2)             # append row
    
for i in range(0, len(sig3)):
    temp = []                       # temporary array of row
    for n in sig3[i]:                # transform into float and append
        temp.append(f"{n:.5f}")
    msg_sig_row3 = ", ".join(map(str, temp)) # convert row to list
    msg_sig3.append(msg_sig_row3)             # append row



plt.show()

######################################################
###### GENERATING CUSTOM SIGNAL WITH RED PITAYA ######
######################################################

# ACQISITION PART IS COMMENTED
IP = '192.168.0.34'
IP1 = '169.254.192.241'
IP2 = '192.168.163.125'

if ip_1:
    rp_1 = scpi.scpi(IP_HACKATHON1)
if ip_2:
    rp_2 = scpi.scpi(IP_HACKATHON2)
if ip_3:
    rp_3 = scpi.scpi(IP_HACKATHON3)

wave_form = 'arbitrary'
freq = input_freq         # generating frequency should be between [7500-7800]    # TODO 7650 almost ideal frequency for full buffer signals 10 Mhz
ampl = 1
unit = 'volts'
gain = ["lv","lv"]

dec = 128
trig_lvl = 0.4
trig_dly = 8100

buff1 = np.zeros((len(sig1),16384))
buff2 = np.zeros((len(sig2),16384))
buff3 = np.zeros((len(sig3),16384))



# Generation
if ip_1:
    rp_1.tx_txt('GEN:RST')
    rp_1.tx_txt('ACQ:RST')

    if msg_1_square:
        rp_1.sour_set(1, "square", 1, freq, burst= True, ncyc=7, nor=65536, period=200)
    else:
        rp_1.sour_set(1, wave_form, ampl, freq, burst= True, ncyc=1, nor=1)     # setting up signal source (channel 1)

    rp_1.acq_set(dec, trig_lvl, trig_dly, units=unit, gain=gain)            # acquisition settings

    rp_1.tx_txt('OUTPUT1:STATE ON')


    if msg_1_square:
        rp_1.tx_txt('SOUR1:TRIG:INT')

    rp_1.close()

if ip_2:
    rp_2.tx_txt('GEN:RST')
    rp_2.tx_txt('ACQ:RST')

    rp_2.sour_set(1, wave_form, ampl, freq, burst= True, ncyc=1, nor=1)     # setting up signal source (channel 1)
    rp_2.acq_set(dec, trig_lvl, trig_dly, units=unit, gain=gain)            # acquisition settings

    rp_2.tx_txt('OUTPUT1:STATE ON')

    rp_2.close()

if ip_3:
    rp_3.tx_txt('GEN:RST')
    rp_3.tx_txt('ACQ:RST')

    rp_3.sour_set(1, wave_form, ampl, freq, burst= True, ncyc=1, nor=1)     # setting up signal source (channel 1)
    rp_3.acq_set(dec, trig_lvl, trig_dly, units=unit, gain=gain)            # acquisition settings

    rp_3.tx_txt('OUTPUT1:STATE ON')

    rp_3.close()

# TODO Implement threading (threading library - look at it during hackathon)
## Constant transmission - the program should never end without an error
while(1):
    
    #time.sleep(1)  # Transmitting the signal once every minute

    if ip_1 and not msg_1_square:
        rp_1 = scpi.scpi(IP_HACKATHON1)

    if ip_2:
        rp_2 = scpi.scpi(IP_HACKATHON2)
    if ip_3:
        rp_3 = scpi.scpi(IP_HACKATHON3) 

    if ip_1 and not msg_1_square:
        rp_1.tx_txt('GEN:RST')
        rp_1.tx_txt('ACQ:RST')

        rp_1.sour_set(1, wave_form, ampl, freq, burst= True, ncyc=1, nor=1)     # setting up signal source (channel 1)
        rp_1.acq_set(dec, trig_lvl, trig_dly, units=unit, gain=gain)            # acquisition settings

        rp_1.tx_txt('OUTPUT1:STATE ON')

        rp_1.tx_txt('SOUR1:TRIG:INT')

    
    if ip_2:
        rp_2.tx_txt('GEN:RST')
        rp_2.tx_txt('ACQ:RST')

        rp_2.sour_set(1, wave_form, ampl, freq, burst= True, ncyc=1, nor=1)     # setting up signal source (channel 1)
        rp_2.acq_set(dec, trig_lvl, trig_dly, units=unit, gain=gain)            # acquisition settings

        rp_2.tx_txt('OUTPUT1:STATE ON')
    
    if ip_3:
        rp_3.tx_txt('GEN:RST')
        rp_3.tx_txt('ACQ:RST')

        rp_3.sour_set(1, wave_form, ampl, freq, burst= True, ncyc=1, nor=1)     # setting up signal source (channel 1)
        rp_3.acq_set(dec, trig_lvl, trig_dly, units=unit, gain=gain)            # acquisition settings

        rp_3.tx_txt('OUTPUT1:STATE ON')
    
    max_len = max(len(msg_sig1), len(msg_sig2), len(msg_sig3))

    for i in range(0, max_len):
        time.sleep(0.5)
        # t0 = timeit.default_timer()
        if (i < len(msg_sig1) and (not msg_1_square) and ip_1):
            rp_1.tx_txt(f"SOUR1:TRAC:DATA:DATA {msg_sig1[i]}")      # updating just the source data
        if ((i < len(msg_sig2)) and ip_2):
            rp_2.tx_txt(f"SOUR1:TRAC:DATA:DATA {msg_sig2[i]}")      # updating just the source data
        if ((i < len(msg_sig3)) and ip_3):
            rp_3.tx_txt(f"SOUR1:TRAC:DATA:DATA {msg_sig3[i]}")      # updating just the source data
        
        #rp_s.tx_txt('ACQ:START')
        ##### !time.sleep(0.1)
            #rp_s.tx_txt('ACQ:TRIG:LEV 0.4')
        #rp_s.tx_txt('ACQ:TRIG CH1_PE')
            #time.sleep(0.01)
        if (i < len(msg_sig1) and (not msg_1_square) and ip_1):
            rp_1.tx_txt('SOUR1:TRIG:INT')                           # triggering the signal generation
        if ((i < len(msg_sig2)) and ip_2):
            rp_2.tx_txt('SOUR1:TRIG:INT')
        if ((i < len(msg_sig3)) and ip_3):
            rp_3.tx_txt('SOUR1:TRIG:INT')
        # while 1:
        #     rp_s.tx_txt('ACQ:TRIG:STAT?')
        #     if rp_s.rx_txt() == 'TD':
        #         break
        # 
        # buff[i, :] = rp_s.acq_data(1, convert= True)
            # t1 = timeit.default_timer()
            # elapsed_time = round((t1 - t0) * 10 ** 6, 3)
            # print(f"Elapsed time: {elapsed_time} µs")
    
    
    
    print("SIGNAL SENT")
    if ip_1:
        rp_1.close()
    if ip_2:
        rp_2.close()
    if ip_3:
        rp_3.close()

    time.sleep(1)


#########################################
### ! NEVER END UP HERE ! ###

print("OUTSIDE INFINITE LOOP!")
# plt.show()

rp_1.close()
rp_2.close()
rp_3.close()

### ? Ploting acquired data ###
if plot_received_data:

    fig_acq, axs_acq = plt.subplots(len(buff1)) #, sharex = True)
    fig_acq.suptitle("Received data")

    if len(buff1) == 1:
        axs_acq.plot(buff1, 'x-')

    else:
        for i in range(0,len(buff1)):
            axs_acq[i].plot(buff1[i], 'x-')
            

#### Transforming received signal to binary signal ####


tresh_data = (buff1 > treshold).astype(int)

# ? Plotting received data #
if plot_received_data:

    fig_acq, axs_acq = plt.subplots(len(tresh_data)) #, sharex = True)
    fig_acq.suptitle("Treshold data")

    if len(tresh_data) == 1:
        axs_acq.plot(tresh_data[0], 'x-')

    else:
        for i in range(0,len(tresh_data)):
            axs_acq[i].plot(tresh_data[i], 'x-')

plt.show()
