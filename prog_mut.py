"""
Created by: Rahul Kande
This script is used to mutate prog files
- Notes: 

- TODOs: 
"""
import subprocess, os, random, sys, re
from string import Template
import logging as lg # critical, error, warning, info, debug
from tqdm import tqdm

import thehuzz_utils


def bits(data, start_index, stop_index):
    l = len(data)
    return data[l-start_index-1:l-stop_index]

def cmp_inst_fields(target_inst, inst_data):
    #check opcode
    for b, b_ref in zip(bits(target_inst,6,0), inst_data[4]):
        if not (b_ref == 'x' or (b == b_ref)):
            return 0
    #check funct3
    for b, b_ref in zip(bits(target_inst,14,12), inst_data[3]):
        if not (b_ref == 'x' or (b == b_ref)):
            return 0
    #check funct7
    for b, b_ref in zip(bits(target_inst,31,25), inst_data[2]):
        if not (b_ref == 'x' or (b == b_ref)):
            return 0
    return 1


#function to convert hex instruction to binary
def inst_hex_to_bin(inst_h):
	try: #make sure that instruction is in correct format
		inst_bin = (bin(int(inst_h,16)))
		inst_bin = inst_bin[2:] #removes unnecessary chars (0 and b)
		inst_bin = inst_bin.zfill(32) #pads string with zeros

	except:	 #if not -> just replace it with all nop instruction
		inst_bin = "00010101000000000000000000000000"
			#print(inst_h,"->",inst_bin,"\n") #for testing
	return inst_bin

#function to convert binary instuction to hex
def inst_bin_to_hex(inst_bin):
	#convert back to hex
	inst_h = hex(int(inst_bin,2))
	inst_h = inst_h[2:] #removes unnecessary chars (0 and x)
	inst_h = inst_h.zfill(8) #pads string with zeros
	return inst_h


def cal_m_index(inst_data, num_bits, bits_type='data'):
    m_index = []
    avail_index = []

    if (bits_type == 'data'):
        #general mask = funct7 + rs2 + rs1 + funct3 + xxxxx + op
        mask = inst_data[2] + "x"*10 + inst_data[3] + "x"*5 + inst_data[4]

        #exceptions:
        #if inst_data[0] in ['System' or 'Sync']:
        #    mask = '0'*32  # nothing can be changed
        #    if inst_data[6] == 'FENCE'
        #        mask = mask[0:4] + 'x'*8 + mask[12:]
        if inst_data[8] in ['LR.W', 'LR.D']:
            mask = mask[0:7] + '0'*5 + mask[12:]
        if inst_data[8] in ['ECALL', 'EBREAK']:
            mask = '0'*32  # no bit can be changed
    elif (bits_type == 'all'):  #only mask the lsb 2 bits which indicate that
                                #instruction is 32 bits
        mask = "x"*30 + '11'
    elif (bits_type == 'opcode'): #mask everything other than opcode bits
        #general mask = funct7 + rs2 + rs1 + funct3 + xxxxx + op
        mask_n = inst_data[2] + "x"*10 + inst_data[3] + "x"*5 + inst_data[4]
        #exceptions:
        if inst_data[8] in ['LR.W', 'LR.D']:
            mask_n = mask_n[0:7] + '0'*5 + mask_n[12:]
        if inst_data[8] in ['ECALL', 'EBREAK']:
            mask_n = '0'*32  # no bit can be changed
        #mask_n masks data bits, reverse it
        mask = ''
        for bit in mask_n:
            bit_reversed = 'x' if bit!='x' else '0' 
            mask += bit_reversed
        #make sure last two bits are always masked since they indicte instrn
        #length
        mask = mask[0:30] + '11'
        
    else:
        print('Error: Unknown bits type found when mutating', bits_type \
               , inst_data, num_bits)

    mask = mask[::-1] #bcz inst data is reversed


    index = 0
    for bit in mask:
        if bit == 'x':
            avail_index.append(index)
        index = index + 1

    if len(avail_index) < num_bits:
        m_index = avail_index
    else:
        m_index_start = random.randint(0, len(avail_index)-1)    
        avail_index = avail_index + avail_index
        m_index = avail_index[m_index_start:m_index_start+num_bits]

    return m_index

#function to perform bit flip mutations
def bitflip(mut, inst_type, num_bits):
    #select bits to flip in registers
    m_index = cal_m_index(inst_type, num_bits)

    mut_list = [b for b in mut]
    for i in m_index:
        mut_list[i] = '0' if mut_list[i] == '1' else '1'

    mut = ''.join(mut_list)
    
    return mut

#add +-35 to a byte/bytes
#num_bytes = 1 or 2
def arith(mut, inst_type, num_bytes):

    if not num_bytes in [1,2]:
        print("Error. invalid num of bytes to arith function")
        exit()

    #select bits to flip in registers
    m_index = cal_m_index(inst_type, 8*num_bytes)

    # if no bit to mutate, return directly
    if len(m_index) == 0:
        return mut

    mut_list = [b for b in mut]
    mut_byte_list = [mut_list[i] for i in m_index]
    mut_byte = ''.join(mut_byte_list)
    mut_byte = int(mut_byte,2) #convert to integer    

    #add random # btw 0 and 35 to byte
    rand_int = random.randint(-35,35)
    mut_byte = mut_byte + rand_int

    #bring mut_byte back to range
    if num_bytes == 1:
        mut_byte = (mut_byte + 2**8 -1) if mut_byte<0 else mut_byte
    elif num_bytes == 2:
        mut_byte = (mut_byte + 2**16 -1) if mut_byte<0 else mut_byte

    #convert back to binary
    if num_bytes == 1:
        mut_byte = "{0:08b}".format(mut_byte)
    elif num_bytes == 2:
        mut_byte = "{0:016b}".format(mut_byte)

    num_bits = len(m_index)
    byte_num = 0
    for i in m_index:  #only overwrite bits given by m_index
        mut_list[i] = mut_byte[-len(m_index) + byte_num]
        byte_num = byte_num + 1

    mut = ''.join(mut_list)

    return mut

#functions to perform variable length bit flip mutations
def bitflip_1(mut, inst_type):   return bitflip(mut, inst_type, 1)
def bitflip_2(mut, inst_type):   return bitflip(mut, inst_type, 2)
def bitflip_4(mut, inst_type):   return bitflip(mut, inst_type, 4)
def byte_flip(mut, inst_type):   return bitflip(mut, inst_type, 8)
def byte_flip_16(mut, inst_type):return bitflip(mut, inst_type, 16)
def arith_8(mut, inst_type):     return arith(mut, inst_type, 1)
def arith_16(mut, inst_type):    return arith(mut, inst_type, 2)
def random_8(mut, inst_type):    return my_random(mut, inst_type, 8, 'data')
def random_8_any(mut, inst_type):    return my_random(mut, inst_type, 8, 'all')
def opcode_mut(mut, inst_type):    return my_random(mut, inst_type, 32, 'opcode')

#function to perform random mutation
def my_random(mut, inst_type, num_bits, bits_type):
    #select bits to flip 
    m_index = cal_m_index(inst_type, num_bits, bits_type)

    mut_list = [b for b in mut]
    for i in m_index:
        mut_list[i] = str(random.randint(0,1))

    mut = ''.join(mut_list)
    
    return mut

#Mutation that "deletes" (makes no op) the instruction
def delete(mut, inst_type, nop_inst_bin_32):            
    mut = nop_inst_bin_32[::-1]
    return mut
    ##randomly decides to replace instruction with no-op
    #delete_num = random.randint(0,3)
    #if (delete_num == 3): 
    #    if len(mut) == 16: 
    #        mut = nop_inst_16[::-1]
    #    elif len(mut) == 32:
    #        mut = nop_inst_bin_32[::-1]


#Mutation that clones a instruction
#mut --> instruction reversed
#replace_inst --> instruction not reversed
def clone(mut, inst_type, replace_inst): 
    #make sure to invert the replace inst
    replace_inst = replace_inst[::-1]
    return replace_inst


# mutate testcases for profiling
def gen_mut_files(ip_hex_files, out_dir\
                  , inst_list_all_w_ext, instr_list, hex_file_itr_re\
                  , num_inst_in_prog_prof, nop_inst_bin_32, hex_file_mut_t):

    #delete previous out log files
    thehuzz_utils.delete_dir(out_dir)
    
    val_mut = [0,1,2,3,4,5,6,7] #Valid mutations-> mutations that are used by
    
    for hex_file_in in tqdm(ip_hex_files, desc="----Mutating progs"):
        filename = os.path.basename(hex_file_in)
        # filter out hex files
        if not re.match(hex_file_itr_re, filename):
            continue
        inst = re.match(hex_file_itr_re, filename)
   
        instr_no = int(inst.group(1))
        inst_data = list((instr_list.items()))[instr_no]
        inst_data_clone = inst_data[1]
        inst_op_clone = inst_data[0]
        lg.debug(f"{inst_data_clone}, {inst_op_clone}")

        #if not int(inst.group(1)) in range(23,25):
        #    continue

        for m_type in val_mut:
            if (str(inst_data_clone[6]) == "1"): # single prog per inst
                no_of_files = num_inst_in_prog_prof
            else:  # all inst in one prog
                no_of_files = 1
            inst_bin_i_prev = -1
            for inst_no in range(no_of_files):
                #for mut_file in range(no_times_to_mut): 
                hex_file_out = hex_file_mut_t.substitute(\
                                             file_no = str(inst.group(1)), \
                                             file_itr = str(inst.group(2)), \
                                             mut = str(m_type), \
                                             inst_no = str(inst_no))
             
                hex_file_out = os.path.join(out_dir, hex_file_out)
                mem_file_out = hex_file_out.replace(".hex", ".mem")
                #print("generating " + mem_file_out.split('/')[-1], end='\r')

                #write new instructions to instruction file 
                hex_file_in_f = open(hex_file_in, 'r')
                hex_file_out_f = open(hex_file_out, 'w')
        
                mut_state = 0
                line_number = 0
                for line in hex_file_in_f.readlines():
	                #Parse input	
	                #these are in hex
                    inst_addr_h = line[1:9]   #instruction address        
                    inst_a_h    = line[10:18] #1st instruction
                    inst_b_h    = line[19:27] #2nd instruction
                    inst_c_h    = line[28:36] #3rd instruction
                    inst_d_h    = line[37:45] #4th instruction

                    if (inst_a_h == "00000013" and inst_b_h == "00000013" \
                        and inst_c_h == "00000013" and inst_d_h == "00000013"): 
                                 #4 nops tells it that it should start/stop mutating
                            if (mut_state == 0):
                                    mut_state = 1
                            elif (mut_state == 2):
                                    mut_state = 3

                            #write line to file
                            hex_file_out_f.write(line)

                    elif (mut_state == 1 or mut_state == 2): 
                        mut_state = 2
                        #only mutate line if it is not nop
                        line.strip('\n') #removes '\n'
                                      
                        #convert instructions to binary and reverse them
                        ###reversing is done so that 0 index in python becomes bit 0
                        inst_bin = [0,0,0,0]
                        inst_bin[0]    = inst_hex_to_bin(inst_a_h)
                        inst_bin[1]    = inst_hex_to_bin(inst_b_h) 
                        inst_bin[2]    = inst_hex_to_bin(inst_c_h) 
                        inst_bin[3]    = inst_hex_to_bin(inst_d_h)  
                        
                        for inst_i in [0,1,2,3]:
                            inst_bin_i = inst_bin[inst_i]

                            #print("start: ", inst_bin_i)

                            #if nop or shouldnt mutate instrn, skip mutation
                            if (inst_bin_i == nop_inst_bin_32): 
                                inst_bin[inst_i] = inst_bin_i
                                continue

                            ###reversing is done so that 0 index in python becomes bit 0
                            inst_bin_i_orig = inst_bin_i
                            # use the prev instruction since this is profiling
                            # stage
                            if (inst_bin_i_prev!=-1):
                                inst_bin_i = inst_bin_i_prev
                            else: 
                                inst_bin_i = inst_bin_i[::-1] #reverse

                            # get the inst type
                            i = inst_data_clone + [inst_op_clone[0]]
                            #got_inst = 0
                            #for inst_op, inst_data in inst_list_all_w_ext.items():
                            #    #print(bits(inst_bin_i_orig,6,0), inst_data[4])
                            #    if cmp_inst_fields(inst_bin_i_orig, inst_data):
                            #        if (inst_bin_i_prev!=-1):
                            #            inst_bin_i = inst_bin_i_prev
                            #        i = inst_data + [inst_op[0]]
                            #        if not (inst_data[7] == inst_data_clone[7]):
                            #            print("clone inst type and actual"\
                            #                    + "inst type not matching")
                            #            print(hex_file_in, hex_file_out)
                            #            print(inst_bin_i_orig, inst_data)
                            #            print(inst_data_clone)
                            #            print(i)
                            #            exit()
                            #        got_inst = 1
                            #        break

                            if   (m_type == 0): inst_bin_i = bitflip_1(inst_bin_i,i)    
                            elif (m_type == 1): inst_bin_i = bitflip_2(inst_bin_i,i)
                            elif (m_type == 2): inst_bin_i = bitflip_4(inst_bin_i,i)
                            elif (m_type == 3): inst_bin_i = arith_8(inst_bin_i,i)
                            elif (m_type == 4): inst_bin_i = arith_16(inst_bin_i,i) #j & r type
                            elif (m_type == 5): inst_bin_i = random_8(inst_bin_i,i)
                            elif (m_type == 6): inst_bin_i = byte_flip(inst_bin_i,i)
                            elif (m_type == 7): inst_bin_i = byte_flip_16(inst_bin_i,i)                   
                            else: print("Error: Incorrect Mutation")
                            #reverse the bits back when storing back to correct 
                            inst_bin[inst_i] = inst_bin_i[::-1]
                            inst_bin_i_prev = inst_bin_i
                            #print("ssoot: ", inst_bin_i[::-1])
                            #print(" ")
                               #print(inst_bin_i)
                               #print(" ")

                        #convert instructions back to hex       
                        inst_a    = inst_bin_to_hex(inst_bin[0]) 
                        inst_b    = inst_bin_to_hex(inst_bin[1])
                        inst_c    = inst_bin_to_hex(inst_bin[2])
                        inst_d    = inst_bin_to_hex(inst_bin[3])
                        
                        #format line for instruction file       
                        line = line[0] + inst_addr_h + " " + inst_a + " " + inst_b + " " + inst_c + " " + inst_d
                        
                        #print line to instruction file 
                        #print(line, end='\n')
                        hex_file_out_f.write(line + "\n")

                    else: #line is not in the range
                        hex_file_out_f.write(line)
                    
                    line_number = line_number + 1
    
                hex_file_in_f.close()
                hex_file_out_f.close()

                #convert the hex file into mem file so that ariane core and emulator can use it to run simulations
                thehuzz_utils.hex_to_mem(hex_file_out, mem_file_out)


# Helper function to check if a mutation is effective
def is_effective_mutation(existing_mutations, new_mutation):
    return new_mutation not in existing_mutations

# Updated mutate_prog function
def mutate_prog(hex_file_in, hex_file_out, mutation_prob, optimizer_sol, nop_inst_bin_32, inst_list_all_w_ext):
    lg.debug("generating", os.path.basename(hex_file_out), "from", os.path.basename(hex_file_in))

    # Open input and output hex files
    hex_file_in_f = open(hex_file_in, 'r')
    hex_file_out_f = open(hex_file_out, 'w')

    val_mut = [0, 1, 2, 3, 4, 5, 6, 7]  # Valid mutations
    opcode_mut_list = [8, 9, 10, 11]  # Opcode-changing mutations
    mut_state = 0
    line_number = 0
    inst_list_fr_clone = []
    existing_mutations = set()  # Track mutations

    for line in hex_file_in_f.readlines():
        inst_addr_h = line[1:9]  # Instruction address
        inst_a_h = line[10:18]  # 1st instruction
        inst_b_h = line[19:27]  # 2nd instruction
        inst_c_h = line[28:36]  # 3rd instruction
        inst_d_h = line[37:45]  # 4th instruction

        if inst_a_h == "00000013" and inst_b_h == "00000013" and inst_c_h == "00000013" and inst_d_h == "00000013":
            if mut_state == 0:
                mut_state = 1
            elif mut_state == 2:
                mut_state = 3
            hex_file_out_f.write(line)
        elif mut_state == 1 or mut_state == 2:
            mut_state = 2
            line = line.strip('\n')

            inst_bin = [
                inst_hex_to_bin(inst_a_h),
                inst_hex_to_bin(inst_b_h),
                inst_hex_to_bin(inst_c_h),
                inst_hex_to_bin(inst_d_h),
            ]

            for inst_i in [0, 1, 2, 3]:
                inst_bin_i = inst_bin[inst_i]
                inst_list_fr_clone.append(inst_bin_i)

                mutate_inst = random.randint(0, 100) < mutation_prob
                if inst_bin_i == nop_inst_bin_32 or not mutate_inst:
                    continue

                inst_bin_i_orig = inst_bin_i
                inst_bin_i = inst_bin_i[::-1]

                got_inst = False
                for inst_op, inst_data in inst_list_all_w_ext.items():
                    if cmp_inst_fields(inst_bin_i_orig, inst_data):
                        i = inst_data + [inst_op[0]]
                        got_inst = True
                        break
                if got_inst:
                    try:
                        opt_sol_mut_list = optimizer_sol[inst_op]
                    except KeyError:
                        opt_sol_mut_list = val_mut
                    m_type_frm_opt = random.choice(opt_sol_mut_list)
                    m_type_opcode = random.choice(opcode_mut_list)
                    m_type = random.choices([m_type_frm_opt, m_type_opcode], weights=[85, 15], k=1)[0]
                else:
                    i = ["none", "z", "xxxxxxx", "xxx", "xxxxxxx", "none", "1", 9999, "none"]
                    m_type = random.choice(opcode_mut_list)

                # Apply mutation based on type
                new_mutation = None
                if m_type == 0: new_mutation = bitflip_1(inst_bin_i, i)
                elif m_type == 1: new_mutation = bitflip_2(inst_bin_i, i)
                elif m_type == 2: new_mutation = bitflip_4(inst_bin_i, i)
                elif m_type == 3: new_mutation = arith_8(inst_bin_i, i)
                elif m_type == 4: new_mutation = arith_16(inst_bin_i, i)
                elif m_type == 5: new_mutation = random_8(inst_bin_i, i)
                elif m_type == 6: new_mutation = byte_flip(inst_bin_i, i)
                elif m_type == 7: new_mutation = byte_flip_16(inst_bin_i, i)
                elif m_type == 8: new_mutation = random_8_any(inst_bin_i, i)
                elif m_type == 9: new_mutation = delete(inst_bin_i, i, nop_inst_bin_32)
                elif m_type == 10: new_mutation = clone(inst_bin_i, i, random.choice(inst_list_fr_clone))
                elif m_type == 11: new_mutation = opcode_mut(inst_bin_i, i)

                if new_mutation and is_effective_mutation(existing_mutations, new_mutation):
                    existing_mutations.add(new_mutation)
                    inst_bin[inst_i] = new_mutation[::-1]

            inst_a = inst_bin_to_hex(inst_bin[0])
            inst_b = inst_bin_to_hex(inst_bin[1])
            inst_c = inst_bin_to_hex(inst_bin[2])
            inst_d = inst_bin_to_hex(inst_bin[3])

            line = line[0] + inst_addr_h + " " + inst_a + " " + inst_b + " " + inst_c + " " + inst_d
            hex_file_out_f.write(line + "\n")
        else:
            hex_file_out_f.write(line)

        line_number += 1

    hex_file_in_f.close()
    hex_file_out_f.close()


    ##convert the hex file into mem file so that ariane core and emulator can use it to run simulations
    #hex_to_mem(hex_file_out, mem_file_out)
    #exit()

#    inst_addr_offset = 0 #reset the instruction address offset

    # End of simulations




