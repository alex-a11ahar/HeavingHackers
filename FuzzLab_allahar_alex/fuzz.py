"""
Created by: Rahul Kande
This is the main script to run the TheHuzz fuzzer
- Notes: 

- TODOs: 
    - Add pause/resume feature
"""

import subprocess, os, sys, re
import time, random, copy
import json, jsonlines
from string import Template
from tqdm import tqdm
import logging as lg # critical, error, warning, info, debug

import config
from configManager import getCONFIG
import prog_gen, riscv_isa, prog_mut, prog_sim, parse_cov
from riscv_isa import inst_list_all_w_ext, nop_inst_bin_32
import thehuzz_utils as TU


"""
Gets the optimal instruction-mutation pairs from the optimizer solution
"""
def get_sol(sol_file):
    sol = {}

    sol_file_p = open(sol_file, 'r')
    sol_data_json = json.load(sol_file_p)

    for variable in sol_data_json["CPLEXSolution"]["variables"]:
        if variable["value"] == '1.0':
            name_data = re.match('bool_([^"]*)_([0-9])', variable["name"])
            sol_opcode = name_data.group(1)
            sol_mut = int(name_data.group(2))
            if sol_opcode in sol: # opcode already there, append the mut type
                sol[sol_opcode].append(sol_mut)
            else:  # new opcode, create a new list with mut type
                sol[sol_opcode] = [sol_mut]

    return sol


"""
Gets the optimizer solution with the optimal instruction-mutation pairs,
and the opcodes to use for seed generation
"""
def get_thehuzz_parameters(core, sol_file):


    # get the solution from the optimizer
    if core == 'cva6': # update the old format to the new one
        old_optimizer_sol = get_sol(sol_file)
        optimizer_sol = {}
        for opc,mut_list in old_optimizer_sol.items():
            for opc_ext in inst_list_all_w_ext:
                if opc == opc_ext[0]:
                    optimizer_sol[opc_ext[0] + "_" + opc_ext[1]] = mut_list

    else:
        optimizer_sol = get_sol(sol_file)

    # get the opcode lists
    opcode_list_temp = [sol for sol in optimizer_sol]
    opcode_list = [tuple(opc.split('_')) for opc in opcode_list_temp]
    first_opcode_list = []
    for opcode in opcode_list:
        if inst_list_all_w_ext[opcode][6] == '0':
            first_opcode_list.append(opcode)

    #print(optimizer_sol, len(optimizer_sol), '\n')
    #print(first_opcode_list, '\n')
    #print(opcode_list, '\n')

    return optimizer_sol, first_opcode_list, opcode_list

"""
Generates num_progs number of seed testcases for the fuzzer
"""
def gen_progs(no_threads, gen_progs_dir, num_progs, num_inst_in_prog, opcode_list, sw_dir\
            , inst_list_all_w_ext, num_nops_at_start, num_nops_at_end\
            , first_opcode_list, trash_dir, debug_print):

    del_repo = 1 
    prog_gen.gen_multi_random_prog(
                          no_threads, gen_progs_dir, del_repo\
                        , num_progs, num_inst_in_prog, opcode_list\
                        , sw_dir, inst_list_all_w_ext\
                        , num_nops_at_start, num_nops_at_end, 'fuzzer'\
                        , first_opcode_list, trash_dir, debug_print)


"""
Simulates all the testcases in the testcases_to_sim array and returns a dictionary 
of coverage data for each simulation
"""
def sim_testcases(testcases_to_sim, testcase_ids, CORE_PT, CONFIG_CORE_PT\
               , store_trace_file, store_cov_file \
               , detecting_bugs, no_threads, core, tot_sim_time, cov_enable\
               , cov_types, vdb_cov_files, instance_list):

    sim_files_to_save = {} # files to save
    # store trace file only if we cannot use custom path and user asks for it
    store_trace_file = (CORE_PT['trace_out_path_t'] != None) and store_trace_file
    data_types = [('cov', store_cov_file), ('bug', detecting_bugs), ('trace', store_trace_file)]
            # false for trace because trace is directly dumped to the outputs dir
    for data, store_en in data_types: 
        sim_files_to_save[data] = { 'en': store_en\
                , 'from': CORE_PT[f'{data}_out_path_t']\
                , 'to': Template(f"{CONFIG_CORE_PT['sim_store_dir']}/{CONFIG_CORE_PT[f'{data}_out_t'].template}") }

    cov_data = prog_sim.sim_progs(testcases_to_sim, no_threads, testcase_ids \
            , core, tot_sim_time, cov_enable, cov_types, vdb_cov_files\
            , instance_list, CONFIG_CORE_PT["sim_bash_file"], CORE_PT, time.time()\
            , sim_files_to_save, detecting_bugs)

    return cov_data


"""
EDIT: this function is incomplete/incorrect. fix it with the help of the
      comments provided
- This function caclculates how many times to mutate a testcase
"""
def calc_no_times_to_mut(num_times_to_mut, times_mutated\
                         , cov_incr_dict, num_new_testcases, feedback_cov_types):

    # if the queue already has more than 1000 programs, dont mutate anymore
    # since we dont need that many tests for this assignment
    if num_new_testcases > 1000: return 0, 'no_mut'
    # compute the queue load (more load if more no of testcases yet to simulate)
    n = num_new_testcases
    queue_load = 1 if n>10_000 else ( 0.75 if n>5_000 else (0.5 if n>1_000 else 0) )

    # set the default
    num_times_to_mut_local = num_times_to_mut

    # calculate cov_incr
    cov_incr = sum([cov_incr_dict[cov_type] for cov_type in feedback_cov_types])

    mut_type = "no_mut"

    if cov_incr > 0:
        # decrease the no of muts based on queue load
        num_times_to_mut_local = num_times_to_mut*(1-queue_load)

        # Reduce further based on testcase age.
        decr_factor = 0.25 if times_mutated > 3 else (0.5 if times_mutated > 1 else 1)
        num_times_to_mut_local *= decr_factor

        # EDIT 1 START:
        # we need to decrease the number of mutations based on the number of
        # times they are already mutated and the number of testcases in the
        # queue. This helps in exploring different testcases instead of getting
        # stuck with a single testcase
        # for this, write a code that reduces the num_times_to_mut_local value
        # by multiplying it with the
        # decreasing factor i.e., decr_factor.
        #Make sure that
        # num_times_to_mut_local remains a integer
        
        #To start the edit we call the decr_factor value above and mutiply
        num_times_to_mut_local *= decr_factor


        # After reducing the variable we then ensure that the variable is an integer.
        num_times_to_mut_local = int(num_times_to_mut_local)

        # EDIT 1 END

        # EDIT 2 START:
        # feedback engine computed the number of times to mutate the current
        # program using the num_times_to_mut_local variable.
        # but, if the value in this variable should be atleast 1.
        # so, write a code below to check if the value of num_times_to_mut_local is less
        # than 1 and if it is, then set num_times_to_mut_local to 1
        
        # A if statement to verify that the variable is at least 1.
        if num_times_to_mut_local < 1:
            num_times_to_mut_local = 1 # If below 1 set to one.

        # uncomment the line below once you set the num_times_to_mut_local
        # variable
        
        #to check
        assert num_times_to_mut_local >= 1, f"your edit to num_times_to_mut_local is incorrect. check it"
        
        # EDIT 2 END

       

        mut_type = 'interesting'
    else:
        # if coverage is not increased, then mutate few times only if it is a
        # newly generated input (this is done to keep mutations alive when
        # coverage saturates and new inputs are being generated)
        if times_mutated == 0: # this is a generated input, dint mutate yet

            # EDIT 3 START:
            # write a code below to assign a random integer value to
            # num_times_to_mut_local from this range: [0:num_times_to_mut/2)
            num_times_to_mut_local = random.randint(0, max(0, num_times_to_mut // 2 - 1)) #call random to create a random integer in the range of 0 and half of total num_times_to_mut, the -1 keep the value in the wanted range.
            # EDIT 3 END
            
            mut_type = 'just_gen'
        else:
            num_times_to_mut_local = 0
            mut_type = 'no_mut'
            
    return num_times_to_mut_local, mut_type

"""
This is the feedback engine that computes which testcases to mutate and how many times
- Any testcase that increased coverage will be mutated
- Seed testcases are mutated even if they did not increase coverage
"""
def feedback_based_selection(num_new_testcases, testcases_to_sim\
                    , cov_increment_data_dict, num_times_to_mut, feedback_cov_types):

    #retain only the performers or first timers
    progs_to_mut = []
    interesting_testcases = []
    just_generated_testcases = []
    no_times_to_mut_prog_local = 0

    for testcase in testcases_to_sim:
        testcase['mut_times'], mut_type = calc_no_times_to_mut(\
                num_times_to_mut,  testcase['times_mutated']\
                , cov_increment_data_dict[testcase['id']]['incr'], num_new_testcases\
                , feedback_cov_types)
        num_new_testcases += testcase['mut_times']

        if mut_type == 'interesting': 
            interesting_testcases.append( [testcase['id'], testcase['mut_times']] )
        elif mut_type == 'just_gen': 
            just_generated_testcases.append( [testcase['id'], testcase['mut_times']] )
        else: 
            assert mut_type == 'no_mut', f"unknown reason for mutation: {mut_type}"

    return testcases_to_sim, interesting_testcases, just_generated_testcases


"""
This function mutates the testcases to generate new testcases
"""
def run_muts(testcases_to_mut, optimizer_sol, nop_inst_bin_32, inst_list_all_w_ext):

    #mutate each of the program selected by the feedback engine
    mutation_prob = 100 #float(20) + float(80/num_progs_to_gen)
    progs_to_sim = []

    for testcase in testcases_to_mut:
        for hex_file_out in testcase['new_hex_files']:
            prog_mut.mutate_prog(testcase['hex_file'], hex_file_out, mutation_prob\
                        , optimizer_sol, nop_inst_bin_32, inst_list_all_w_ext)


"""
EDIT: this function is incomplete/incorrect. fix it with the help of the
      comments provided
- This function computes the number of coverage points covered so far and
  returns the coverage achieved as cov_points_ach and the percentage of coverage achieved as 
  cov_per_ach

- You can find the tests for this function in thehuzz_utils.py file in the
  function: check_compute_cov_achieved_func
"""
def compute_cov_achieved(merged_cov_dict, tot_cov_points):
    
    # EDIT1 START: calculate the number of coverage points achieved as the sum of the
    # individual points covered. merged_cov_dict is a dictionary that has
    # coverage achieved for each coverage type in the form of a string of 0's
    # and 1's
    # for example: merged_cov_dict = {'line': '001000101...', 'branch': '1110111011...', ...}
    # so, we first need to create a new dictionary where we count the number of
    # 1's in each string and store this integer count value. then we count
    # the number of coverage points of all coverage types to get one integer
    # number for coverage and assign it to cov_points_ach


    # uncomment the code below and complete it:
    
    merged_cov_points_dict = {} #Dictonary for different coverage types
    for coverage_type, coverage_string in merged_cov_dict.items(): # for loop going through the dictonary created above
        merged_cov_points_dict[coverage_type] = coverage_string.count('1') # Count the 1s in the coverage for this type.

    
    cov_points_ach = sum(merged_cov_points_dict.values()) # Sum coverage points to get the total coverage points achieved.

    # EDIT 1 END

    # EDIT 2 START: calculate the percentage of coverage points achieved by dividing
    # the total number of coverage points i.e., cov_points_ach with the total
    # number of coverage points in the design, ie., tot_cov_points and
    # multiplying it with 100. You can
    # round off this number to two decimal digits to improve readability, but
    # this is not mandatory.
     
    if tot_cov_points > 0:  #If statement to prevent divide by 0, error handling
        cov_per_ach = (cov_points_ach / tot_cov_points) * 100  # If not 0 divide by total and multiply by 100 to get a percentage
    else:
        cov_per_ach = 0  # No coverage points to divide by.
        
    
    cov_per_ach = round(cov_per_ach, 2) # Round the percentage value to two decimal places using the round function.

    # EDIT 2 END

    return cov_points_ach, cov_per_ach


"""
Main function that runs the fuzzer
- Fuzzer stops when timelimit, testcase limit, or coverage % limit is reached
"""
def run_thehuzz(fuzz_time, CONFIG_PT, CONFIG_CORE_PT\
              , core, instance_list, max_fuzz_time, max_fuzz_progs, target_cov\
              , sim_batch_size\
              , optimizer_sol, first_opcode_list, opcode_list\
              , num_inst_in_prog, inst_list_all_w_ext, num_nops_at_start, num_nops_at_end\
              , detecting_bugs, no_threads, tot_sim_time, cov_enable\
              , cov_types, vdb_cov_files, store_elf_file, store_trace_file, store_cov_file\
              , num_times_to_mut, nop_inst_bin_32, feedback_cov_types\
              , debug_print): 

    input_database = TU.DATABASE(CONFIG_PT['all_progs_dir']\
                            , CONFIG_PT['hex_file_t']) # database of input testcases
    tot_cov_points = None # total coverage points
    cov_per_ach    = 0 # percentage of coverage achieved so far 
    merged_cov_dict = None # merged coverage of all simulations
    inputs_log_file = CONFIG_PT['inputs_log_file']

    TU.TIMELOG(fuzz_time, f" -- Loading input seeds (.hex format)")
    # load any seeds from input_seeds dir (these seeds should be in .hex format)
    input_test_files = TU.get_files_in_dir(CONFIG_PT['input_seeds_dir']\
                                                , CONFIG_PT['seed_input_file_re'])
    input_database.add_testcases(input_test_files)
    TU.log(inputs_log_file, f"Loaded {len(input_test_files)} input seeds | Total testcases = {input_database.num_testcases()}\n", fuzz_time)
    TU.TIMELOG(fuzz_time, f" -- Loading input seeds", True)

    while (fuzz_time.time_diff() < max_fuzz_time \
       and input_database.num_testcases_simulated() < max_fuzz_progs \
       and cov_per_ach < target_cov): 

        # generate inputs if database doesnt have enough testcases
        num_progs = sim_batch_size - input_database.num_new_testcases()
        save_filetypes = ['riscv'] if store_elf_file else [] 
        
        if num_progs > 0:
            TU.TIMELOG(fuzz_time, f" -- Generating {num_progs} testcases")
            gen_progs(no_threads, CONFIG_PT['gen_progs_dir'], num_progs, num_inst_in_prog\
                , opcode_list, CONFIG_PT['sw_run_dir']\
                , inst_list_all_w_ext, num_nops_at_start, num_nops_at_end\
                , first_opcode_list, CONFIG_PT['trash_run_dir'], debug_print)

            generated_test_files = TU.get_files_in_dir(CONFIG_PT['gen_progs_dir']\
                                                , CONFIG_PT['seed_input_file_re'])
            input_database.add_testcases(generated_test_files, save_filetypes)

            TU.log(inputs_log_file, f"Generated {num_progs} testcases | Total testcases = {input_database.num_testcases()}\n", fuzz_time)
            TU.TIMELOG(fuzz_time, f" -- Generating {num_progs} testcases", True)

        TU.TIMELOG(fuzz_time, f" -- Running simulations")
        testcases_to_sim = input_database.get_testcases_to_sim(sim_batch_size)
        files_to_sim = [i['hex_file'] for i in testcases_to_sim]
        save_ids = [i['id'] for i in testcases_to_sim]
        cov_data_dict = sim_testcases(files_to_sim, save_ids\
               , CONFIG_CORE_PT, CONFIG_PT, store_trace_file, store_cov_file \
               , detecting_bugs, no_threads, core, tot_sim_time, cov_enable\
               , cov_types, vdb_cov_files, instance_list)
        TU.TIMELOG(fuzz_time, f" -- Running simulations", True)

        TU.TIMELOG(fuzz_time, f" -- Analyzing coverage data")
        # merge coverage
        merged_cov_dict, cov_increment_data_dict \
                    = parse_cov.merge_cov_dicts(fuzz_time.get_time(False)\
                                    , cov_data_dict, merged_cov_dict)

        # update the cov log file
        with jsonlines.open(CONFIG.pt['cov_log_file'], 'a') as fp: 
            for cov_data in cov_increment_data_dict.values(): fp.write(cov_data)

        # coverage feedback
        testcases_to_mut, interesting_testcases, just_generated_testcases = \
            feedback_based_selection(input_database.num_new_testcases()\
                    , testcases_to_sim, cov_increment_data_dict, num_times_to_mut\
                    , feedback_cov_types)

        TU.log(inputs_log_file, f"Testcases to mutate: Interesting:{interesting_testcases} | Just generated: {just_generated_testcases}\n", fuzz_time)
        TU.TIMELOG(fuzz_time, f" -- Analyzing coverage data", True)

        TU.TIMELOG(fuzz_time, f" -- Mutating testcases")
        testcases_to_mut = input_database.allocate_testcases_to_mut(testcases_to_mut)
        run_muts(testcases_to_mut, optimizer_sol, nop_inst_bin_32, inst_list_all_w_ext)
        TU.log(inputs_log_file, f"Mutation done | Total testcases = {input_database.num_testcases()}\n", fuzz_time)
        TU.TIMELOG(fuzz_time, f" -- Mutating testcases", True)


        # compute coverage achieved so far
        if not tot_cov_points: 
            tot_cov_points = sum([len(cov_str) for cov_str in merged_cov_dict.values()]) 
        cov_points_ach, cov_per_ach = compute_cov_achieved(merged_cov_dict, tot_cov_points)
        TU.TIMELOG(fuzz_time, f" -- {input_database.num_testcases_simulated()} testcases, {cov_per_ach}% coverage achieved", False, True)


    # log stats:
    if merged_cov_dict: 
        tot_cov = {key: len(cov_str) for key, cov_str in merged_cov_dict.items()}
        tot_cov['Total'] = sum(tot_cov.values())
        if cov_points_ach > 0: 
            ach_cov = {key: cov_str.count('1') for key, cov_str in merged_cov_dict.items()}
            ach_cov['Total'] = sum(ach_cov.values())
        else: 
            ach_cov = 0
        cov_per = cov_per_ach
    else:  # need to run atleast one simulation to get these stats
        tot_cov = {}
        ach_cov = {}
        cov_per = 0
    stats_string  = f"\n{'-'*60}\n"
    stats_string += f"  Benchmark              : {core}\n"
    stats_string += f"  Run time               : {fuzz_time.get_time(False)} sec\n"
    stats_string += f"  No. of testcases       : {input_database.num_testcases_simulated()}\n"
    stats_string += f"  No. of coverage points : {tot_cov}\n"
    stats_string += f"  No. of points covered  : {ach_cov}\n"
    stats_string += f"  % coverage achieved    : {cov_per}%\n"
    stats_string += f"{'-'*60}\n"
    TU.TIMELOG(fuzz_time, stats_string, False, True)

    # save final coverage
    with open(CONFIG_PT['merged_cov_file'], 'w') as fp: json.dump(merged_cov_dict, fp, indent=2)


"""
Deletes any previous log files and starts TheHuzz
"""
def main(prog_time): 

    print(f"[-------] Checking compute_cov_achieved function")
    success = TU.check_compute_cov_achieved_func()
    print(f"[-------] Checking compute_cov_achieved function done")


    print(f"[-------] Deleting previous log files")
    TU.delete_dir(CONFIG.pt['outputs_run_dir'], CONFIG.force_delete) 
    TU.delete_dir(CONFIG.pt['tmp_outputs_run_dir'], CONFIG.force_delete) 
    TU.delete_dir(CONFIG.pt['trash_run_dir'], CONFIG.force_delete) 
    subprocess.call([ 'mkdir', CONFIG.pt['all_progs_dir'] ])
    subprocess.call([ 'mkdir', CONFIG.pt['sim_store_dir'] ])
    print(f"[-------] Deleting previous log files done")

    
    print(f"[-------] Setup simulation repositories")
    sim_dirs = [CONFIG.CORE.pt['sim_dir_t'].substitute(tno=i) for i in range(CONFIG.no_threads)]
    assert os.path.isdir(CONFIG.CORE.pt['sim_dir_t'].substitute(tno=0))\
                        , f"no simulation repos found: {CONFIG.CORE.pt['sim_dir_t'].template}"
    repos_to_create = [repo for repo in sim_dirs if not os.path.isdir(repo)]
    for repo in tqdm(repos_to_create, desc="[-------] creating simulation repositories"): 
        subprocess.call([ 'cp', '-r', CONFIG.CORE.pt['sim_dir_t'].substitute(tno=0), repo ])
    print(f"[-------] Setup simulation repositories done")

    prog_time.reset_start_time() # count time only after deleting previous logs

    # set the log file
    debug_level = lg.DEBUG if CONFIG.debug_print else lg.INFO 
    lg.basicConfig(filename=CONFIG.pt['fuzz_log_file'], filemode='w', level=debug_level)

    TU.TIMELOG(prog_time, f" Getting the parameters for the fuzzer", False, True)
    optimizer_sol, first_opcode_list, opcode_list = get_thehuzz_parameters(CONFIG.core_name, CONFIG.pt['opt_sol_file'])
    TU.TIMELOG(prog_time, f" Getting the parameters for the fuzzer", True, True)


    TU.TIMELOG(prog_time, f" Running TheHuzz on given benchmark, {CONFIG.core_name}", False, True)
    
    run_thehuzz(prog_time, CONFIG.pt, CONFIG.CORE.pt\
              , CONFIG.core_name,           CONFIG.core_instance_list, CONFIG.max_fuzz_time\
              , CONFIG.max_fuzz_progs,      CONFIG.target_cov\
              , CONFIG.sim_batch_size\
              , optimizer_sol,              first_opcode_list, opcode_list\
              , CONFIG.num_inst_in_prog,    inst_list_all_w_ext\
              , CONFIG.num_nops_at_start,   CONFIG.num_nops_at_end\
              , CONFIG.detecting_bugs,      CONFIG.no_threads\
              , CONFIG.CORE.tot_sim_time,   CONFIG.cov_enable\
              , CONFIG.cov_types,           CONFIG.vdb_cov_files, CONFIG.store_elf_file\
              , CONFIG.store_trace_file,    CONFIG.store_cov_file\
              , CONFIG.num_times_to_mut, nop_inst_bin_32, CONFIG.feedback_cov_types\
              , CONFIG.debug_print)

    TU.TIMELOG(prog_time, f" Running TheHuzz on given benchmark, {CONFIG.core_name} done", False, True)




if __name__ == '__main__': 

    # custom time object
    prog_time = TU.Mytime()

    # get variables from config file or dict, and update any present in args
    CONFIG = getCONFIG(config, configType='file')
   
    # uncomment the line below to see all the config variables
    #print(CONFIG.printConfig(CONFIG)); exit()

    main(prog_time)






##############################
##### depreciated code #######
##############################

#    print(f"[-------] Checking that the tmp repo is set correctly")
#    ret = subprocess.call(f"source {CONFIG.pt['check_tmp_dir_script']} {CONFIG.no_threads}",shell=True)
#    assert ret == 0, f"Checking tmp repo failed"
#    print(f"[-------] Checking that the tmp repo is set correctly done")

