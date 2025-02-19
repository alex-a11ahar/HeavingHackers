"""
This script is used to generate program files
- Notes: 

- TODOs: 
"""

import json, os, subprocess, re, json
import time, copy
from tqdm import tqdm
import logging as lg # critical, error, warning, info, debug

import fuzz

testcase_template = {'id': None\
                    , 'hex_file': None, 'times_mutated': 0}

"""
Deletes the target file
"""
def delete_file(target_file, force_delete=False, action='delete'):
    if not os.path.exists(target_file): # just create if file doesnt exist
        return 'deleted'
    if action == 'delete': 
        if force_delete:
            confirm = 'y'
        else: 
            confirm = input("Are you sure you want to delete "+ target_file + "? (y,n,a) : ")

        if (confirm == 'y'):
            subprocess.call("rm -rf " + target_file,shell=True)
            return 'deleted'
        elif (confirm == 'a'): # abort
            exit()
        else: 
            return 'not_deleted'
    else: 
        print("ERROR: Unknown action:", action)
        exit()
    

"""
Deletes the target dir
- Actions: 
    - delete: deletes the target dir
    - move: moves the target dir to a trash dir
"""
def delete_dir(target_dir, force_delete=False, action='delete', trash_dir='/tmp/'):
    if not os.path.isdir(target_dir): # just create if dir doesnt exist
        subprocess.call("mkdir -p "  + target_dir,shell=True)
        return 'deleted'
    if action in ['delete', 'move']: 
        if len(os.listdir(target_dir)) > 0:
            if force_delete:
                confirm = 'y'
            else: 
                confirm = input("Are you sure you want to delete "+ target_dir + "? (y,n,a) : ")

            if (confirm == 'y'):
                if action == 'delete': 
                    subprocess.call("rm -rf " + target_dir,shell=True)
                else: 
                    subprocess.call([ 'mv', target_dir, trash_dir ])
                subprocess.call("mkdir -p "  + target_dir,shell=True)
                return action
            elif (confirm == 'a'): # abort
                exit()
            else: 
                return 'not_deleted'
        else: 
            subprocess.call("rm -rf " + target_dir,shell=True)
            subprocess.call("mkdir -p "  + target_dir,shell=True)
            return 'deleted'
    else: 
        print("ERROR: Unknown action:", action)
        exit()

"""
Gets a sorted filelist of all the files with a given pattern in a dir
"""
def get_files_in_dir(target_dir, pattern=""):
    # get a list of all the files
    all_filelist = os.listdir(target_dir)

    # filter out files based on the pattern
    filelist = []
    for filename in all_filelist:
        if re.match(pattern, filename):
            filelist.append(filename)

    # sort the files
    filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # add full path to the names
    filelist_fullpath = [os.path.join(target_dir, filename) for filename in filelist]

    return filelist_fullpath

"""
"""
def update_json_file(json_file, input_data): 

    data_changed = False

    if os.path.exists(json_file):
        with open(json_file, 'r') as fp:
            data = json.load(fp)
    else:
        data = {}
        data_changed = True
    
    for key, value in input_data.items():
        if key in data.keys():
            if data[key] != value:
                data[key] = value
                data_changed = True
        else: 
            data[key] = value
            data_changed = True


    if data_changed: # update json file only if it is changed to save time
        with open(json_file, 'w') as fp: 
            json.dump(data, fp, indent=2)


def dummy1(a,n): 
    #return [s.update({'hex_file':a[0]['hex_file']}) for s in a[:n]]
    b = copy.deepcopy(a)
    for i in range(len(b)): b[i]['hex_file'] = a[0]['hex_file']; return [s for s in b[:n]]


"""
"""
def change_extension(filepath, new_extension): 
    basepath = os.path.splitext(filepath)[0]
    return f"{basepath}.{new_extension}"


"""
"""
def log(logfile, data, time=None, overwrite=False): 

    mode = 'w' if overwrite else 'a'
    # get time string, dont update time difference
    time_string = f"[{time.get_time(False)} sec] " if time else ""
   
    with open(logfile, mode) as fp: fp.write(f"{time_string}{data}")


"""
"""
def TIMELOG(time, string, done=False, terminal=False, log_file=True):
    
    done_string = f"done in {time.time_diff()} sec" if done else ""

    if terminal: 
        print(f"[{time.get_time()} sec]{string} {done_string}")

    if log_file: 
        lg.info(f"[{time.get_time()} sec]{string} {done_string}")


"""
"""
class Mytime: 
    def __init__(self, init_time=None): 
        self.start_time = init_time if init_time else time.time()
        self.latest_queried_time = self.start_time

    # return the time from creating this object
    def get_time(self, update=True): 
        if update: # update latest queried time 
            self.latest_queried_time = time.time()
        return round(time.time()-self.start_time, 2)

    # return the time diff from last query
    def time_diff(self, update=True): 
        self.latest_queried_time_prev = self.latest_queried_time
        if update: # update latest queried time 
            self.latest_queried_time = time.time()
        return round(time.time()-self.latest_queried_time_prev, 2)

    # reset start time
    def reset_start_time(self): 
        self.__init__()


"""
"""
class DATABASE: 
    def __init__(self, testcase_dir, hex_file_t):
        self.testcase_dir = testcase_dir
        self.new_testcases = []
        self.simulated_testcases = []
        self.testcases_to_sim = []
        self.hex_file_t = hex_file_t

    # add hex files to database
    def add_testcases(self, filelist, save_filetypes=[]):
        for file_i in filelist:
            # add the testcase to all testcases
            testcase = copy.deepcopy(testcase_template)
            testcase['id'] = self.num_testcases()

            testcase_filename = self.hex_file_t.substitute(fno=testcase['id'])
            testcase['hex_file'] = os.path.join(self.testcase_dir, testcase_filename)
            testcase['times_mutated'] = 0

            self.new_testcases.append(testcase)

            # put the testcase in the daatabase dir
            subprocess.call([ 'mv', file_i, testcase['hex_file'] ])
            for save_file_i in save_filetypes:
                subprocess.call([ 'mv', change_extension(file_i, save_file_i), change_extension(testcase['hex_file'], save_file_i) ])

    # EDIT: this function is imcomplete/incorrect. fix it with the help of the
    # comments provided
    # NOTE THAT THIS FUNCTION IS INSIDE A CLASS
    def get_testcases_to_sim(self, no_testcases):
        # EDIT 1 START: 
        # check to make sure that we have enough testcases to give to the fuzzer. 
        # fuzzer is asking for "no_testcases" number of testcases. You can
        # get the number of testcases in the database by using the
        # self.num_new_testcases() function: 

        # here is pseudo code to help you:
        # if (number of testcases required > number of testcases in the database): 
        #   print an error
        #   exit()  # this command stops the fuzzer

        # EDIT 1 END

        # EDIT 2 START: 
        # all the testcases are stored as elements in the new_testcases array
        # we need to pick the testcases to simulate 
        # for this, we will select the first no_testcases number of testcases from the new_testcases array in this class
        # and put them in the testcases_to_sim array in this class
        # note that you need to use self.variable style syntax to operate on variables inside this class. 

        # here is pseudo code to help you:
        # testcases to give to fuzzer = database array[0:number of testcases needed by the fuzzer]

        self.testcases_to_sim = dummy1(self.new_testcases, no_testcases) # THIS IS DUMMY CODE. REMOVE THIS LINE WHEN YOU WRITE THE CORRECT CODE 
                       # TO GENERATE THE TESTCASES_TO_SIM ARRAY
        # EDIT 2 END

        self.new_testcases = self.new_testcases[no_testcases:]
        self.simulated_testcases += self.testcases_to_sim

        return self.testcases_to_sim 


    def allocate_testcases_to_mut(self, testcases_to_mut): 
        for testcase in testcases_to_mut: 
            testcase['new_hex_files'] = []
            for i in range(testcase['mut_times']): 
                new_testcase = copy.deepcopy(testcase_template)
                new_testcase['id'] = self.num_testcases()
                new_testcase_filename = self.hex_file_t.substitute(fno=new_testcase['id'])
                new_testcase['hex_file'] = os.path.join(self.testcase_dir, new_testcase_filename)
                new_testcase['times_mutated'] = testcase['times_mutated'] + 1

                self.new_testcases.append(new_testcase)
                testcase['new_hex_files'].append(new_testcase['hex_file'])

        return testcases_to_mut

    def sim_done(self): 
        t= 1

    def num_testcases(self): 
        return len(self.new_testcases) + len(self.simulated_testcases)

    def num_new_testcases(self): 
        return len(self.new_testcases)

    def num_testcases_simulated(self): 
        return len(self.simulated_testcases)



#converts the hex files into mem format readble by the ariane and emulator
def hex_to_mem(hex_file, mem_file):

    #Copy hex file to be modified (original file will be unchanged)
    #seed_cpy = "cp -f " + hex_file + " " + mem_file
    #subprocess.call(seed_cpy,shell=True) #runs command
    #out_file = fileinput.input(files=mem_file, inplace=1, backup='.back')
    inf = open(hex_file, 'r')
    in_file = inf.readlines()
    out_file = open(mem_file, 'w')
    
    for line in in_file:
    	#reformats important info so that the fuzzer can parse it
    	line = line.lower()
    	hex_addr = line[1:9]
    	inst1 = line[10:18]#[::-1]
    	inst1 = inst1[6:8] + inst1[4:6] + inst1[2:4] + inst1[0:2] #flip bits b/c of endiniess
    	inst2 = line[19:27]#[::-1]
    	inst2 = inst2[6:8] + inst2[4:6] + inst2[2:4] + inst2[0:2]
    	inst3 = line[28:36]#[::-1]
    	inst3 = inst3[6:8] + inst3[4:6] + inst3[2:4] + inst3[0:2]
    	inst4 = line[37:45]#[::-1]
    	inst4 = inst4[6:8] + inst4[4:6] + inst4[2:4] + inst4[0:2]
    	inst_total = inst1+inst2+inst3+inst4
    	i=0
    	while i < 31:
    		out_file.write(inst_total[i:i+2] + "\n")
    		i=i+2	
    	#print(inst1)
    	#print(inst2)
    	#print(inst3)
    	#print(inst4)
    
    inf.close()	
    out_file.close()	


"""
This function runs some basic tests to check the compute_cov_achieved function
"""
def check_compute_cov_achieved_func(): 
    success = True

    # test1
    merged_cov_dict = {'line': '0001110101101', 'branch': '11100010110000000'\
                      , 'cond': '00', 'fsm': '1111', 'tgl': '1110110'}
    tot_cov_points = 43
    cov_points_ach, cov_per_ach = fuzz.compute_cov_achieved(merged_cov_dict, tot_cov_points)

    if not ((cov_points_ach == 22) and (cov_per_ach >= 51 and cov_points_ach <= 52)):
        success = False
    
    # test2
    merged_cov_dict = {'line': '000111010110100', 'branch': '11110010110000'\
                      , 'cond': '0011', 'fsm': '1111', 'tgl': '11111101111001011011'}
    tot_cov_points = 57
    cov_points_ach, cov_per_ach = fuzz.compute_cov_achieved(merged_cov_dict, tot_cov_points)

    if not ((cov_points_ach == 35) and (cov_per_ach >= 61 and cov_points_ach <= 62)):
        success = False

    if success: 
        print("----------compute_cov_achieved PASSED the basic tests. Note that function could still be incorrect as this is only a basic test.")
    else: 
        print("----------compute_cov_achieved FAILED the basic tests. Ignore this warning if you are yet to fix this function, but if you already fixed it, then it is incorrect. Fix it and try again.")
    


def main():
    temp = 1 # do nothing


if __name__ == '__main__':
    main()
        
        

