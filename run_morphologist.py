#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import sys
import getopt


subject_list, mri_list = [], [] 
data_dir = '/home/leonie/Documents/data'
pisa_dir = '/home/leonie/Documents/data/pisa/t1mri'
subjects = sorted(os.listdir(pisa_dir))
for subject in subjects:
    subject_path = os.path.join(pisa_dir, subject)
    mri_files = os.listdir(subject_path)
    for mrif in mri_files:
        subject_list.append(subject)
        mri_list.append(os.path.join(
            '/home/data/pisa/t1mri', subject, mrif))


###############################################################################
###############################################################################
###############################################################################
# theory: 1918*10/60/24/3 = ~5 days
# memory: 1918*125MB = 240GB

def main(argv):
    subject = None
    t1mri_file = None
    brainvisa_path = None
    try:
        opts, args = getopt.getopt(argv, "n:i:",
                                   ["number=", "iteration="])
    except getopt.GetoptError:
        print('run_morphologist.py -n <number of iteration> -i <iteration>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('run_morphologist.py -n <number of iteration> -i <iteration>')
            sys.exit()
        elif opt in ("-n", "--number"):
            number = int(arg)
        elif opt in ("-i", "--iteration"):
            it = int(arg)
    
    ntot = len(subject_list)
    npart = int(float(ntot)/number)+1
    i = 1
    for subject, mri in zip(subject_list[npart*it:npart*(it+1)], 
                            mri_list[npart*it:npart*(it+1)]):
        print()
        print()
        print(f"--- {subject} ({i}/{npart}) ---")
        print()
        print()
        start_time = time.time()
        cmd = f'docker run --rm -v {data_dir}:/home/data: ' +\
              'leonieborne/morphologist-deepsulci:26082020 /bin/bash -c ' +\
              '". /home/brainvisa/build/bug_fix/bin/bv_env.sh /home/brainvisa/build/bug_fix ' +\
              f'&& python ./morphologist.py -s {subject} -d /home/data/pisa/brainvisa -m {mri}"'
        print(cmd)
        os.system(cmd)
        print()
        print()
        print(f"--- {(time.time() - start_time)/60:.0f} minutes ---")
        print()
        print()
        i += 1
    
if __name__ == "__main__":
    main(sys.argv[1:])