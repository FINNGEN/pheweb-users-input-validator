#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import argparse
import subprocess
from datetime import datetime
from utils.check_meta import check_meta
from utils.check_stats import check_stats, timing


def main():

    # parse the arguments
    parser = argparse.ArgumentParser(description='Python tool \
        for scanning and fixing errors in meta and stats files.')

    # parameters used for modules and runs specifications
    parser.add_argument('-m', '--metadata', help='Metadata file.', required=True)
    parser.add_argument('-s', '--stats', help='Statistics file.', required=True)
    parser.add_argument('-o', '--outdir', help='Output directory.', required=True)
    parser.add_argument('-d', '--deep', help='Deep scan (default: True).', default=True, 
        type=str2bool, required=False)
    parser.add_argument('-f', '--fix',  help='Fix errors whenever possible (default: True).', 
        type=str2bool, default=True, required=False)

    # analysis started
    start = datetime.now()
    ts = start.strftime("%d%m%YT%H%M")

    # initialize
    meta_fixed_file = stats_fixed_file = stats_lines_with_errors = None
    t_read = t_write = t_sort = ''
    report_stats = report_stats_scan = report_meta = ''
    tot_meta_checks = 4
    tot_stats_checks = 8
    sep_length = 90
    space = 10

    # start the timer
    t1 = time.time()
    mes_start = "\n%s\n" % (" SCAN STARTED AT: %s " % start).center(sep_length, '=')
    sep = "\n%s\n" % ('').center(sep_length, '=')
    print(mes_start)

    # parse arguments and exit if none are provided
    args = vars(parser.parse_args())
    if all(x == None for x in args.values()):
        print("Please, provide all required input arguments (see -h for help)!")
        return

    # initialize variables
    meta = args['metadata']
    stats = args['stats']
    outdir_prefix = args['outdir']
    deep = str2bool(args['deep'])
    fix = str2bool(args['fix'])

    # proceed with the scan
    infomes="\nArguments: \n\tDeep scan  - %s\n\tFix issues - %s" % (deep, fix)

    # check if provided input meta file exists
    if not os.path.exists(meta):
        message = "File %s not found. Please, check path to the provided metadata file. Terminate." % meta
        sys.exit(message)
    
    # check if provided input stats file exists
    if not os.path.exists(stats):
        message = "File %s not found. Please, check path to the provided stats file. Terminate." % stats
        sys.exit(message)
    
    # check if provided output folder exists
    if not os.path.exists(outdir_prefix):
        message = "Output folder %s doesn't exist. Please, check the path. Terminate." % outdir_prefix
        sys.exit(message)

    # create output folder
    outdir = os.path.join(outdir_prefix, "scan%s" % ts)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # check meta
    print("[INFO]  Check metadata file.")
    try:
        report_meta, meta_fixed_file = check_meta(meta, stats, outdir, fix)
    except Exception as e: 
        print("Error occurred during processing of the metadata file :: %s" % e)
    print("[INFO]  Metadata file check completed.")

    # check stats
    print("[INFO]  Check stats file.")
    report_stats, report_stats_scan, t_read, t_write, \
            t_sort, stats_fixed_file, stats_lines_with_errors = check_stats(stats, deep, fix, outdir)
    # try:
    #     report_stats, report_stats_scan, t_read, t_write, \
    #         t_sort, stats_fixed_file, stats_lines_with_errors = check_stats(stats, deep, fix, outdir)
    # except Exception as e:
    #     print("Error occurred during processing of the stats file :: %s" % e)
    print("[INFO]  Stats file check completed.")
    
    # calculate totals:
    mstats = count(report_meta)
    sstats = count(report_stats)
    meta_md5sum = get_md5sum(meta)
    stats_md5sum = get_md5sum(stats)
     
    # stats table header
    chl = max([len(os.path.basename(stats)) + 5, len(os.path.basename(meta)) + 5, len("Total successful scans:") + 5])
    th = "FILENAME".ljust(chl) + "PASSED".ljust(space) + "FAILED".ljust(space) + \
        "FIXED".ljust(space) + "MD5SUM"
    
    # totals for the meta
    mcounts = [str(s).ljust(space) for s in [mstats[0], 
        mstats[1], mstats[2], meta_md5sum]]
    mcounts = [os.path.basename(meta).ljust(chl)] + mcounts

    # totals for the stats
    scounts = [str(s).ljust(space) for s in [sstats[0], 
        sstats[1], sstats[2], stats_md5sum]]
    scounts = [os.path.basename(stats).ljust(chl)] + scounts

    # total
    tot_success = mstats[0] + mstats[2] + sstats[0] + sstats[2]
    tot_success_mes = "%s%s / %s" % ("Total successful scans:".ljust(chl), tot_success, tot_meta_checks + tot_stats_checks) 
    table_border = "%s" % ('').center(len(tot_success_mes), '-')
    tot_stats = '\n'.join([th, ''.join(mcounts), ''.join(scounts)])

    # save report
    fout = os.path.join(outdir, "report.log")

    # stop the timer
    end = datetime.now()
    t2 = time.time()
    t_total = timing(t1, t2)

    # prepare messages
    mes_end = "\nSCAN ENDED AT: %s\n" % end
    outputs = [os.path.abspath(fout), meta_fixed_file, stats_fixed_file, stats_lines_with_errors]
    outputs = [o for o in outputs if o is not None]
    mes_output = "\nOUTPUT FILES:\n\t%s" % '\n\t'.join(outputs)    
    mes_input = "\nINPUT FILES:\n\t%s" % '\n\t'.join([os.path.abspath(meta), os.path.abspath(stats)])

    # save the reports
    exec_time = "Read stats file %s\nWrite stats file %s\nSort stats file %s\nTotal %s\n" % \
        (t_read.lower(), t_write.lower(), t_sort.lower(), t_total.lower())
    report = "\n".join([mes_start, report_meta, report_stats, sep,
        tot_stats, table_border, tot_success_mes, sep, mes_input, mes_output, infomes, 
        report_stats_scan, sep, mes_end, exec_time])
    
    # save files
    with open(fout, 'w') as f:
        f.write(report)

    # print on the screen
    print('\n'.join([sep, tot_stats, table_border, tot_success_mes, 
                    sep, mes_input, mes_output, mes_end, exec_time]))
    

def get_md5sum(filename):
    output = subprocess.check_output("md5sum %s" % filename, 
        shell=True, executable='/bin/bash')
    md5sum = output.decode().split(' ')[0]
    return md5sum


def count(report):
    passed = len(re.findall("PASS", report))
    failed = len(re.findall("FAIL", report))
    fixed = len(re.findall("FIXED", report))
    remain = len(re.findall("SKIP", report))
    return (passed, failed, fixed, remain)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    main()