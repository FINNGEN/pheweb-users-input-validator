# -*- coding: utf-8 -*-

import io
import os
import re
import time
import xopen
import mgzip
import pysam
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool

warnings.simplefilter(action='ignore', category=FutureWarning)
    
def check_stats(filename, deep, fix, outdir):

    t1 = time.time()
    chunk_size = 10_000_000 if not deep else 100_000_000

    # combine summary across the stats
    report = {}
    opened = False

     # check if file can be opened
    if is_gz_file(filename):
        report.update({'IS_COMPRESSED': make_summary("IS_COMPRESSED", None, None, (), (), 'PASS')})
        # read header of the file
        try:
            tbl = pysam.BGZFile(filename)
            first_row = tbl.readline().decode()
        except Exception as e:
            report.update({'IS_NOT_COMPRESSED': make_summary("IS_NOT_COMPRESSED", None, None, (), ())})

        else: 
            # 3. Check if file is tab-delimited
            if is_tab_separated(first_row):
                report.update({'IS_TAB_DELIMITED': make_summary("IS_TAB_DELIMITED", None, None, (), (), 'PASS')})
                delim = '\t'
            elif is_comma_separated(first_row):
                delim = ','
                report.update({'IS_NOT_TAB_DELIMITED': make_summary("IS_NOT_TAB_DELIMITED", None, None, (), ())})
            elif is_space_separated(first_row):
                delim = ' '
                report.update({'IS_NOT_TAB_DELIMITED': make_summary("IS_NOT_TAB_DELIMITED", None, None, (), ())})
            opened = True
    else:
        report.update({'IS_NOT_COMPRESSED': make_summary("IS_NOT_COMPRESSED", None, None, (), ())})
    
    # if file cannot be opened - return report as is
    if not opened:
        return report
    
    columns = first_row.split(delim)

    # open file
    print("[INFO]  Start reading stats file in chunks.")
    text_chunks = []
    chunk = 0
    rest = ''
    with xopen.xopen(filename, "rb") as f:
        while 1:
            block = f.read(chunk_size)
            if not block:
                break
            
            # print("\tREAD CHUNK:", chunk)
            text = block.decode(errors = 'ignore')

            # append text from prev iteration
            text = rest + text
            if not text.endswith('\n'):
                text, rest = text.rsplit('\n', 1)
            else: 
                rest = ''
            
            # scan and prefix the chunk
            text_chunks.append(text)
            
            # increase chunk count
            chunk += 1

            # break if shallow is activated and check only first chunk
            if not deep:
                break
    
    # check the last line of the stat file
    if rest != '' and deep:
        print("[WARN]  Last line of the stats file is not complete - it will be skipped.")
        report.update({'DOES_NOT_HAVE_COMPLETE_LAST_LINE': make_summary("DOES_NOT_HAVE_COMPLETE_LAST_LINE", None, None, (), ())})

    print("[INFO]  Start scanning stats file in chunks. This might take some time.")

    struct = [ (t, columns, delim) for i, t in enumerate(text_chunks)]
    with Pool() as pool:
        # save fixed file to the output folder
        res = pool.map(multi_run_wrapper, struct)    
    
    # combine chunks and re-index results obtained during scan
    report_add, data = combine_chunks(res)
    report.update(report_add)
    
    # print execution time
    t2 = time.time()
    t_read = timing(t1, t2)
    print("[INFO]  Finished scanning stats file. %s"  % t_read)

    # create report and fix if needed
    data_fixed, r1, r2 = create_report_and_fix(data, report, fix) 
    failed_n = len(re.findall("FAIL", r1))
    fixed_n = len(re.findall("FIXED", r1))

    # write result file
    t1 = time.time()
    try:

        # save fixed file to the output folder
        fout = os.path.join(outdir, os.path.basename(filename))
        if not deep:
            fout = fout + '.CHUNK.gz'
        if failed_n > 0:
            print("[WARN]  Found errors in the stats file. Check details in the final report.")
        else:
            print("[INFO]  Found NO errors in the stats file.")

        # write file
        if fix and fixed_n:
            print("[WARN]  Fixed some errors in the stats file. Check details in the final report.")
            write_stats(fout, data_fixed, num_cpus=0)    
            fout = os.path.abspath(fout)
        else:
            print("[INFO]  No fixed were made to the stats file.")
        
        if fixed_n == 0 and failed_n == 0:
            print('[INFO]  No errors found in the stats file.')
            fout = None

    except Exception as e:
        print("Error occurred while writing output file :: %s. Skip fixing." % e)
    
    # time
    t2 = time.time()
    t_write = timing(t1, t2)
    if fix and w > 0:
        print("[INFO]  Finished writing stats file: %s"  % t_write)

    return r1, r2, t_read, t_write, fout

def flatten(l):
    new_list = []
    for sublist in l:
        if isinstance(sublist, list):
            for item in sublist:
                new_list.append(item)
        else:
            new_list.append(sublist)
    return new_list

def combine_chunks(multithreads_res):

    print("[INFO]  Combining %s scanned chunks." % len(multithreads_res))
    
    # re-enumerate the row ids
    ids = [r[1].shape[0] for r in multithreads_res]
    pandas_id = [0] + list(np.cumsum(ids))
    pandas_id.pop()
    offset = [2] + list(np.cumsum(ids) + 2)
    offset.pop()
    
    # combine report from chunks
    report_combined = []
    issues = []
    j = 0
    tot_reports = {i: multithreads_res[i][0] for i in list(range(len(multithreads_res)))}
    for i in tot_reports.items():
        offset_rowid = offset[i[0]]
        pandas_rowid = pandas_id[i[0]]
        for k in list(range(len(i[1]))):
            if len(i[1][k]) > 0 and len(i[1][k]['row_id']) > 0:
                item = i[1][k].copy()
                item['pandas_row_id'] = [row + pandas_rowid for row in item['row_id']]
                item['row_id'] = [row + offset_rowid for row in item['row_id']]
                if item['colname'] is not None:
                    issues.append(item['issue'] + '_' + item['colname'])
                else:
                    issues.append(item['issue'])
                report_combined.append(item)
        j += 1
    
    issues = list(set(issues))
    report_final = {}
    for item in report_combined:
        if item['colname'] is not None:
            key = item['issue'] + '_' + item['colname']
        else:
            key = item['issue']
        if key not in report_final.keys():
            report_final.update({key: item})
        else:
            if item['invalid_entry'] is not None:
                report_final[key]['invalid_entry'] = report_final[key]['invalid_entry'] + item['invalid_entry']
            report_final[key]['row_id'] = report_final[key]['row_id'] + item['row_id']
            report_final[key]['pandas_row_id'] = report_final[key]['pandas_row_id'] + item['pandas_row_id']

    # combine data:
    data = pd.DataFrame()
    for i in list(range(len(multithreads_res))):
        data = data.append(multithreads_res[i][1])
    
    return report_final, data

def multi_run_wrapper(args):
   r, d = chunk_scan(*args)
   return r, d

def chunk_scan(text, columns, delim):

    report = []

    # special chars
    entries, ids = scan_special_chars(text)
    for sp in entries:
        text = text.replace(sp, '')
        if 'chrom' in text[0:15] and sp == '#':
            text = text.replace('chrom', '#chrom')
        
    # read data 
    if 'chr' in text[0:15]:
        dat = pd.read_csv(io.StringIO(text), sep=delim, dtype={0: 'str'}, low_memory=False)
    else:
        dat = pd.read_csv(io.StringIO(text), sep=delim, dtype={0: 'str'}, header=None, low_memory=False)
        dat.columns = columns
    
    if len(entries) > 0:
        report.append({'col_id': None, 'colname': None,  'issue': 'SPECIAL_CHARACTERS', 
                       'status': 'FIXED', 'row_id': ids, 'invalid_entry': entries})

    # check if first column contain correct values
    entries, ids = get_invalid(dat, '#chrom', lambda x: x not in [str(item) 
        for item in list(range(1,25))] + ['X', 'Y', 'M', 'MT'])
    s = make_summary("INVALID_FORMATTING", list(dat.columns).index('#chrom'), '#chrom', entries, ids)
    report.append(s)
    
    # check if pval column contain correct values
    entries, ids = get_invalid(dat, 'pos', lambda x: not isint(x))
    s = make_summary("INVALID_FORMATTING", list(dat.columns).index('pos'), 'pos', entries, ids)
    report.append(s)

    # check if ref column contain correct values
    entries, ids = get_invalid(dat, 'ref', lambda x: not bool(re.match("[ATCG]+", x)))
    s = make_summary("INVALID_FORMATTING", list(dat.columns).index('ref'), 'ref', entries, ids)
    report.append(s)

    # check if pval column contain correct values
    try:
        # entries, ids = get_invalid(dat, 'pval', lambda x: not ((x >= 0 and x <= 1) or np.isnan(x)))
        entries, ids = get_invalid(dat, 'pval', lambda x: not (x >= 0 and x <= 1))
    except TypeError as e:
        entries, ids = get_invalid(dat, 'pval', lambda x: not bool(re.match("[0-9+.-]+", x)))
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('pval'), 'pval', entries, ids)
    else:    
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('pval'), 'pval', entries, ids)
    report.append(s)
    
    # check if pval column contain correct values
    entries, ids = get_invalid(dat, 'mlogp', lambda x: not (isfloat(x) and x != np.inf))
    s = make_summary("INVALID_FORMATTING", list(dat.columns).index('mlogp'), 'mlogp', entries, ids)
    report.append(s)
    
    # check if pval column contain correct values
    try:
        entries, ids = get_invalid(dat, 'beta', lambda x: not (isfloat(x) and not np.isnan(x)))
    except TypeError as e:
        entries, ids = get_invalid(dat, 'beta', lambda x: not bool(re.match("[0-9+.-]+", x)))
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('beta'), 'beta', entries, ids)
    else:
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('beta'), 'beta', entries, ids)
    report.append(s)

    # check if pval column contain correct values
    try:
        entries, ids = get_invalid(dat, 'sebeta', lambda x: not ((isfloat(x) or isint(x)) and not np.isnan(x)))
    except TypeError as e:
        entries, ids = get_invalid(dat, 'sebeta', lambda x: not bool(re.match("[0-9+.-]+", x)))
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('sebeta'), 'sebeta', entries, ids)
    else:
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('sebeta'), 'sebeta', entries, ids)
    report.append(s)

    # check if pval column contain correct values, will pick NAs as well
    try:
        entries, ids = get_invalid(dat, 'af_alt', lambda x: not (x >= 0 and x <= 1))
    except TypeError as e:
        entries, ids = get_invalid(dat, 'af_alt', lambda x: not bool(re.match("[0-9+.-]+", x)))
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt'), 'af_alt', entries, ids)
    else:
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt'), 'af_alt', entries, ids)
    report.append(s)

    # check if pval column contain correct values, will pick NAs as well
    try:
        entries, ids = get_invalid(dat, 'af_alt_cases', lambda x: not (x >= 0 and x <= 1))
    except TypeError as e:
        entries, ids = get_invalid(dat, 'af_alt_cases', lambda x: not bool(re.match("[0-9+.-]+", x)))
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt_cases'), 'af_alt_cases', entries, ids)
    else:
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt_cases'), 'af_alt_cases', entries, ids)
    report.append(s)

    # check if pval column contain correct values, will pick NAs as well
    try:
        entries, ids = get_invalid(dat, 'af_alt_controls', lambda x: not (x >= 0 and x <= 1))
    except TypeError as e:
        entries, ids = get_invalid(dat, 'af_alt_controls', lambda x: not bool(re.match("[0-9+.-]+", x)))
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt_controls'), 'af_alt_controls', entries, ids)
    else:
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt_controls'), 'af_alt_controls', entries, ids)
    report.append(s)

    # return 
    return report, dat

def scan_special_chars(t):
    # split lines
    lines = t.split('\n')
    sizes = [len(l) + 1 for l in lines]
    sizes_cumsum = [0] + list(np.cumsum(sizes))
    st = sizes_cumsum[0:(len(sizes_cumsum)-1)]
    en = sizes_cumsum[1:len(sizes_cumsum)]

    # scan for the not allowed chars
    accepted_chars = re.compile("[^A-Za-z0-9-. +\t\n]")
    finder = re.finditer(accepted_chars, t) 
    if '#chr' in t:
        tmp = t.split('\n')
        first_line_len = len(tmp[0])
    else:
        first_line_len = 0
    # iterate through all matches
    line_numb = []
    illegal_chars = []
    # iterate through all matches
    count = 0
    for match in finder:
        char_pos = match.start(0) + 1
        if char_pos >= first_line_len:
            ids = np.logical_and(match.start(0) > np.array(st), match.start(0) < np.array(en))
            ln = np.where(ids)[0][0] - 1
            illegal_chars.append(match.group())
            line_numb.append(ln)
            if count > 50:
                print('[INFO]   Too many special characters - limiting to first 50.')
                break
            count += 1

    return illegal_chars, line_numb

def scan_sorted(df):

    # specify the order
    df.reset_index(drop=True, inplace=True) 
    chrs = [str(item) for item in list(range(1,25))] + ['X', 'Y', 'M', 'MT']
    d = {'chr_numb': list(range(1,len(chrs)+1)), 'chrs': chrs}
    chr_numbers = pd.DataFrame(d)
    chr_numbers.index = chr_numbers['chrs']
    
    # append the numeric chr column and sort by it and pos    
    df_sorted = df.copy()
    df_sorted['chr_numb'] = list(chr_numbers.loc[df_sorted.iloc[:, 0]]['chr_numb'])
    df_sorted = df_sorted.sort_values(by=['chr_numb', 'pos'], ascending = [True, True])
    
    # prepare summary
    summary = []
    if not all(df.index == df_sorted.index):
        unsorted_ids = df.index[df.index != df_sorted.index]
        first_unsorted = unsorted_ids[0]
        summary = {'col_id': None, 
                   'colname': None,
                   'issue': 'UNSORTED', 
                   'status': 'FAIL', 
                   'row_id': [first_unsorted + 2],
                   'pandas_row_id': [first_unsorted],
                   'invalid_entry': None}

    return df_sorted[list(df.columns)], summary
    
def get_invalid(df, colname, func):
    ids = list(df.loc[df.loc[:, colname].apply(func)].index)
    entries = list(df.loc[ids, colname])
    return entries, ids

def make_summary(issue, col_id, colname, entries, ids, status='FAIL'):
    return {'col_id': col_id, 
            'colname': colname,
            'issue': issue,
            'status': status, 
            'row_id': ids,
            'invalid_entry': entries}

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def isint(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

def extract_lnumb(tfull, char):
    lines = tfull.split('\n')
    for i in list(range(len(lines))):
        l = lines[i]
        if l.find(char) != -1:
            return i

def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'

def is_tab_separated(line):
    return len(line.split('\t')) > 1

def is_comma_separated(line):
    return len(line.split(',')) > 1

def is_space_separated(line):
    return len(line.split(' ')) > 1

def timing(t1, t2):
    td = t2 - t1
    m = "Execution time: " + "%s mins" % \
         round(td/60, 2) if td > 60 else "%s sec" % round(td, 2)
    return m

def fix_chrom_col(df, ids):
    df.iloc[ids, 0] = df.iloc[ids, 0].apply(lambda x: re.sub(r"[^0-9]+", "", x))
    return df

def format_entries(report, key, status):
    z = zip(report[key]['row_id'], report[key]['invalid_entry'])
    space = 20
    lines = [("\tLine: %s" % e[0]).ljust(space) + 
             ("\tEntry: %s" % e[1]).ljust(space) + 
             ("\tStatus: %s" % status).ljust(space) for e in z]
    warn = ' - print first 50, total of %s' % len(lines) if len(lines) >= 50 else ''
    if report[key]['colname'] is not None:
        colname = report[key]['colname']
        lines = ["\n%s:" % colname.upper(), '\n'.join(lines[0:49])]
    else:
        lines = ['\n', '\n'.join(lines[0:49])]
    return lines, warn
     
def filter(ind, arr):
    arr_updated = [elem for j, elem in enumerate(arr) if j in ind]
    return arr_updated

# fix: special chars, mising, chr prefix, sorting, column order
def create_report_and_fix(df, report, fix):

    report_issues_all = list(report.keys())
    general_report  = [] 
    detailed_report = []
    cols_order = ["#chrom", "pos", "ref", "alt", "pval", "mlogp", "beta", 
                  "sebeta", "af_alt", "af_alt_cases", "af_alt_controls"]
    
    # special characters
    key = 'SPECIAL_CHARACTERS'
    if key in list(report.keys()):
        report_issues_all.remove(key)
        status = 'FIXED' if fix else 'FAIL'
        lines, warn = format_entries(report, key, status.lower())
        message = "%sSpecial characters found in the stats file%s.\n        Check details under SPECIAL CHARACTERS section below." \
            % (('[%s]' % status).ljust(8, ' '), warn)
        header = ['\n', (' SPECIAL CHARACTERS ').center(90, '=')]
        detailed_report = detailed_report + header + lines[0:49] 
    else:
        message = "[PASS]  No special characters were found in the stats file."
    general_report.append(message)

    # fix chromosome characters
    check_sort = True
    invalid_entries_keys = []
    k = 'INVALID_FORMATTING_#chrom'
    if k in report.keys():
        if fix:
            report_issues_all.remove(k)
            invalid_entries_keys.append(k)
            df = fix_chrom_col(df, report[k]['pandas_row_id'])
            ids_fixed = get_invalid(df.iloc[report[k]['pandas_row_id'], :], '#chrom', 
                lambda x: x not in [str(item) for item in list(range(1,25))] + ['X', 'Y', 'M', 'MT'])
            if len(ids_fixed[0]) > 0:
                message = "[FAIL] Chromosome colum contains invalid entries and \n        validator wasn't" \
                        + "able to fix that. Check details under INVALID ENTRIES section below."
                check_sort = False
            else:
                message = "[FIXED] Chromosome colum contains invalid entries. Check details under INVALID ENTRIES section below."
        else:
            message = "[FAIL] Chromosome colum contains invalid entries. Check details under INVALID ENTRIES section below."
            check_sort = False
    else:
        message = "[PASS]  No invalid entries in the column #CHROM were found in the stats file."
    general_report.append(message)

    # invalid entries in columns, exclude missing values
    missing_vals_keys = []
    reports_invalid = {}
    reports_missing = {}
    col_ids=[1,2] + list(range(4,len(cols_order)))
    for col in list(np.array(cols_order)[col_ids]):
        key = "INVALID_FORMATTING_%s" % col
        if key in list(report.keys()):
            entries = report[key]['invalid_entry']            
            report_nas = report[key].copy()
            report_invalid = report[key].copy()

            # ids of nas and invalid
            nas_ids = [i for i,elem in enumerate(entries) if np.isnan(elem)]
            invalid_ids = [i for i,elem in enumerate(entries) if not np.isnan(elem)]

            # filter indices
            for k in ['row_id', 'invalid_entry', 'pandas_row_id']:
                report_nas[k] = filter(nas_ids, report_nas[k])
                report_invalid[k] = filter(invalid_ids, report_invalid[k])
            
            # append to reports for later summary of all cols in a single section
            if len(nas_ids)  > 0 :
                reports_missing[key] = report_nas
                missing_vals_keys.append(key)
            
            # append to reports for later summary of all cols in a single section
            if len(invalid_ids) > 0:
                reports_invalid[key] = report_invalid
                invalid_entries_keys.append(key)

            # remove from processed keys
            report_issues_all.remove(key)
    
    # invalid entries in columns
    if len(invalid_entries_keys) > 0:
        message = "[FAIL]  Invalid entries found in the columns of the stats file.\n" + \
                  "        Check details under INVALID ENTRIES section."
        tuples = [format_entries(reports_invalid, k, "fail") for k in invalid_entries_keys]
        lines = [t[0] for t in tuples]
        header = ['\n', (' INVALID ENTRIES ').center(90, '=')]
        detailed_report = detailed_report + header + lines
    else:
        message = "[PASS]  No invalid entries in columns of the stats file were found."
    general_report.append(message)

    # missing values
    cols_fix_nas = ["beta", "sebeta", "af_alt", "af_alt_cases", "af_alt_controls"]
    if len(missing_vals_keys) > 0:
        if fix:
            df[cols_fix_nas] = df[cols_fix_nas].fillna(value = 0.5)
            status = "FIXED"
        else:
            status = "FAIL"
        message = "%sMissing values found in columns 7-11 - substitute with 0.5.\n        Check details under MISSING VALUES section below." % ('[%s]' % status).ljust(8, ' ')
        tuples = [format_entries(reports_missing, k, status.lower()) for k in missing_vals_keys]
        lines = [t[0] for t in tuples]
        header = ['\n', (' MISSING VALUES ').center(90, '=')]
        detailed_report = detailed_report + header + lines
    else:
        message = "[PASS]  No missing values found in columns 7-11 of stats file."
    general_report.append(message)
    
    # check if data is sorted
    if check_sort:
        df, summary = scan_sorted(df)

        if len(summary) > 0:
            print("[WARN]  Unsorted positions were found in the data.")
            if fix:
                print("[INFO]  Sorting the data. This might take some time.")
                df = sort_table(df)
                message = '[FIXED] Unsorted positions found in the stats file. Fist unsorted row - %s.' % summary['row_id'][0]
            else:
                message = '[FAIL]  Unsorted positions found in the stats file. Fist unsorted row - %s.' % summary['row_id'][0]
        else:
            message = '[PASS]  No unsorted positions found in the stats file.'
    else:
        message = "[FAIL]  Unsorted positions were found in the data. " + \
            "\nValidator wasn't able to fix this problem due to invalid positions in the #chrom column."
    general_report.append(message)
    
    # if not all required colums are in the table or order is wrong
    if list(df.columns) != cols_order:

        # if all required columns are in the table but in the wrong order or there are some extra columns
        if len(set(cols_order).difference(set(df.columns))) == 0:

            # if can be re-ordered and selected, do it 
            if fix:
                df = df[cols_order]
                message = "[FIXED] Fixed columns order/number in the stats file."
            
            # otherwise, fail
            else:
                message = "[FAIL]  Wrong columns order or columns number: check that your stats file contains exacly 11 required columns " + \
                      "        as described in the FinnGen Analyst Handbook."
        
        # NOT all required columns are in the table
        else:
            message = "[FAIL]  Not all required columns are in the table: check that your stats file contains exacly 11 required columns " + \
                      "        as described in the FinnGen Analyst Handbook."

    # if all required colums are in the table AND order is right
    else:
        message = "[PASS]  Correct columns order/number in the stats file."

    general_report.append(message)
    
    # add other errors
    for key in report_issues_all:
        if key != 'UNSORTED':
            message = "[%s]  File %s." % (report[key]['status'], key.replace('_', ' ').lower())
            general_report.append(message)

    general_report.sort()
    r1 = '\n'.join(general_report) + '\n'
    r2 = '\n'.join(flatten(detailed_report)) + '\n'

    return df, r1, r2
    
def sort_table(df):
    
    chrs = [str(item) for item in list(range(1,25))] + ['X', 'Y', 'M', 'MT']
    d = {'chr_numb': list(range(1,len(chrs)+1)), 'chrs': chrs}
    chr_numbers = pd.DataFrame(d)
    chr_numbers.index = chr_numbers['chrs']
    
    # append the column and sort data
    col_id = df.shape[1]
    df_sorted = df.copy()
    df_sorted[col_id] = list(chr_numbers.loc[df_sorted.iloc[:, 0]]['chr_numb'])
    df_sorted = df_sorted.sort_values([col_id, 
        df_sorted.columns[1]], ascending = [True, True])

    # drop the column
    df_sorted = df_sorted.drop(columns=[col_id])
    return df_sorted
    
def  write_stats(fileout, df, num_cpus=0):

    print("[INFO]  Start writing fixed stats file to the file. This might take a few mins.")

    # start the timer
    df.reset_index(drop=True, inplace=True) 
    st, e = chunk_df_for_writing(df, step=100000)

    # convert to strings
    h = '\t'.join(str(elem) for elem in list(df.columns))
    with mgzip.open(fileout, 'wb', thread=num_cpus, blocksize=2 * 10 ** 8) as fw:

        # go through the chuck and combine rows
        for i, j in enumerate(st):
            if i == (len(st)-1):
                start_i = st[i]
                end_i = e[i]
            else:
                start_i = st[i]
                end_i = e[i] - 1
            
            chunk = df.iloc[start_i:end_i, ].copy()
            res = chunk.apply(lambda y: '\t'.join(str(elem) for elem in y), axis = 1)

            # append header if processing first chunk
            if i == 0:
                res.loc[-1] = h

                # shifting index
                res.index = res.index + 1 
                res.sort_index(inplace=True) 
            
            # concat to a single string
            res_list = res.tolist()
            res_str = '\n'.join(res_list)
            out = fw.write(res_str.encode())

def chunk_df_for_writing(df, line_numb=None, step=None):
    if step is None:
        step = round(df.shape[0] / line_numb)
    v = list(range(0, df.shape[0], step))
    v.append(df.shape[0]-1)
    v = np.array(v)
    start = v[list(range(0, len(v)-1))]
    end = v[list(range(1, len(v)))]
    return start, end



