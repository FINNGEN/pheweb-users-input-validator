import io
import os
import re
import time
import xopen
import mgzip
import pysam
import platform
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool

    
def check_stats(filename, deep, fix, outdir):

    t1 = time.time()
    chunk_size = 10_000_000 if not deep else 100_000_000

    # combine summary across the stats
    report = {}
    opened = False

    # check if file can be opened
    if is_gz_file(filename):
        # read header of the file
        try:
            tbl = pysam.BGZFile(filename)
            first_row = tbl.readline().decode()
        except Exception as e:
            report.update({'IS_NOT_COMPRESSED': make_summary("IS_NOT_COMPRESSED", None, None, (), ())})

        else: 
            # 3. Check if file is tab-delimited
            if is_tab_separated(first_row):
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
            # print("\tChunk: %s" % chunk)
            text_chunks.append(text)
            
            # increase chunk count
            chunk += 1

            # break if shallow is activated and check only first chunk
            if not deep:
                break
    
    # check the last line of the stat file
    if rest != '' and deep:
        print("[WARN]  Last line of the stats file is not complete - it will be skipped.")
        report.update({'DOES_NOT_HAVE_COMPLETE_LAST_LINE': 
            make_summary("DOES_NOT_HAVE_COMPLETE_LAST_LINE", None, None, (), ())})

    # scan chunks
    print("[INFO]  Start scanning stats file in chunks. This might take some time.")
    struct = [ (t, columns, delim) for i, t in enumerate(text_chunks)]
    with Pool() as pool:
        res = pool.map(multi_run_wrapper, struct)    

    # fix the chunks
    struct = [ (t[0], t[1], fix) for i, t in enumerate(res)]
    with Pool() as pool:
        res_fixed = pool.map(multi_run_wrapper_fix, struct)
    
    # combine the reports
    report_add = combine_chunks(res_fixed)
    report.update(report_add)

    # print execution time
    t2 = time.time()
    t_read = timing(t1, t2)
    print("[INFO]  Finished scanning stats file. %s"  % t_read)

    # create report and fix if needed
    summary = summarize(res_fixed)
    gr, dr = create_report(report, summary)

    # create a list of chhunk data frames
    data_fixed = [ch[1] for ch in res_fixed]

    # write result file
    t1 = time.time()
    try:
        # save fixed file to the output folder
        fout = os.path.join(outdir, os.path.basename(filename))
        fout = fout + '.CHUNK.gz' if not deep else fout
        # write file if some fixes were made
        if fix and summary['fixed'].sum() > 0:
            write_stats(fout, data_fixed, num_cpus=0)    
            fout_path = os.path.abspath(fout)
        else:
            print('[INFO]  No errors or fixes found in stats file.')
            fout_path = None
    except Exception as e:
        print("Error occurred while writing output file :: %s. Skip fixing." % e)
    t2 = time.time()
    t_write = timing(t1, t2)
    print("[INFO]  Finished writing stats file: %s"  % t_write)
    
    # try sorting the data if unsorted contigs found
    t1 = time.time()
    if summary.loc['unsorted', 'fail'] > 0:
        print("[INFO]  Unsorted contigs found in the data - fixing the data.")
        gr = sort_data(fout_path, gr)
    
    t2 = time.time()
    t_sort = timing(t1, t2)
    print("[INFO]  Finished sorting stats file: %s"  % t_write)

    return gr, dr, t_read, t_write, t_sort, fout_path


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

    return report_final


def multi_run_wrapper(args):
   r, d = chunk_scan(*args)
   return r, d


def multi_run_wrapper_fix(args):
   r, d, f = chunk_fix(*args)
   return r, d, f


def chunk_scan(text, columns, delim):

    report = []

    # special chars
    if '#chrom' in text[0:15]:
        first_line, text = text.split('\n', 1)
    else: 
        first_line = ''
    entries, ids = scan_special_chars(text)
    for sp in entries:
        text = text.replace(sp, '')
    
    # append the line back if it was a header
    if first_line != '':
        text = first_line + '\n' + text
        
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
        entries, ids = get_invalid(dat, 'pval', lambda x: not ((x >= 0 and x <= 1) or np.isnan(x)))
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

    # check if pval column contain correct values
    try:
        entries, ids = get_invalid(dat, 'af_alt', lambda x: not (x >= 0 and x <= 1))
    except TypeError as e:
        entries, ids = get_invalid(dat, 'af_alt', lambda x: not bool(re.match("[0-9+.-]+", x)))
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt'), 'af_alt', entries, ids)
    else:
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt'), 'af_alt', entries, ids)
    report.append(s)

    # check if pval column contain correct values
    try:
        entries, ids = get_invalid(dat, 'af_alt_cases', lambda x: not (x >= 0 and x <= 1))
    except TypeError as e:
        entries, ids = get_invalid(dat, 'af_alt_cases', lambda x: not bool(re.match("[0-9+.-]+", x)))
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt_cases'), 'af_alt_cases', entries, ids)
    else:
        s = make_summary("INVALID_FORMATTING", list(dat.columns).index('af_alt_cases'), 'af_alt_cases', entries, ids)
    report.append(s)

    # check if pval column contain correct values
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

    # iterate through all matches
    line_numb = []
    illegal_chars = []

    # iterate through all matches
    count = 0
    for match in finder:
        char_pos = match.start(0) + 1
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
    
    # append the  numeric chr column and sort by it and pos
    col_id = df.shape[1]
    df_sorted = df.copy()
    df_sorted[col_id] = list(chr_numbers.loc[df_sorted.iloc[:, 0]]['chr_numb'])
    df_sorted = df_sorted.sort_values([col_id, df_sorted.columns[1]], 
        ascending = [True, True])

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

    return summary


def get_invalid(df, colname, func):
    ids = list(df.loc[df.loc[:, colname].apply(func)].index)
    entries = list(df.loc[ids, colname])
    return entries, ids


def make_summary(issue, col_id, colname, entries, ids):
    return {'col_id': col_id, 
            'colname': colname,
            'issue': issue,
            'status': 'FAIL', 
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

    chr = df.iloc[ids, 0].apply(lambda x: [re.sub(r"[^0-9]+", "", x) \
        if not c in x else x.replace(x, c) for c in ['X', 'Y', 'M', 'MT']])
    chr = list(chr.apply(lambda x: [e for e in x if e != ''][0]))
    df.iloc[ids, 0] = chr

    return df
    

# fix chunks: mising, chr prefix, sorting, column order
def chunk_fix(report, df, fix):

    fixes = {}
    check_sort = True
    has_nas = False

    chroms = []
    missing =  []
    special = []
    for item in report:
        if item['issue'] == 'SPECIAL_CHARACTERS':
            if fix:
                special.append('fixed')
            else:
                special.append('fail')
        else:
            special.append('pass')

        # fix chromosome characters
        if item['colname'] == "#chrom" and item['issue'] == 'INVALID_FORMATTING' and len(item['row_id']) > 0:
            if fix:
                df = fix_chrom_col(df, item['row_id'])
                ids_fixed = get_invalid(df.iloc[item['row_id'], :], '#chrom', 
                    lambda x: x not in [str(item) for item in list(range(1,25))] + ['X', 'Y', 'M', 'MT'])
                if len(ids_fixed[0]) > 0:
                    check_sort = False
                    chroms.append('fail') # failed_fix fixed pass fail skip
                else:
                    chroms.append('fixed')
            else:
                chroms.append('fail')
        else:
            chroms.append('pass')

        # missing values characters
        if item['colname'] != "#chrom" and item['issue'] == 'INVALID_FORMATTING' and len(item['row_id']) > 0:
            entries = item['invalid_entry']
            nas = [np.isnan(e) if not isinstance(e, str) else e == '' for e in entries]
            if sum(nas) > 0:
                has_nas = True
                if fix:
                    missing.append('fixed')
                else:
                    missing.append('fail')
            else:
                missing.append('pass')
        else:
            missing.append('pass')

    # missing values
    if has_nas and fix:
        df = df.fillna(value = 0.5)
    
    # check if data is sorted
    if check_sort:
        s = scan_sorted(df)
        report.append(s)
        if len(s) > 0:
            fixes['unsorted'] = ['fail']
        else:
            fixes['unsorted'] = ['pass']
    else:
        s = []
        fixes['unsorted'] = ['skip']

    # reorder the cols
    cols = ["#chrom", "pos", "ref", "alt", "pval", "mlogp", "beta", "sebeta", "af_alt", "af_alt_cases", "af_alt_controls"]
    if len(set(cols).difference(set(df.columns))) == 0:
        if fix:
            df = df[cols]
            fixes['cols_order'] = ['fixed']
        else:
            fixes['cols_order'] = ['fail']
    else:
        fixes['cols_order'] = ['pass']

    fixes['chrom_column'] = list(set(chroms))
    fixes['missing_values'] = list(set(missing))
    fixes['special_chars'] = list(set(special))

    return report, df, fixes
    

def write_stats(fileout, df_lst, num_cpus=0):

    print("[INFO]  Start writing fixed stats file to the file. This might take a few mins.")

    # convert to strings
    h = '\t'.join(str(elem) for elem in list(df_lst[0].columns))
    with mgzip.open(fileout, 'wb', thread=num_cpus, blocksize=2 * 10 ** 8) as fw:

        # go through the chuck and combine rows
        for i, df in enumerate(df_lst):

            # combine strings
            res = df.apply(lambda y: '\t'.join(str(elem) for elem in y), axis = 1)
            
            # concat to a single string
            res_list = res.tolist()

            # append header if processing first chunk
            if i == 0:
                res_list.insert(0, h)
            
            res_str = '\n'.join(res_list)

            # append the last EOL symbol at the end of the text
            if i < len(df_lst):
                res_str = res_str + '\n'
            out = fw.write(res_str.encode())


def summarize(res):
    summary = {}
    for status in ['pass','fail', 'fixed', 'skip']:
        status_scan = []
        for r in res:
            keys = r[2].keys()
            status_scan.append([status in r[2][k] for k in r[2].keys()])
        df_scan = pd.DataFrame(status_scan)
        df_scan.columns = keys
        s = df_scan.apply(lambda x: sum(x), axis = 0)
        summary.update({status: s})
    return pd.DataFrame.from_dict(summary)


def format_entries(report, key):
    z = zip(report[key]['row_id'], report[key]['invalid_entry'])
    lines = [("\tLine: %s" % e[0]).ljust(15) + "\tEntry: %s" % e[1] for e in z]
    warn = ' - print first 50' if len(lines) >= 50 else ''
    nm = "INVALID ENTRIES %s" % report[key]['colname']
    h = (' %s ' % nm).center(80, '=')
    rline = ["\n%s\n" % h, '\n'.join(lines[0:49])]
    return rline, warn
        

def create_report(report, summary):

    # create reports
    general_report  = [] 
    detailed_report = []
    report_issues_all = list(report.keys())

    # special characters
    key = 'SPECIAL_CHARACTERS'
    if key in list(report.keys()):
        report_issues_all.remove(key)
        drep, warn = format_entries(report, key)
        detailed_report = detailed_report + drep
        status = 'FIXED' if summary['fixed']['special_chars'] > 0 else 'FAIL'
        message = "%sSpecial characters found in the stats file%s. Check details below." \
            % (('[%s]' % status).ljust(8, ' '), warn)
    else:
        message = "[PASS]  No special characters were found in the stats file."
    general_report.append(message)

    # chrom column
    if summary[['fail', 'fixed']].sum(axis=1)['chrom_column'] > 0:
        key = 'INVALID_FORMATTING_#chrom'
        report_issues_all.remove(key)
        drep, warn = format_entries(report, key)
        detailed_report = detailed_report + drep
        status = 'FIXED' if summary['fixed']['chrom_column'] > 0 else 'FAIL'
        message = "%sChromosome colum contains invalid entries%s. "\
            "Check details below." % (('[%s]' % status).ljust(8, ' '), warn)
    else:
        message = "[PASS]  No invalid entries in the column #CHROM "\
                  "were found in the stats file."
    general_report.append(message)

    # invalid columns
    cols = ["pos", "ref", "pval", "mlogp", "beta", "sebeta", 
            "af_alt", "af_alt_cases", "af_alt_controls"]
    for key in report.keys():
        if key != 'INVALID_FORMATTING_#chrom' and 'INVALID_FORMATTING' in key:
            report_issues_all.remove(key)
            col = report[key]['colname']
            cols.remove(col)
            entries = report[key]['invalid_entry']
            drep, warn = format_entries(report, key)
            detailed_report = detailed_report + drep
            nas = [np.isnan(e) for e in entries if not isinstance(e, str)]
            if sum(nas) == len(entries):
                continue
            else:
                entries_no_nas = [entries[i] for i, x in enumerate(nas) if not x]
                message = "[FAIL]  Invalid entries found in the column %s of "\
                    " stats file were found (total of %s)." \
                    % (col.upper(), len(entries_no_nas))
                general_report.append(message)
    
    # columns which passed the scan for invalid entry
    for col in cols:
        message = "[PASS]  No invalid entries in column %s were found in the stats file." % col.upper()
        general_report.append(message)

    # missing values
    if summary['fixed']['missing_values'] > 0 or summary['fail']['missing_values'] > 0:
        if summary['fixed']['missing_values'] > 0:
            message = "[FIXED] Missing values found in the data - substitute with 0.5. Check details below."
        elif summary['fail']['missing_values'] > 0:
            message = "[FAIL]  Missing values found in the data. Check details below."
    else:
        message = "[PASS]  No missing values found in the data."
    general_report.append(message)

    # add other errors
    for key in report_issues_all:
        if key != 'UNSORTED':
            message = "[FAIL]  File %s." % key.replace('_', ' ').lower()
            general_report.append(message)

    grp = '\n'.join(general_report) + '\n'
    drp = '\n'.join(detailed_report) + '\n'

    return grp, drp


def sort_data(filename, report):
    
    # create output temporaty files
    f1 =  filename + '_part1'
    f2 = filename + '_part2'
    f3 = filename + '_sorted'

    # bash cmds    
    cmd1 = "zgrep '^[0-9#]' %s | sort -k1,1g -k2,2n > %s" % (filename, f1)
    cmd2 = "zgrep -v '^[0-9#]' %s | sort -k1,1 -k2,2n > %s" % (filename, f2)
    cmd3 = "cat %s %s > %s" % (f1, f2, f3)
    cmd4 =  "gzip %s" % f3
    cmd5 =  "mv %s.gz %s" % (f3, filename)
    
    # sort data,gzip and rename
    try:
        subprocess.call(cmd1, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        subprocess.call(cmd2, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        subprocess.call(cmd3, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        subprocess.call(cmd4, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        subprocess.call(cmd5, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print("[ERROR] Sort data error : %s" % e)
        message  = "[FAIL]  Unsorted contigs found in the data - tried to fix by "\
            "running command `sort -g -k 1,1 -g -k 2,2` but failed with error %s\n" % e
    else:
        message  = "[FIXED] Sorted the data by chromosome and position.\n"  
    
    subprocess.call("rm %s %s" % (f1, f2), stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
    
    return report + message