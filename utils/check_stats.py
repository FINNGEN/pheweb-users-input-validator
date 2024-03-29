# -*- coding: utf-8 -*-
 
import io
import os
import re
import time
import xopen
import mgzip
import pysam
import warnings
import subprocess
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
    report_add, rows_map = combine_chunks(res_fixed)
    report.update(report_add)

    # extract rows with issues
    lines_errors_exclusing_sp = extract_rows(res, rows_map)
    lines_errors_sp = extract_rows_sp(res, rows_map)

    if len(lines_errors_sp) > 0 or len(lines_errors_exclusing_sp) > 0:
        lines_errors_all = lines_errors_sp + '\n' + lines_errors_exclusing_sp
        fname = os.path.basename(filename)
        fout_lines_path = os.path.join(outdir, "{0}_lines_with_errors".format(fname.split('.gz')[0]))
        with open(fout_lines_path, 'w') as f:
            f.write(lines_errors_all)
        fout_lines_path = os.path.abspath(fout_lines_path)
    else:
        fout_lines_path = None

    # print execution time
    t2 = time.time()
    t_read = timing(t1, t2)
    print("[INFO]  Finished scanning stats file. %s"  % t_read)

    # create report and fix if needed
    summary = summarize(res_fixed)
    gr, dr = create_report(report, summary)

    # create a list of chunk data frames
    data_fixed = [ch[1] for ch in res_fixed]

    # save file if some fixes were made
    t1 = time.time()
    try:
        fout = os.path.join(outdir, os.path.basename(filename))
        fout = fout + '.CHUNK.gz' if not deep else fout
        if fix and summary['fixed'].sum() > 0:
            print("[WARN]  Some fixes were made in the stats file. Check details in the final report.")
            write_stats(fout, data_fixed, num_cpus=0)    
            fout_path = os.path.abspath(fout)
        else:
            print('[INFO]  No errors or fixes found in stats file.')
            fout_path = None
    except Exception as e:
        print("[ERROR]  Error occurred while writing output file :: %s. Skip fixing." % e)
    
    t2 = time.time()
    t_write = timing(t1, t2)
    print("[INFO]  Finished writing stats file: %s"  % t_write)
    
    # try sorting the data if unsorted contigs found
    t1 = time.time()
    if summary.loc['unsorted', 'fail'] > 0:
        print("[WARN]  Unsorted positions found in the data - fixing the data.")
        if fix:
            gr = sort_data(fout_path, gr, report)
        else:
            first_unsorted = report['UNSORTED']['row_id'][0]
            message  = "[FAIL]  Unsorted positions found in the data, first unsorted line: %s.\n" % first_unsorted
            gr.append(message)
    else:
        gr.append("[PASS]  No unsorted positions were found in the data.")

    gr.sort()
    gr_combined = '\n'.join(gr) + '\n'

    t2 = time.time()
    t_sort = timing(t1, t2)
    print("[INFO]  Finished sorting stats file: %s"  % t_sort)

    return gr_combined, dr, t_read, t_write, t_sort, fout_path, fout_lines_path

def extract_rows_sp(struct, rmap):
    if rmap is None:
        return ''
    lines_issues_sp = []
    for i in list(range(len(struct))):        
        x = struct[i]
        rmap_chunk = rmap[i]
        rmap_chunk_df = pd.DataFrame(rmap_chunk)
        # do not report the same rows several times
        rmap_chunk_df = rmap_chunk_df[~rmap_chunk_df.duplicated()]
        rmap_chunk_df.index = rmap_chunk_df['row_id_chunk']
        if len(x[2]) > 0:
            rowids = list(set(x[2].keys()))
            lines = [x[2][ri] for ri in rowids]
            prefix = list(rmap_chunk_df.loc[rowids, 'row_id_final'].apply(lambda x: ("LINE %s:" % x).ljust(15, ' ')))
            lines_h = [ "%s%s" % z for z in zip(prefix, lines)]
            lines_issues_sp += lines_h

    combined = '\n'.join(lines_issues_sp)

    return combined

def extract_rows(struct, rmap):
    if rmap is None:
        return ''
    df_extracted_rows =pd.DataFrame()
    index = []
    for i in list(range(len(struct))):
        r = struct[i][0]
        df = struct[i][1]
        rmap_chunk = rmap[i]
        rmap_chunk_df = pd.DataFrame(rmap_chunk)
        # do not report the same rows several times
        rmap_chunk_df = rmap_chunk_df[~rmap_chunk_df.duplicated()]
        rmap_chunk_df.index = rmap_chunk_df['row_id_chunk']
        for x in r:
            if len(x['row_id']) > 0 and x['issue'] != 'SPECIAL_CHARACTERS' and x['issue'] != 'UNSORTED':
                df_extracted_rows = df_extracted_rows.append(df.iloc[x['row_id']])
                index = index + list(rmap_chunk_df.loc[x['row_id'], 'row_id_final'])
   
    combined = list(df_extracted_rows.apply(lambda y: '\t'.join(str(elem) for elem in y), axis = 1))
    prefix = [("LINE %s:" % i).ljust(15, ' ') for i in index]
    lines_final = ["%s%s" % (prefix[j], combined[j]) for j in list(range(len(combined)))]
    res = '\n'.join(lines_final)
   
    return res


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
    rows_map = []
    j = 0
    tot_reports = {i: multithreads_res[i][0] for i in list(range(len(multithreads_res)))}
    for i in tot_reports.items():
        offset_rowid = offset[i[0]]
        pandas_rowid = pandas_id[i[0]]
        for k in list(range(len(i[1]))):
            if len(i[1][k]) > 0 and len(i[1][k]['row_id']) > 0:
                item = i[1][k].copy()
                row_id_chunk = item['row_id']
                item['pandas_row_id'] = [row + pandas_rowid for row in item['row_id']]
                item['row_id'] = [row + offset_rowid for row in item['row_id']]
                if item['colname'] is not None:
                    issues.append(item['issue'] + '_' + item['colname'])
                else:
                    issues.append(item['issue'])
                d = {
                    'row_id_chunk': row_id_chunk,
                    'row_id_final': item['row_id'],
                    'chunk_id': i[0],
                }
                rows_map.append(d)
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

    if len(rows_map) > 0:
        rows_map_final = {i: {'row_id_chunk': [],  'row_id_final': []} for i in list(range(len(multithreads_res)))}
        for r in rows_map:
            key = r['chunk_id']
            rows_map_final[key]['row_id_chunk'] = rows_map_final[key]['row_id_chunk'] + r['row_id_chunk']
            rows_map_final[key]['row_id_final'] = rows_map_final[key]['row_id_final'] + r['row_id_final']
    else:
        rows_map_final = None
    
    return report_final, rows_map_final

def multi_run_wrapper(args):
   r, d, sp = chunk_scan(*args)
   return r, d, sp

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
    entries, ids, lines_sp = scan_special_chars(text)

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
    return report, dat, lines_sp

def scan_special_chars(t):
    # split lines
    lines = t.split('\n')
    sizes = [len(l) + 1 for l in lines]
    sizes_cumsum = [0] + list(np.cumsum(sizes))
    st = sizes_cumsum[0:(len(sizes_cumsum)-1)]
    en = sizes_cumsum[1:len(sizes_cumsum)]

    # scan for the not allowed chars
    accepted_chars = re.compile("[^A-Za-z0-9-. +,\t\n]")
    finder = re.finditer(accepted_chars, t) 

    # iterate through all matches
    line_numb = []
    illegal_chars = []

    # iterate through all matches
    count = 0
    lines_issues = {}
    for match in finder:
        ids = np.logical_and(match.start(0) > np.array(st), match.start(0) < np.array(en))
        ln = np.where(ids)[0][0]
        lines_issues.update({ln: lines[ln]})
        illegal_chars.append(match.group())
        line_numb.append(ln)
        if count > 50:
            print('[INFO]   Too many special characters - limiting to first 50.')
            break
        count += 1

    return illegal_chars, line_numb, lines_issues

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

    chr = df.iloc[ids, 0].apply(lambda x: [re.sub(r"[^0-9]+", "", x) \
        if not c in x else x.replace(x, c) for c in ['X', 'Y', 'M', 'MT']])
    chr = list(chr.apply(lambda x: [e for e in x if e != ''][0]))
    df.iloc[ids, 0] = chr

    return df
    
def flatten(l):
    new_list = []
    for sublist in l:
        if isinstance(sublist, list):
            for item in sublist:
                new_list.append(item)
        else:
            new_list.append(sublist)
    return new_list

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
            check_sort = False
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
        cols_fix_nas = ["beta", "sebeta", "af_alt", "af_alt_cases", "af_alt_controls"]
        df[cols_fix_nas] = df[cols_fix_nas].fillna(value = 0.5)
        # df = df.fillna(value = 0.5)
    
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

    # if not all required colums are in the table or order is wrong
    if list(df.columns) != cols:
        
        # if all required columns are in the table but in the wrong order or there are some extra columns
        if len(set(cols).difference(set(df.columns))) == 0:

            # if can be re-ordered and selected, do it 
            if fix:
                df = df[cols]
                fixes['cols_order'] = ['fixed']
            
            # otherwise, fail
            else:
                fixes['cols_order'] = ['fail']
        
        # NOT all required columns are in the table
        else:
            fixes['cols_order'] = ['fail']

    # if all required colums are in the table AND order is right
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
            _ = fw.write(res_str.encode())

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

def create_report(report, summary):

    # create reports
    general_report  = [] 
    detailed_report = []
    report_issues_all = list(report.keys())

    # special characters
    key = 'SPECIAL_CHARACTERS'
    if key in list(report.keys()):
        report_issues_all.remove(key)
        status = 'FIXED' if summary['fixed']['special_chars'] > 0 else 'FAIL'
        lines, warn = format_entries(report, key, status.lower())
        header = ['\n', (' SPECIAL CHARACTERS ').center(90, '=')]
        detailed_report = detailed_report + header + lines[0:49] 
        message = "%sSpecial characters found in the stats file%s.\n        Check details under section SPECIAL CHARACTERS below." % (('[%s]' % status).ljust(8, ' '), warn)
    else:
        message = "[PASS]  No special characters were found in the stats file."
    general_report.append(message)

    # fix chromosome characters
    invalid_entries_keys = []
    if summary[['fail', 'fixed']].sum(axis=1)['chrom_column'] > 0:
        
        # remove key from the remaining issues to report
        key = 'INVALID_FORMATTING_#chrom'
        invalid_entries_keys.append(key)
        report_issues_all.remove(key)

        # add to a detailed report
        if summary['fixed']['chrom_column'] > 0:
            message = "[FIXED] Invalid formatting found in the chromosome column." + \
                      "           Check details under INVALID ENTRIES section below."
        else:
            message = "[FAIL]  Invalid formatting found in the chromosome column. Validator failed to \n" + \
                      "        fix the issue. Check details under INVALID ENTRIES section below."
    else:
        message = "[PASS]  Chromosome column is formatted correctly."
    general_report.append(message)

    # invalid entries in columns, exclude missing values
    missing_vals_keys = []
    reports_invalid = {}
    reports_missing = {}
    for key in report.keys():
        if key != 'INVALID_FORMATTING_#chrom' and 'INVALID_FORMATTING' in key:
            entries = report[key]['invalid_entry']            
            report_nas = report[key].copy()
            report_invalid = report[key].copy()

            # ids of nas and invalid
            nas_ids = [i for i,k in enumerate(entries) if np.isnan(k)]
            invalid_ids = [i for i,k in enumerate(entries) if not np.isnan(k)]

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
        message = "[FAIL]  Invalid entries found in some of the columns of the stats file.\n" + \
                  "        Check details under INVALID ENTRIES section below."
        tuples = [format_entries(reports_invalid, k, "fail") for k in invalid_entries_keys]
        lines = [t[0] for t in tuples]
        header = ['\n', (' INVALID ENTRIES ').center(90, '=')]
        detailed_report = detailed_report + header + lines
    else:
        message = "[PASS]  No invalid entries in columns of the stats file were found below."
    general_report.append(message)

    # missing values
    if len(missing_vals_keys) > 0:
        if summary['fixed']['missing_values'] > 0:
            status = "FIXED"
        elif summary['fail']['missing_values'] > 0:
            status = "FAIL"        
        message = "%sMissing values found in columns 7-11 of stats file - substitute with 0.5.\n        Check details under MISSING VALUES section below." % ('[%s]' % status).ljust(8, ' ')
        tuples = [format_entries(reports_missing, k, status.lower()) for k in missing_vals_keys]
        lines = [t[0] for t in tuples]
        header = ['\n', (' MISSING VALUES ').center(90, '=')]
        detailed_report = detailed_report + header + lines
    else:
        message = "[PASS]  No missing values found in columns 7-11 of stats file."
    general_report.append(message)

    # columns order
    if summary['fixed']['cols_order']  > 0:
        message = "[FIXED] Fixed columns order/number in the stats file."
    elif summary['fail']['cols_order'] > 0:
        message = "[FAIL]  Wrong columns order or columns number: check that your stats file contains exacly 11 required columns\n " + \
                  "        as described in the FinnGen Analyst Handbook."
    else:
        message = "[PASS]  Correct columns order/number in the stats file."
    general_report.append(message)

    # add other errors
    for key in report_issues_all:
        if key != 'UNSORTED':
            message = "[%s]  File %s." % (report[key]['status'], key.replace('_', ' ').lower())
            general_report.append(message)

    # sort general repport
    # general = '\n'.join(general_report) + '\n'
    detailed = '\n'.join(flatten(detailed_report)) + '\n'

    return general_report, detailed

def sort_data(filename, report, full_report):

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

    first_unsorted = full_report['UNSORTED']['row_id'][0]
    
    # sort data,gzip and rename
    try:
        subprocess.call(cmd1, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        subprocess.call(cmd2, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        subprocess.call(cmd3, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        subprocess.call(cmd4, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
        subprocess.call(cmd5, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print("[ERROR] Sort data error : %s" % e)
        message  = "[FAIL]  Unsorted positions found in the data - tried to fix but failed with error %s\n" % e
    else:
        message  = "[FIXED] Unsorted positions found in the data - sorted data by chromosome and position.\n" + \
                   "        First unsorted row id: %s" % str(first_unsorted)  
    
    subprocess.call("rm %s %s" % (f1, f2), stderr=subprocess.STDOUT, shell=True, executable='/bin/bash')
    report.append(message)

    return report
