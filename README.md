# PHEWEB USERS INPUTS VALIDATOR

## Overview
Python tool for validating users input files for uploading to the pheweb browser. User needs to provide:
+ Metadata file in JSON format 
+ Statistics file

There are **two modes** for scanning your stats file: deep and shallow (specified by setting the parameter "--deep true/false", see user manual below). With the deep mode the whole file is scanned while with the shallow mode ~80k lines are subsampled from the stats file and subjected to the scan. 

Also, a user can enable fixing mode (by setting the parameter "--fix true", see user manual below) in order to fix issues found by the validator. See section "Scans and fixes performed by the validator" for more details on the checks performed by the validator and what issues can be fixed automatically.

Prepare your files according to the instruction given in the [FinnGen Analyst Handbook](https://finngen.gitbook.io/finngen-analyst-handbook/working-in-the-sandbox/which-tools-are-available/untitled/how-to-set-up-a-pheweb-browser-for-summary-statistics). 

**Recommendation**: first run validator in a shallow mode to check whether your metadata is correctly formatted and check if some basic requirements for the stats file are met. Once that is checked - proceed to the deep check of the files: you can either enable fixing straight away (by adding --fix true) or without it. Note that running **fix** mode might take a long time (more than 20 minutes) if your file is large and it requires sorting. 


## Resources and expected runtimes
Validator uses multiprocessing python package and perfomes fastest when it is executed on the machine with multiple CPUs and good amount of memory, e.g. 4CPUs and memory 16GB. The recommened version of the tool is **Version 1: validator.py**. However, there is a second version of the tool, **Version 2: validator_req25GBmem.py**, which is faster than the first one but it requires more memory.

Expected runtimes:
+ Shallow scan: less than a minute
+ Deep scan: 700MB file is scanned, fixed and written in ~10-20 mins depending on the available resources.


## Scans and fixes performed by the validator

The following scans are executed by the validator:

METADATA:
1. Check for special characters in metadata.
2. Check that metadata contains all required fields.
3. Check that metadata fields have correct format.
4. Check that metadata field "name" matches stats filename. 

STATS:
1. Check that file is compressed.
2. Check that file is tab delimited.
3. Check that columns order is correct.
4. Ceck that file doesn't contain special characters.
5. Check that chromosome column ("#chrom") has correct formatting.
6. Check that columns 7-11 (beta, sebeta, af_alt, af_alt_cases, af_alt_controls) don't contain missing values.
7. Check that columns 2-11 have correct formatting.
8. Check that stats file doesn't contain unsorted positions.


The following fixes can be done by the validator **when possible**:
- Remove special characters from the metadata file
- Remove special characters from the stats file
- Fix missing values in the colums 7-11
- Remove chromosome prefix, e.g. "chrN" change to "N"
- Sort stats file if unsorted positions are found
- Fix column order

What is **not** fixed in the stats file:
- Incorrect values in the columns, for instance negative p-values
- Naming of the stats file in te metadata json file


## Requirements

+ Python 3
+ Python packages: `pandas`, `pysam`, `xopen`, `mgzip`

To install python packages, run:
```
pip install -r requirements.txt
```

## Usage
```
USAGE: python3 validator.py -m <metadata> -s <stats> -o <outdir> -d <deep_check> -f <fix_issues>

Input arguments:
-m   <metadata>    : path to metadata json file
-s   <stats>       : path to stats file
-o   <outdir>      : path to output directory
-d   <deep_check>  : enable deep scan. Possible values {0, 1, true, false, T, F, True, False}, Default: True.
-d   <fix_issues>  : enable fix of the issues. Possible values {0, 1, true, false, T, F, True, False}, Default: True. 
```

## Outputs of the validator.py

Output files:
1. Full report on the results of validator scanning will be saved in file <DIR_OUT>/scan<SCAN_TIMESTAMP>/report.log.
2. If fix mode is activated and some issues were fixed by the validator, a new stats file is written in <DIR_OUT>/scan<SCAN_TIMESTAMP>/<STATS_FILENAME>.
3. Lines from stats file in which validator was able to detect issues are saved in  <DIR_OUT>/scan<SCAN_TIMESTAMP>/<STATS_FILENAME>_lines_with_errors.

## Example

Report example of the **deep scan** using example data specified in the Handbook:
```
bash validator.sh -m metadata.json -s C3_COLORECTAL.gz -o ${DIROUT}/ -d true -f true


================= SCAN STARTED AT: 2022-12-14 20:54:59.218729 =================

[PASS]  Metadata file doesn't contain special characters.
[PASS]  Metadata file contains all required fields.
[PASS]  Metadata fields have correct format.
[PASS]  Metadata field "name" matches with summary stats file name.
[FIXED] Columns order fixed.
[PASS]  Chromosome column is formatted correctly.
[PASS]  File is compressed.
[PASS]  File is tab delimited.
[PASS]  No invalid entries in columns of the stats file were found below.
[PASS]  No missing values found in columns 7-11 of stats file.
[PASS]  No special characters were found in the stats file.
[PASS]  No unsorted positions were found in the data.

================================================================================

FILENAME                    PASSED    FAILED    FIXED     MD5SUM
metadata.json               4         0         0         946f7747b57e8cf31578a3a46cf66abf
C3_COLORECTAL.gz            7         0         1         97d1ec64a91cbc4b5c277f2a2034fa5d
-----------------------------------
Total successful scans:     12 / 12

================================================================================

OUTPUT FILES:
	${DIROUT}/scan05092022T1545/report.log
	${DIROUT}/scan05092022T1545/C3_COLORECTAL.gz

SCAN ENDED AT: 2022-12-14 21:03:00.059127

Read stats file execution time: 2.96 mins
Write stats file execution time: 4.88 mins
Sort stats file 0.0 sec
Total execution time: 8.01 mins

```