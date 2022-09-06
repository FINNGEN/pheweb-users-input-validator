# PHEWEB INPUTS VALIDATOR


Python tool for validating users input files for uploading to the pheweb browser. 

User needs to provide:
+ Metadata file in JSON format 
+ Statistics file

There are two modes for scanning your stats file: deep and shallow. With the deep mode the whole file is scanned while with the shallow mode ~80k lines are subsampled from the stats file and subjected to the scan. 

**Recommendation**: first run validator in a shallow mode to check whether your metadata is correctly formatted and check if some basic requirements for the stats file are met. Once that is checked - proceed to the deep check of the files.

Expected runtimes:
+ Shallow scan: less than a minute
+ Deep scan: 700MB file is scanned, fixed and written in ~10-20 mins depending on the available resources.


**Machine resource requirements**: validator uses multiprocessing python package and perfomes fastest when it is executed on the machine with multiple CPUs and good amount of memory, e.g. 4CPUs and memory 16GB.


Guidelines for formatting your metadata and stats file can be found in [FinnGen Analyst Handbook](https://finngen.gitbook.io/finngen-analyst-handbook/working-in-the-sandbox/which-tools-are-available/untitled/how-to-set-up-a-pheweb-browser-for-summary-statistics). 

## Requirements

+ Python 3
+ Python packages: `pandas`, `pysam`, `xopen`, `mgzip`

To install python packages, run:
```
pip install -r requirements.txt
```


## Usage
```
USAGE: python3 validator.py -m <metadata> -s <stats> -o <outdir> -d <deep_check> -f <fix>

Input arguments:
-m   <metadata>    : path to metadata json file
-s   <stats>       : path to stats file
-o   <outdir>      : path to output directory
-d   <deep_check>  : enable deep scan. Possible values {0, 1, true, false, T, F, True, False}, Default: True.
-d   <fix>         : enable fix of the issues. Possible values {0, 1, true, false, T, F, True, False}, Default: True. 
```

## Output

The results of the scan are saved under <DIR_OUT>/scan<TIMESTAMP>/. Output files:
- report1.log
- report2.log
- 

## Example

Report example of the **deep scan** using example data specified in the Handbook:
```
bash validator.sh -m metadata.json -s C3_COLORECTAL.gz -o ${DIROUT}/ -d true -f true

================= SCAN STARTED AT: 2022-09-06 06:47:44.667495 ==================

[INFO]  Check metadata file.
[INFO]  Metadata file check completed.
[INFO]  Check stats file.
[INFO]  Start reading stats file in chunks.
[INFO]  Start scanning stats file in chunks. This might take some time.
[INFO]  Finished scanning stats file. Execution time: 2.44 mins
[INFO]  Start writing fixed stats file to the file. This might take a few mins.
[INFO]  Finished writing stats file: Execution time: 5.23 mins
[INFO]  Stats file check completed.

================================================================================

FILENAME             PASSED    FAILED    FIXED     REMAINING MD5SUM
metadata.json        4/4       0         0         0         4b52275beb25f84fd010e570ee7de44d
C3_COLORECTAL.gz     13/13     0         0         0         97d1ec64a91cbc4b5c277f2a2034fa5d
Total                17/17     0         0         0         -

================================================================================


OUTPUT FILES:
	${DIROUT}/scan05092022T1545/report.log
	${DIROUT}/scan05092022T1545/metadata.json
	${DIROUT}/scan05092022T1545/C3_COLORECTAL.gz

SCAN ENDED AT: 2022-09-06 06:55:58.433856

Read stats file execution time: 2.44 mins
Write stats file execution time: 5.23 mins
Total execution time: 8.23 mins

```

Report example of the **shallow scan** using example data specified in the Handbook:
```
================= SCAN STARTED AT: 2022-09-06 11:07:23.814803 ==================

[INFO]  Check metadata file.
[INFO]  Metadata file check completed.
[INFO]  Check stats file.
[INFO]  Start reading stats file in chunks.
[INFO]  Start scanning stats file in chunks. This might take some time.
[INFO]  Finished scanning stats file. 1.64 sec
[INFO]  No errors or fixes found - skip writing to a file.
[INFO]  Stats file check completed.

================================================================================

FILENAME             PASSED    FAILED    FIXED     REMAINING MD5SUM
metadata.json        4/4       0         0         0         4b52275beb25f84fd010e570ee7de44d
C3_COLORECTAL.gz     13/13     0         0         0         97d1ec64a91cbc4b5c277f2a2034fa5d
Total                17/17     0         0         0         -         

================================================================================


OUTPUT FILES:
	${DIROUT}/report.log

SCAN ENDED AT: 2022-09-06 11:07:27.121926

Read stats file 1.64 sec
Write stats file 0.0 sec
Total 3.31 sec


```
