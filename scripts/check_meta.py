import re
import os
import json

def check_meta(meta_filename, stats_filename, outdir, fix):

    field_types = [ list, str, str, str, int, str, int, int, str, str, list, list, str ]
    required_metadata_fileds=["admin_email", "analysis_type", "category", 
                              "description", "freeze", "name", "num_cases",
                              "num_controls", "output_bucket", "pheno_coding",
                              "submitter", "submitter_email", "title"]
    
    # create report dictionary
    report_dict = {}

    # 1. check if file contains non-ASCII characters
    message = ''
    with open(meta_filename, 'rb') as f:
        dat = f.read()
        try: 
            dat_ascii = dat.decode('ascii')
        except UnicodeDecodeError as e:
            # substitute the double quotes if that is a problem
            dat_ascii = re.sub('”', '"', dat.decode('utf-8').strip())

            # remove all other characters if present
            dat_ascii = dat_ascii.encode("ascii", "ignore")
            dat_ascii = dat_ascii.decode()

            # write to a report
            message = """[FAIL]  Metadata file contains special characters.\
                \n        HINT: Check with the command (MAC OSX: `cat -e METADATA`, \
                \n        Linux: `cat -A METADATA`) to reveal those."""

        else:
            message = "[PASS]  Metadata file doesn't contain special characters."
    report_dict.update({'special_chars': message})
    
    # proceed with the scan
    try:
        mdata = json.loads(dat_ascii)
    except Exception as e:
        message="""[INFO]  Tried to fix the errors in the user's metadata \
            \n        file but didn't succeed. Check it manually."""
        report_dict.update({'open_file': message})
        report = '\n'.join([item[1] for item in report_dict.items()])
        return report

    # 2. missing_fields     
    missing_fields = [f for f in required_metadata_fileds if f not in list(mdata.keys())]
    if len(missing_fields) > 0:
        message="""[FAIL]  Metadata file doesn't contain all required fields.\
        \n        Missing fields: %s """ % missing_fields
    else:
        message="[PASS]  Metadata file contains all required fields."
    report_dict.update({'misisng_fields': message})

    # 3. wrong type
    wrong_type = []
    for i in list(range(len(mdata.keys()))):
        key = list(mdata.keys())[i]
        if not isinstance(mdata[key], field_types[i]):
            wrong_type.append((field_types[i], type(mdata[key])))

    #  check if some fields were in the wrong format
    if len(wrong_type) > 0:
        message="""[FAIL]  Metadata fields have incorrect format. Malformatted fields found:\
            \n        %s""" % ';'.join(["Provided: %s, Required: %s""" % tupl for tupl in wrong_type])
    else:
        message="[PASS]  Metadata fields have correct format."
    report_dict.update({'fields_type': message})

    # 4. check filename errors
    if mdata['name'] != os.path.basename(stats_filename).split('.gz')[0]:
        message="""[FAIL]  Field "name" in metadata doesn't match stats filename. \
        \n        Provided name in metadata: %s \
        \n        Stats filename: %s. \
        \n        HINT: If stats file is C3_COLORECTAL.gz then name field \
        \n        should be C3_COLORECTAL.""" \
        % (mdata['name'], os.path.basename(stats_filename))
    else:
        message="""[PASS]  Metadata field "name" matches with summary stats file name."""
    report_dict.update({'filename': message})

    # save the mdata
    if fix:
        report_dict['special_chars'] = re.sub('FAIL\]\ ', 'FIXED]', report_dict['special_chars'])
        fout = os.path.join(outdir, os.path.basename(meta_filename))
        fout = os.path.abspath(fout)
        with open(fout, 'w') as f:
            f.write(json.dumps(mdata, indent=2))
    else:
        fout = None

    report = '\n'.join([item[1] for item in report_dict.items()])

    return report, fout
