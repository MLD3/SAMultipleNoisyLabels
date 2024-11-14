import numpy as np
import pandas as pd
import pickle as pkl

#locations of files

#where the raw mimic tables are stored
raw_data_dir = ''
med_file = raw_data_dir + 'PRESCRIPTIONS.csv'
chart_file = raw_data_dir + 'CHARTEVENTS.csv'
lab_file = raw_data_dir + 'LABEVENTS.csv'
inpm_file = raw_data_dir + 'INPUTEVENTS_MV.csv'
inpc_file = raw_data_dir + 'INPUTEVENTS_CV.csv'
out_file = raw_data_dir + 'OUTPUTEVENTS.csv'
adm_file = raw_data_dir + 'ADMISSIONS.csv'

save_to = '' #the mimic directory in directories.py


'''
sepsis criteria
'''

#######################################################
'''
based on https://github.com/BorgwardtLab/mgp-tcn/blob/master/src/query/abx-poe-list.sql

columns: ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE', 'ENDDATE',
       'DRUG_TYPE', 'DRUG', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
       'FORMULARY_DRUG_CD', 'GSN', 'NDC', 'PROD_STRENGTH', 'DOSE_VAL_RX',
       'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'ROUTE']
'''
def get_antibiotics(med_file):
    print('getting antibiotics')
    meds = pd.read_csv(med_file)
    print(meds.shape)

    #filter out cream, desensitization, ophth oint, and gel drugs
    meds['DRUG_NAME_GENERIC'] = meds['DRUG_NAME_GENERIC'].str.lower()
    meds['DRUG_NAME_GENERIC'] = meds['DRUG_NAME_GENERIC'].fillna('NAN')
    filter_out = ['cream', 'desensitization', 'ophth oint', 'gel']
    for i in range(len(filter_out)): 
        meds = meds[~meds['DRUG_NAME_GENERIC'].str.contains(filter_out[i])]
    print('filter some drugs', meds.shape)

    #filter out OU, OS, OD, AU, AS, AD, TP, ear, and eye routes
    meds['ROUTE'] = meds['ROUTE'].str.lower()
    meds['ROUTE'] = meds['ROUTE'].fillna('NAN')
    filter_out = ['ou', 'os', 'od', 'au', 'as', 'ad', 'tp', 'ear', 'eye']
    for i in range(len(filter_out)): 
        meds = meds[~meds['ROUTE'].str.contains(filter_out[i])]
    print('filter routes', meds.shape)

    #keep only MAIN and ADDITIVE drug types (alternatively, filter out BASE)
    meds['DRUG_TYPE'] = meds['DRUG_TYPE'].str.lower()
    meds['DRUG_TYPE'] = meds['DRUG_TYPE'].fillna('NAN')
    keep = 'main|additive' 
    meds = meds[meds['DRUG_TYPE'].str.contains(keep)]
    print('filter types', meds.shape)

    #keep only the antibiotics in the antibiotic list
    antibiotics = 'adoxa|ala-tet|alodox|amikacin|amoxicillin|clavulanate|ampicillin|augmentin|' \
                  + 'avelox|avidoxy|azactam|azithromycin|aztreonam|axetil|bactocill|bactrim|bethkis|' \
                  + 'biaxin|bicillin l-a|cayston|cefazolin|ceftazidime|cefaclor|cefadroxil|cefdinir|' \
                  + 'cefditoren|cefepime|cefotetan|cefotaxime|cefpodoxime|cefprozil|ceftibuten|ceftin|' \
                  + 'ceftin|cefuroxime|cephalexin|chloramphenicol|cipro|ciprofloxacin|claforan|' \
                  + 'cleocin|clindamycin|cubicin|dicloxacillin|doryx|duricef|dynacin|ery-tab|eryc|' \
                  + 'eryped|erythrocin|erythromycin|factive|flagyl|fortaz|furadantin|garamycin|' \
                  + 'gentamycin|kanamycin|keflex|ketek|levaquin|levofloxacin|lincocin|macrobid|' \
                  + 'macrodantin|maxipime|mefoxin|metronidazole|minocin|minocycline|monodox|monurol|' \
                  + 'morgidox|moxatag|moxifloxacin|myrac|nafcillin sodium|nicazel doxy 30|nitrofurantoin|' \
                  + 'noroxin|ocudox|ofloxacin|omnicef|oracea|oraxyl|oxacillin|pc pen vk|pce dispertab|' \
                  + 'panixine|pediazole|penicillin|periostat|pfizerpen|piperacillin|tazobactam|primsol|' \
                  + 'proquin|raniclor|rifadin|rifampin|rocephin|smz-temp|septra|septra ds|solodyn|' \
                  + 'spectracef|streptomycin sulfate|sulfadiaine|sulfamethoxazole|trimethoprim|' \
                  + 'sulfatrim|sulfisoxazole|suprax|tazicef|tetracycline|timentin|tobi|tobramycin|' \
                  + 'trimethoprim|unasym|vancocin|vancomycin|vantin|vibativ|vibra-tabs|vibramycin|' \
                  + 'zinacef|zithromax|zmax|zosyn|zyvox'
    meds = meds[meds['DRUG_NAME_GENERIC'].str.contains(antibiotics)]
    meds = meds.drop_duplicates()
    print('keep antibiotics', meds.shape)

    meds.to_csv(save_to + 'lab_preprocess_antibiotics.csv')

    return meds


'''
based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/severityscores/sirs.sql
'''
def get_sirs_parts(chart_file, lab_file):
    print('getting sirs parts')
    keep_cols = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']

    #chartevents - can't read entire table because big
    chunk = 2000000
    temp_f, temp_c, resprate, heartrate = [], [], [], []

    with pd.read_csv(chart_file, chunksize=chunk, dtype=str, na_values='NAN') as whole_chart:
        for chart in whole_chart:
            #body temperature (min, max), keep (70,12) or (10, 50)
            keep_f = '223761|678'
            temp_f.append(chart[chart['ITEMID'].str.contains(keep_f)][keep_cols])
            keep_c = '223762|676'
            temp_c.append(chart[chart['ITEMID'].str.contains(keep_c)][keep_cols])
            #heart rate (max) keep values in (0, 300)
            keep = '211|220045'
            heartrate.append(chart[chart['ITEMID'].str.contains(keep)][keep_cols])
            #respiratory rate (max), keep (0, 70)
            keep = '615|618|220210|224690'
            resprate.append(chart[chart['ITEMID'].str.contains(keep)][keep_cols])
    temp_f = pd.concat(temp_f, axis=0)
    temp_c = pd.concat(temp_c, axis=0)
    heartrate = pd.concat(heartrate, axis=0)
    resprate = pd.concat(resprate, axis=0)
    
    #labevents
    labs = pd.read_csv(lab_file, dtype=str, na_values='NAN')
    #paco2 (min) they use pco2
    keep = '50818'
    paco2 = labs[labs['ITEMID'].str.contains(keep)][keep_cols]
    #white blood cell count (min, max) exclude >1000
    keep = '51300|51301'
    wbc = labs[labs['ITEMID'].str.contains(keep)][keep_cols]
    #presence of >10% immature neutrophils (band forms), exclude <0 and >100
    keep = '51144'
    bands = labs[labs['ITEMID'].str.contains(keep)][keep_cols]

    #save it so don't have to run this again
    sirs_parts = {'temp_f': temp_f, 'temp_c': temp_c, 'heartrate': heartrate, 'resprate': resprate, 'paco2': paco2, \
                  'wbc': wbc, 'neutophil_band': bands}
    for key in list(sirs_parts.keys()):
        sirs_parts[key] = sirs_parts[key].drop_duplicates()
        print(key, sirs_parts[key].shape)
        sirs_parts[key].to_csv(save_to + 'lab_preprocess_sirs_' + key + '.csv')
    
    return 1


'''
based on https://github.com/BorgwardtLab/mgp-tcn/blob/master/src/query/sofa-hourly.sql
GCS, MAP, FiO2, ventillation status (chartevents)
creatinine, bilirubin, FiO2, PaO2, platelets (labevents)
dobutamine, epinephrine, norepinephrine, dopamine (inputevents_mv, inputevents_cv)
urine output (outputevents)

labevents columns: ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE',
       'VALUENUM', 'VALUEUOM', 'FLAG']
'''
def get_sofa_parts(chart_file, lab_file, inpm_file, inpc_file, out_file):
    print('getting sofa parts')
    labs = pd.read_csv(lab_file, dtype=str, na_values='NAN')
    inp_m = pd.read_csv(inpm_file, dtype=str, na_values='NAN')
    outp = pd.read_csv(out_file, dtype=str, na_values='NAN')
    print('raw tables', labs.shape, inp_m.shape, outp.shape)

    #respiration (pao2fio2 vent, pao2fio2 no vent) -> later calculate ratio of 100*pao2 / fio2
    keep = '50821' #keep values <= 800 later
    pao2 = labs[labs['ITEMID'].str.contains(keep)][['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']]
    print('pao2', pao2.shape)
    keep = '50816' #keep values 21-100 inclusive later
    fio2_lab = labs[labs['ITEMID'].str.contains(keep)][['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']]
    print('fio2_lab', fio2_lab.shape)
    pao2['NAME'] = 'pao2'
    fio2_lab['NAME'] = 'fio2_lab'
    
    #coagulation (platelet)
    keep = '51265' 
    platelet = labs[labs['ITEMID'].str.contains(keep)][['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']]
    print('platelet', platelet.shape)

    #liver (bilirubin)
    keep = '50885' 
    bilirubin = labs[labs['ITEMID'].str.contains(keep)][['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']]
    print('bilirubin', bilirubin.shape)

    #cardiovascular (dopamine, epinephrine, norepinephrine, dobutamine, map)
    keep = {'norepinephrine': '221906', 'epinephrine': '221289', 'dopamine': '221662', 'dobutamine': '221653'}
    cardio = {}
    for key in list(keep.keys()):
        cardio[key] = inp_m[inp_m['ITEMID'].str.contains(keep[key])][['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME', 'RATE']]
        print(key, cardio[key].shape)

    #renal (creatinine, urine output)
    keep = '50912' 
    creatinine = labs[labs['ITEMID'].str.contains(keep)][['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']]
    creatinine['NAME'] = 'creatinine'
    print('creatinine', creatinine.shape)
    keep = '40055|43175|40069|40094|40715|40473|40085|40057|40056|40405|40428|40086|40096|40651' 
    urine = outp[outp['ITEMID'].str.contains(keep)][['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']]
    urine['NAME'] = 'urine'
    renal = pd.concat([creatinine, urine], axis=0)
    print('urine', urine.shape)

    #dealing with chartevents - can't read entire table because big
    chunk = 2000000
    fio2_chart, vents = [], []
    gcs_eyes, gcs_verbal, gcs_motor = [], [], []
    mean_bp = []
    chart_cols = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']

    with pd.read_csv(chart_file, chunksize=chunk, dtype=str, na_values='NAN') as whole_chart:
        for chart in whole_chart:
            #respiration (getting vent status and some fio2)
            keep = '223835|3420|190|3422' #keep values 21-100 inclusive later
            fio2_chart.append(chart[chart['ITEMID'].str.contains(keep)][['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE']])
            keep = '445|448|449|450|1340|1486|1600|224687|639|654|681|682|683|684|224684|224686|' \
               + '218|436|535|444|459|224697|224695|224696|224746|224747|221|1|1211|1655|2000|' \
               + '226873|224738|224419|224750|227187|543|5865|5866|224707|224709|224705|224706|' \
               + '60|437|505|506|686|220339|224700|3459|501|502|503|224702|223|667|668|669|670|' \
               + '671|672|224701|223849'
            vent1 = chart[chart['ITEMID'].str.match(keep)][chart_cols].drop_duplicates()
            vent2 = chart[chart['ITEMID'].str.match('720') & chart['VALUE'] != 'Other/Remarks'][chart_cols].drop_duplicates()
            vent3 = chart[chart['ITEMID'].str.match('223848') & chart['VALUE'] != 'Other'][chart_cols].drop_duplicates()
            vent4 = chart[chart['ITEMID'].str.match('467') & chart['VALUE'] == 'Ventilator'][chart_cols].drop_duplicates()
            keep = 'Nasal cannula|Face tent|Aerosol-cool|Trach mask|High flow neb|Non-rebreather|Venti mask|' \
               + 'Medium conc mask|T-piece|High flow nasal cannula|Ultrasonic neb|Vapomist'
            vent5 = chart[chart['ITEMID'].str.match('226732') & chart['VALUE'].str.contains(keep)][chart_cols].drop_duplicates()
            keep = 'Cannula|Nasal Cannula|None|Face Tent|Aerosol-Cool|Trach Mask|Hi Flow Neb|Non-Rebreather|' \
               + 'Venti Mask|Medium Conc Mask|Vapotherm|T-Piece|Hood|Hut|TranstrachealCat|Heated Neb|UltrasonicNeb'
            vent6 = chart[chart['ITEMID'].str.match('467') & chart['VALUE'].str.contains(keep)][chart_cols].drop_duplicates()
            vent7 = chart[chart['ITEMID'].str.match('640') & chart['VALUE'] == 'Extubated'][chart_cols].drop_duplicates()
            vent8 = chart[chart['ITEMID'].str.match('640') & chart['VALUE'] == 'Self Extubation'][chart_cols].drop_duplicates()
            vents.append(pd.concat([vent1, vent2, vent3, vent4, vent5, vent6, vent7, vent8], axis=0).drop_duplicates())

            #cns (gcs)
            gcs_motor.append(chart[chart['ITEMID'].str.match('454')][chart_cols])
            gcs_verbal.append(chart[chart['ITEMID'].str.match('723')][chart_cols])
            gcs_eyes.append(chart[chart['ITEMID'].str.match('184')][chart_cols])

            #cardiovascular (mean bp)
            keep = '456|52|6702|443|220052|220181|225312'
            mean_bp.append(chart[chart['ITEMID'].str.match(keep)][chart_cols])

    fio2_chart = pd.concat(fio2_chart, axis=0)
    fio2_chart['NAME'] = 'fio2_chart'
    vents = pd.concat(vents, axis=0)
    vents['NAME'] = 'vent'
    gcs_motor, gcs_verbal, gcs_eyes = pd.concat(gcs_motor, axis=0), pd.concat(gcs_verbal, axis=0), pd.concat(gcs_eyes, axis=0)
    gcs_motor['NAME'] = 'motor'
    gcs_verbal['NAME'] = 'verbal'
    gcs_eyes['NAME'] = 'eyes'
    mean_bp = pd.concat(mean_bp, axis=0).drop_duplicates()

    #save it so don't have to run this again
    sofa_parts = {'pao2': pao2, 'fio2_lab': fio2_lab, 'fio2_chart': fio2_chart, 'vents': vents, \
                  'coagulation': platelet, 'liver': bilirubin, 'cardiovascular': cardio, \
                  'gcs_motor': gcs_motor, 'gcs_eyes': gcs_eyes, 'gcs_verbal': gcs_verbal, \
                  'creat': creatinine, 'urine': urine, 'mean_bp': mean_bp} 
    for key in list(sofa_parts.keys()):
        if key == 'cardiovascular':
            for key2 in list(cardio.keys()):
                print(key2, cardio[key2].shape)
                cardio[key2]['VALUE'] = cardio[key2]['RATE']
                cardio[key2].to_csv(save_to + 'lab_preprocess_sofa_' + key2 + '.csv')
        else:
            print(key, sofa_parts[key].shape)
            sofa_parts[key].to_csv(save_to + 'lab_preprocess_sofa_' + key + '.csv')
    
    return sofa_parts


#######################################################
'''
make hourly windows relative to admission time
'''
def find_windows(adm_file, raw_data, keep_name=False):
    adm = pd.read_csv(adm_file, dtype=str, na_values='NAN')

    #find time difference between each value and admission time
    raw_data = raw_data.merge(adm, on='HADM_ID')
    windowed_data = {'hadm_id': raw_data['HADM_ID'], 'name': raw_data['NAME']}

    admit_time = pd.to_datetime(raw_data['ADMITTIME'], errors='coerce')
    if 'CHARTTIME' in list(raw_data.columns):
        charttime = pd.to_datetime(raw_data['CHARTTIME'], errors='coerce')
        diff = np.floor((charttime - admit_time)/pd.Timedelta(hours=1))
        diff[diff < 0] = 0
        windowed_data['charttime_diff'] = diff
    if 'STARTTIME' in list(raw_data.columns):
        starttime = pd.to_datetime(raw_data['STARTTIME'], errors='coerce')
        diff = np.floor((starttime - admit_time)/pd.Timedelta(hours=1))
        diff[diff < 0] = 0
        windowed_data['starttime_diff'] = diff
    if 'ENDTIME' in list(raw_data.columns):
        endtime = pd.to_datetime(raw_data['ENDTIME'], errors='coerce')
        diff = np.floor((endtime - admit_time)/pd.Timedelta(hours=1))
        diff[diff < 0] = 0
        windowed_data['endtime_diff'] = diff
    if 'STARTDATE' in list(raw_data.columns):
        endtime = pd.to_datetime(raw_data['STARTDATE'], errors='coerce')
        diff = np.floor((endtime - admit_time)/pd.Timedelta(hours=1))
        diff[diff < 0] = 0
        windowed_data['startdate_diff'] = diff

    windowed_data['value'] = pd.to_numeric(raw_data['VALUE'], errors='coerce')
    if keep_name:
        windowed_data['name'] = raw_data['NAME']

    return pd.DataFrame(windowed_data)


'''
given windowed data, find the most recent window relative to input window 
'''
def find_recent_window(all_data, window, col):
    to_consider = all_data[all_data[col] <= window][['hadm_id', col]]
    to_consider = to_consider.groupby('hadm_id').agg('max')
    return to_consider


'''
calculate sirs scores
based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/severityscores/sirs.sql
'''
def combine_sirs_parts(adm_file):
    #calculate sirs from temperature (min, max) keep (70,120) or (10, 50), heart rate (max) keep (0, 300), 
    #respiratory rate (max) keep (0, 70), paco2 (min), white blood cell count (min, max) exclude >1000, 
    #presence of >10% immature neutrophils (band forms) exclude <0 and >100

    adms = pd.read_csv(adm_file, dtype=str, na_values='NAN')[['HADM_ID', 'ADMITTIME']]
    adms['hadm_id'] = adms['HADM_ID']
    
    #gather the parts and window
    part_names = ['temp_f', 'temp_c', 'heartrate', 'resprate', 'paco2', 'wbc', 'neutophil_band']
    parts = {}
    max_window = 0
    for i in range(len(part_names)):
        part = pd.read_csv(save_to + 'lab_preprocess_sirs_' + part_names[i] + '.csv', dtype=str, na_values='NAN')
        part['NAME'] = part_names[i]
        part = find_windows(adm_file, part).drop_duplicates()
        #take out odd values
        part = part[~pd.isnull(part['charttime_diff']) & ~pd.isnull(part['value'])]
        if part_names[i] == 'temp_f':
            part = part[(part['value'] > 70) & (part['value'] < 120)]
            part['value'] = (part['value'] - 32) * (5/9)
            #find how many windows patients usually have
            longest_window = part.groupby('hadm_id').agg('max')['charttime_diff']
            print('distr of num windows', np.percentile(longest_window, [0, 25, 50, 75, 90, 95, 99, 100]))
        elif part_names[i] == 'temp_c':
            part = part[(part['value'] > 10) & (part['value'] < 50)]
        elif part_names[i] == 'heartrate':
            part = part[(part['value'] > 0) & (part['value'] < 300)]
        elif part_names[i] == 'resprate':
            part = part[(part['value'] > 0) & (part['value'] < 70)]
        elif part_names[i] == 'paco2':
            part = part[part['value'] > 0]
        elif part_names[i] == 'wbc':
            part = part[part['value'] < 1000]
        elif part_names[i] == 'neutophil_band':
            part = part[(part['value'] > 0) & (part['value'] < 100)]
        parts[part_names[i]] = part
        print(part_names[i], part[['charttime_diff']].max())
        if part[['charttime_diff']].max().iloc[0] > max_window:
            max_window = part[['charttime_diff']].max().iloc[0]

    parts['temp'] = pd.concat([parts['temp_f'], parts['temp_c']], axis=0).drop_duplicates()
    print('finished windowing. max window: ', max_window)

    #calculate sirs over windows
    sirs_scores = []
    window = 0
    stop_window = 336 #two weeks worth of hours - should be more than enough, too much maybe?

    while window < stop_window:
        print('current window', window)
        to_consider = find_recent_window(parts['temp'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['temp'], on=['hadm_id', 'charttime_diff'])
        max_temp = to_consider.groupby('hadm_id').agg('max')
        min_temp = to_consider.groupby('hadm_id').agg('min')
        temp = max_temp.merge(min_temp, on='hadm_id', how='outer').drop_duplicates()
        temp_score = np.zeros((temp.shape[0],))
        temp_score[temp['value_x'] > 38] = 1
        temp_score[temp['value_y'] < 36] = 1 
        temp_score = pd.DataFrame({'hadm_id': temp.index, 'temp_score': temp_score})

        to_consider = find_recent_window(parts['heartrate'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['heartrate'], on=['hadm_id', 'charttime_diff'])
        max_heart = to_consider.groupby('hadm_id').agg('max')
        heart_score = np.zeros((max_heart.shape[0],))
        heart_score[max_heart['value'] > 90] = 1 
        heart_score = pd.DataFrame({'hadm_id': max_heart.index, 'heart_score': heart_score})

        to_consider = find_recent_window(parts['resprate'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['resprate'], on=['hadm_id', 'charttime_diff'])
        max_resp = to_consider.groupby('hadm_id').agg('max')
        to_consider = find_recent_window(parts['paco2'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['paco2'], on=['hadm_id', 'charttime_diff'])
        min_paco = to_consider.groupby('hadm_id').agg('min')
        resp_paco = max_resp.merge(min_paco, on='hadm_id', how='outer').drop_duplicates()
        resp_score = np.zeros((resp_paco.shape[0],))
        resp_score[resp_paco['value_x'] > 20] = 1 
        resp_score[resp_paco['value_y'] < 32] = 1 
        resp_score = pd.DataFrame({'hadm_id': resp_paco.index, 'resp_score': resp_score})

        to_consider = find_recent_window(parts['wbc'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['wbc'], on=['hadm_id', 'charttime_diff'])
        max_wbc = to_consider.groupby('hadm_id').agg('max')
        min_wbc = to_consider.groupby('hadm_id').agg('min')
        to_consider = find_recent_window(parts['neutophil_band'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['neutophil_band'], on=['hadm_id', 'charttime_diff'])
        max_band = to_consider.groupby('hadm_id').agg('max')
        wbc_band = max_wbc.merge(min_wbc, on='hadm_id', how='outer')
        wbc_band = wbc_band.merge(max_band, on='hadm_id', how='outer').drop_duplicates()
        wbc_score = np.zeros((wbc_band.shape[0],))
        wbc_score[wbc_band['value_x'] > 12] = 1
        wbc_score[wbc_band['value_y'] < 4] = 1 
        wbc_score[wbc_band['value'] > 10] = 1 
        wbc_score = pd.DataFrame({'hadm_id': wbc_band.index, 'wbc_score': wbc_score})

        sirs = adms.merge(temp_score, on='hadm_id', how='left')
        sirs = sirs.merge(heart_score, on='hadm_id', how='left')
        sirs = sirs.merge(resp_score, on='hadm_id', how='left')
        sirs = sirs.merge(wbc_score, on='hadm_id', how='left')
        sirs = sirs.fillna(value=0)
        sirs['sirs_score'] = sirs['temp_score'] + sirs['heart_score'] + sirs['resp_score'] + sirs['wbc_score']
        sirs['window'] = window
        sirs_scores.append(sirs)

        window += 1
    
    sirs_scores = pd.concat(sirs_scores, axis=0).drop_duplicates()
    sirs_scores.to_csv(save_to + 'lab_sirs_scores.csv')
    return sirs_scores


'''
calculate sofa scores
https://github.com/BorgwardtLab/mgp-tcn/blob/master/src/query/sofa-hourly.sql
'''
def combine_sofa_parts(adm_file):
    #calculate sofa score from {'pao2': pao2, 'fio2_lab': fio2_lab, 'fio2_chart': fio2_chart, 'vents': vents, \
    #'coagulation': platelet, 'liver': bilirubin, 'cardiovascular': cardio, \
    #'gcs_motor': gcs_motor, 'gcs_eyes': gcs_eyes, 'gcs_verbal': gcs_verbal, 'renal': renal, 'mean_bp': mean_bp} 
    #cardio: 'norepinephrine', 'epinephrine', 'dopamine', 'dobutamine'

    adms = pd.read_csv(adm_file, dtype=str, na_values='NAN')[['HADM_ID', 'ADMITTIME']]
    adms['hadm_id'] = adms['HADM_ID']
    stop_window = 336
    
    #gather the parts and window
    part_names = ['pao2', 'fio2_lab', 'fio2_chart', 'vents', 'coagulation', 'liver', 'norepinephrine', 'epinephrine', \
                  'dopamine', 'dobutamine', 'gcs_motor', 'gcs_eyes', 'gcs_verbal', 'creat', 'urine', 'mean_bp']
    parts = {}
    for i in range(len(part_names)):
        print('windowing', part_names[i])
        part = pd.read_csv(save_to + 'lab_preprocess_sofa_' + part_names[i] + '.csv', dtype=str, na_values='NAN').drop_duplicates()
        if part_names[i] != 'vents':
            part = find_windows(adm_file, part)
        else: #have to process vents in chunks because big
            print(part.head(10))
            print(part.shape)
            part['VALUE'] = 1
            part = part.drop_duplicates()
            print(part.shape)
            return
        #take out odd values
        if 'charttime_diff' in list(part.columns):
            part = part[~pd.isnull(part['charttime_diff']) & ~pd.isnull(part['value'])]
        if 'starttime_diff' in list(part.columns):
            part = part[~pd.isnull(part['starttime_diff']) & ~pd.isnull(part['value'])]
        if 'endtime_diff' in list(part.columns):
            part = part[~pd.isnull(part['end_diff']) & ~pd.isnull(part['value'])]

        if part_names[i] == 'pao2':
            part = part[(part['value'] > 0) & (part['value'] < 800)]
        elif part_names[i] == 'fio2_lab':
            part = part[(part['value'] > 0) & (part['value'] < 100)]
        elif part_names[i] == 'fio2_chart':
            part = part[(part['value'] > 0) & (part['value'] < 100)]
        elif part_names[i] == 'vents':
            a = 1
        elif part_names[i] == 'coagulation':
            part = part[(part['value'] > 0) & (part['value'] < 10000)]
        elif part_names[i] == 'liver':
            part = part[(part['value'] > 0) & (part['value'] < 150)]
        elif part_names[i] == 'norepinephrine':
            a = 1
        elif part_names[i] == 'epinephrine':
            a = 1
        elif part_names[i] == 'dopamine':
            a = 1
        elif part_names[i] == 'dobutamine':
            a = 1
        elif part_names[i] == 'gcs_motor':
            part = part[(part['value'] > 2) & (part['value'] < 16)]
        elif part_names[i] == 'gcs_eyes':
            part = part[part['value'] > 2 & part['value'] < 16]
        elif part_names[i] == 'gcs_verbal':
            part = part[part['value'] > 2 & part['value'] < 16]
        elif part_names[i] == 'creat':
            part = part[part['value'] > 0 & part['value'] < 150]
        elif part_names[i] == 'urine':
            a = 1
        elif part_names[i] == 'mean_bp':
            a = 1
        parts[part_names[i]] = part

    parts['fio2'] = pd.concat([parts['fio2_lab'], parts['fio2_chart']], axis=0).drop_duplicates()
    print('finished windowing')

    #calculate sofa over windows
    sofa_scores = []
    window = 333

    while window < stop_window:
        print(window)
        to_consider = find_recent_window(parts['poa2'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['poa2'], on=['hadm_id', 'charttime_diff'])
        poa2_min = to_consider.groupby('hadm_id').agg('min')
        poa2_min['poa2'] = poa2_min['value']
        to_consider = find_recent_window(parts['fio2'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['fio2'], on=['hadm_id', 'charttime_diff'])
        fio2_max = to_consider.groupby('hadm_id').agg('max')
        fio2_max['fio2'] = fio2_max['value']
        vents = find_recent_window(parts['vents'], window, 'charttime_diff')
        pofi = pao2_min.merge(fio2_max, on='hadm_id', how='outer')
        pofi['ratio'] = pofi['poa2']*100/pofi['fio2']
        resp_score = np.zeros((pofi.shape[0],))
        resp_score[pofi['ratio'] < 100 & pofi['hadm_id'].isin(vents['hadm_id'])] = 4
        resp_score[pofi['ratio'] < 200 & pofi['raio'] >= 100 & pofi[['hadm_id']].isin(vents['hadm_id'])] = 3
        resp_score[pofi['ratio'] < 300 & pofi['raio'] >= 200 & ~pofi[['hadm_id']].isin(vents['hadm_id'])] = 2
        resp_score[pofi['ratio'] < 300 & pofi['raio'] >= 300 & ~pofi[['hadm_id']].isin(vents['hadm_id'])] = 1
        resp_score = pd.DataFrame({'hadm_id': pofi.index, 'resp_score': resp_score})

        to_consider = find_recent_window(parts['coagulation'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['coagulation'], on=['hadm_id', 'charttime_diff'])
        min_val = to_consider.groupby('hadm_id').agg('min')
        coag_score = np.zeros((min_val.shape[0],))
        coag_score[min_val['value'] < 20] = 4
        coag_score[min_val['value'] >= 20 & min_val['value'] < 50] = 3
        coag_score[min_val['value'] >= 50 & min_val['value'] < 100] = 2
        coag_score[min_val['value'] >= 100 & min_val['value'] < 150] = 1
        coag_score = pd.DataFrame({'hadm_id': min_val.index, 'coag_score': coag_score})

        to_consider = find_recent_window(parts['liver'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['liver'], on=['hadm_id', 'charttime_diff'])
        max_val = to_consider.groupby('hadm_id').agg('max')
        liver_score = np.zeros((max_val.shape[0],))
        liver_score[max_val >= 12] = 4
        liver_score[max_val >= 6 & max_val < 12] = 3
        liver_score[max_val >= 2 & max_val < 6] = 2
        liver_score[max_val >= 1.2 & max_val < 2] = 1
        liver_score = pd.DataFrame({'hadm_id': max_val.index, 'liver_score': liver_score})

        to_consider = find_recent_window(parts['dopamine'], window, 'starttime_diff')
        to_consider = to_consider.merge(parts['dopamine'], on=['hadm_id', 'starttime_diff'])
        dop_max = to_consider.groupby('hadm_id').agg('max')
        dop_max['dop_rate'] = dop_max['value']
        to_consider = find_recent_window(parts['dobutamine'], window, 'starttime_diff')
        to_consider = to_consider.merge(parts['dobutamine'], on=['hadm_id', 'starttime_diff'])
        dob_max = to_consider.groupby('hadm_id').agg('max')
        dob_max['dob_rate'] = dob_max['value']
        to_consider = find_recent_window(parts['epinephrine'], window, 'starttime_diff')
        to_consider = to_consider.merge(parts['epinephrine'], on=['hadm_id', 'starttime_diff'])
        epi_max = to_consider.groupby('hadm_id').agg('max')
        epi_max['epi_rate'] = epi_max['value']
        to_consider = find_recent_window(parts['norepinephrine'], window, 'starttime_diff')
        to_consider = to_consider.merge(parts['norepinephrine'], on=['hadm_id', 'starttime_diff'])
        nor_max = to_consider.groupby('hadm_id').agg('max')
        nor_max['nor_rate'] = nor_max['value']
        to_consider = find_recent_window(parts['mean_bp'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['mean_bp'], on=['hadm_id', 'charttime_diff'])
        mbp_min = to_consider.groupby('hadm_id').agg('min')
        mbp_min['mbp_val'] = mbp_min['value']
        cardio = dop_max.merge(dob_max, on='hadm_id', how='outer')
        cardio = cardio.merge(epi_max, on='hadm_id', how='outer')
        cardio = cardio.merge(nor_max, on='hadm_id', how='outer')
        cardio = cardio.merge(mbp_min, on='hadm_id', how='outer')
        cardio_score = np.zeros((cardio.shape[0],))
        cardio_score[cardio['dop_rate']>15 | cardio['epi_rate']>0.1 | cardio['nor_rate']>0.1] = 4
        cardio_score[(cardio['dop_rate']>5 | cardio['epi_rate']>0 | cardio['nor_rate']>0) & cardio_score == 0] = 3
        cardio_score[(cardio['dop_rate']>0 | cardio['dob_rate']>0) & cardio_score == 0] = 2
        cardio_score[cardio['mbp_val']<70 & cardio_score == 0] = 1
        cardio_score = pd.DataFrame({'hadm_id': cardio.index, 'cardio_score': cardio_score})

        to_consider = find_recent_window(parts['gcs_motor'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['gcs_motor'], on=['hadm_id', 'charttime_diff'])
        motor_min = to_consider.groupby('hadm_id').agg('min')
        to_consider = find_recent_window(parts['gcs_eyes'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['gcs_eyes'], on=['hadm_id', 'charttime_diff'])
        eyes_min = to_consider.groupby('hadm_id').agg('min')
        to_consider = find_recent_window(parts['gcs_verbal'], window, 'charttime_diff')
        to_consider = to_consider.merge(parts['gcs_verbal'], on=['hadm_id', 'charttime_diff'])
        verbal_min = to_consider.groupby('hadm_id').agg('min')
        gcs = pd.concat([motor_min, eyes_min, verbal_min], axis=0)
        gcs_min = gcs.groupby('hadm_id').agg('min')
        cns_score = np.zeros((gcs_min.shape[0],))
        cns_score[gcs_min['value'] >= 13 & gcs_min['value'] <= 14] = 1
        cns_score[gcs_min['value'] >= 10 & gcs_min['value'] <= 12] = 2
        cns_score[gcs_min['value'] >= 6 & gcs_min['value'] <= 9] = 3
        cns_score[gcs_min['value'] < 6] = 4
        cns_score = pd.DataFrame({'hadm_id': gcs_min.index, 'cns_score': cns_score})

        to_consider_c = find_recent_window(parts['creat'], window, 'charttime_diff')
        to_consider_c = to_consider.merge(parts['creat'], on=['hadm_id', 'charttime_diff'])
        max_creat = to_consider.groupby('hadm_id').agg('max')
        to_consider_u = find_recent_window(parts['urine'], window, 'charttime_diff')
        to_consider_u = to_consider.merge(parts['urine'], on=['hadm_id', 'charttime_diff'])
        sum_urine = to_consider.groupby('hadm_id').agg('sum')
        creat_urine = max_creat.merge(sum_urine, on='hadm_id', how='outer')
        renal_score = np.zeros((creat_urine.shape[0],))
        renal_score[creat_urine['value_x'] >= 5] = 4
        renal_score[creat_urine['value_y'] < 200] = 4
        renal_score[creat_urine['value_x'] >= 3.5 & creat_urine['valye_x'] < 5] = 3
        renal_score[creat_urine['value_y'] >= 200 & creat_urine['valye_y'] < 500] = 3
        renal_score[creat_urine['value_x'] >= 2 & creat_urine['valye_x'] < 3.5] = 2
        renal_score[creat_urine['value_x'] >= 1.2 & creat_urine['valye_x'] < 2] = 1
        renal_score = pd.DataFrame({'hadm_id': creat_urine.index, 'renal_score': renal_score})

        sofa = adms.merge(resp_score, on='hadm_id', how='left')
        sofa = sofa.merge(coag_score, on='hadm_id', how='left')
        sofa = sofa.merge(liver_score, on='hadm_id', how='left')
        sofa = sofa.merge(cardio_score, on='hadm_id', how='left')
        sofa = sofa.merge(cns_score, on='hadm_id', how='left')
        sofa = sofa.merge(renal_score, on='hadm_id', how='left')
        sofa = sofa.fillna(value=0)
        sofa['sofa_score'] = sofa['resp_score'] + sofa['coag_score'] + sofa['liver_score'] \
                            + sofa['cardio_score'] + sofa['cns_score'] + sofa['renal_score']
        sofa['window'] = window
        sofa_scores.append(sofa)

        window += 1

    sofa_scores = pd.concat(sofa_scores, axis=0).drop_duplicates()
    return sofa_scores


'''
get antibiotics and window them
antibiotics are stored by date rather than time so windowing may not be accurate
previous work does this too, doesn't seem to be a good way around it
'''
def window_antibiotics(adms_file):
    part = pd.read_csv(save_to + 'lab_preprocess_antibiotics.csv', dtype=str, na_values='NAN')
    part['NAME'] = 'antibiotic'
    part['VALUE'] = 1
    part = find_windows(adm_file, part)
    return part.drop_duplicates()


#######################################################
'''
calculate time of sepsis (sepsis1)
format output as labels
'''
def find_sepsis1_onset(adm_file, antibiotics, cultures):
    sirs_scores = pd.read_csv(save_to + 'lab_sirs_scores.csv')
    adms = pd.read_csv(adm_file, dtype=str, na_values='NAN')[['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME']]

    admit_time = pd.to_datetime(adms['ADMITTIME'], errors='coerce')
    dischtime = pd.to_datetime(adms['DISCHTIME'], errors='coerce')
    deathtime = pd.to_datetime(adms['DEATHTIME'], errors='coerce')
    
    disch_windows = np.floor((dischtime - admit_time)/pd.Timedelta(hours=1))
    disch_windows[disch_windows < 0] = 0
    death_windows = np.floor((deathtime - admit_time)/pd.Timedelta(hours=1))
    death_windows[death_windows < 0] = 0
    adms['disch_window'] = disch_windows
    adms['death_window'] = death_windows

    hadm_id, sepsis_time, last_time = [], [], []
    hadms = np.unique(adms['HADM_ID'])

    max_window = 336 - 1 #minus 1 for 0 indexing
    sirs_scores = sirs_scores[sirs_scores['sirs_score'] >= 2]
    print('sirs >= 2', sirs_scores.shape)
    have_sepsis = 0

    #case 1a: cultures with antibiotics 48 hours after, sirs>=2 48 hours before either
    #case 1b: cultures with antibiotics 48 hours after, sirs>=2 24 hours after either
    #case 2a: antibiotics with cultures 24 hours after, sirs>=2 48 hours before either
    #case 2b: antibiotics with cultures 24 hours after, sirs>=2 2 hours after either
    #all cases: time 0 = earlier of antibiotics or cultures
    for i in range(len(hadms)):
        if i % 1000 == 0: 
            print('iteration', i, have_sepsis)
        hadm_i = hadms[i]
        #last observed time
        last_i = adms[adms['HADM_ID'] == hadm_i]['disch_window'].iloc[0]
        if adms[adms['HADM_ID'] == hadm_i]['death_window'].isna().iloc[0] == False:
            last_i = adms[adms['HADM_ID'] == hadm_i]['death_window'].iloc[0]
        last_i = int(max(max_window, last_i))
        
        #sepsis
        sepsis_i = 'None'
        antib_i = antibiotics[(antibiotics['hadm_id'] == hadm_i) & (~antibiotics['startdate_diff'].isna())]
        sirs_i = sirs_scores[(sirs_scores['HADM_ID'] == int(hadm_i)) & (~sirs_scores['window'].isna())]
        cult_i = cultures[(cultures['hadm_id'].astype(str) == hadm_i)  & (~cultures['charttime_diff'].isna())]

        for j in range(last_i):
            if j in cult_i['charttime_diff'].astype(int).to_numpy(): #case 1: have cultures at this window
                #antibiotics up to 48 hours after
                antib_ij = antib_i[(antib_i['startdate_diff'] >= j) & (antib_i['startdate_diff'] <= j+48)]
                #sirs 48 hours after or 24 hours before
                if antib_ij.shape[0] > 0:
                    antib_time = antib_ij['startdate_diff'].iloc[0]
                    sirs_after = sirs_i[(sirs_i['window'] >= j) & (sirs_i['window'] <= antib_time+24)]
                    sirs_before = sirs_i[(sirs_i['window'] >= j-48) & (sirs_i['window'] < j)]
                    if sirs_after.shape[0] > 0 or sirs_before.shape[0] > 0:
                        sepsis_i = j
                        have_sepsis += 1
                        break

            elif j in antib_i['startdate_diff'].astype(int).to_numpy(): #case 2: have antibiotics at this window
                #cultures up to 24 hours after
                cult_ij = cult_i[(cult_i['charttime_diff'] >= j) & (cult_i['charttime_diff'] <= j+24)]
                #sirs 48 hours after or 24 hours before
                if cult_ij.shape[0] > 0:
                    cult_time = cult_ij['charttime_diff'].iloc[0]
                    sirs_after = sirs_i[(sirs_i['window'] >= j) & (sirs_i['window'] <= cult_time+24)]
                    sirs_before = sirs_i[(sirs_i['window'] >= j-48) & (sirs_i['window'] < j)]
                    if sirs_after.shape[0] > 0 or sirs_before.shape[0] > 0:
                        sepsis_i = j
                        have_sepsis += 1
                        break
        
        hadm_id.append(hadm_i)
        sepsis_time.append(sepsis_i)
        last_time.append(last_i)

    sep1_labs = {'HADM_ID': hadm_id, 'sepsis1 time': sepsis_time, 'last obs': last_time}
    sep1_labs = pd.DataFrame(sep1_labs).drop_duplicates()  
    sep1_labs.to_csv(save_to + 'labels_sep1.csv')

    print('have sepsis', have_sepsis, have_sepsis/len(hadm_id))
    sepsis_time = np.array(sepsis_time)
    sepsis_time = sepsis_time[sepsis_time != 'None'].astype(int)
    print('tte distr', np.percentile(sepsis_time, [0, 5, 25, 50, 75, 95, 100]))
    return sep1_labs


'''
calculate time of sepsis (sepsis1) - cms version
format output as labels
'''
def find_sepsis1_cms_onset(adm_file, antibiotics, cultures):
    sirs_scores = pd.read_csv(save_to + 'lab_sirs_scores.csv')
    adms = pd.read_csv(adm_file, dtype=str, na_values='NAN')[['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME']]

    admit_time = pd.to_datetime(adms['ADMITTIME'], errors='coerce')
    dischtime = pd.to_datetime(adms['DISCHTIME'], errors='coerce')
    deathtime = pd.to_datetime(adms['DEATHTIME'], errors='coerce')
    
    disch_windows = np.floor((dischtime - admit_time)/pd.Timedelta(hours=1))
    disch_windows[disch_windows < 0] = 0
    death_windows = np.floor((deathtime - admit_time)/pd.Timedelta(hours=1))
    death_windows[death_windows < 0] = 0
    adms['disch_window'] = disch_windows
    adms['death_window'] = death_windows

    hadm_id, sepsis_time, last_time = [], [], []
    hadms = np.unique(adms['HADM_ID'])

    max_window = 336 - 1 #minus 1 for 0 indexing
    sirs_scores = sirs_scores[sirs_scores['sirs_score'] >= 2]
    print('sirs >= 2', sirs_scores.shape)
    have_sepsis = 0

    #case 1: antibiotics, sirs>=2 6 hours after
    #case 2: antibiotics, sirs>=2 6 hours before
    #all cases: time 0 = later of antibiotics or sirs
    for i in range(len(hadms)):
        if i % 1000 == 0: 
            print('iteration', i, have_sepsis)
        hadm_i = hadms[i]
        #last observed time
        last_i = adms[adms['HADM_ID'] == hadm_i]['disch_window'].iloc[0]
        if adms[adms['HADM_ID'] == hadm_i]['death_window'].isna().iloc[0] == False:
            last_i = adms[adms['HADM_ID'] == hadm_i]['death_window'].iloc[0]
        last_i = int(max(max_window, last_i))
        
        #sepsis
        sepsis_i = 'None'
        antib_i = antibiotics[(antibiotics['hadm_id'] == hadm_i) & (~antibiotics['startdate_diff'].isna())]
        sirs_i = sirs_scores[(sirs_scores['HADM_ID'] == int(hadm_i)) & (~sirs_scores['window'].isna())]
        #cult_i = cultures[(cultures['hadm_id'].astype(str) == hadm_i)  & (~cultures['charttime_diff'].isna())]

        for j in range(last_i):
            if j in antib_i['startdate_diff'].astype(int).to_numpy(): #antibiotics
                #sirs 6 hours after or 6 hours before
                #if cult_ij.shape[0] > 0:
                sirs_after = sirs_i[(sirs_i['window'] >= j) & (sirs_i['window'] <= j+6)]
                sirs_before = sirs_i[(sirs_i['window'] >= j-6) & (sirs_i['window'] < j)]
                if sirs_before.shape[0] > 0:
                    sepsis_i = j
                    have_sepsis += 1
                    break
                elif sirs_after.shape[0] > 0:
                    sepsis_i = sirs_after['window'].min()
                    have_sepsis += 1
                    break
        
        hadm_id.append(hadm_i)
        sepsis_time.append(sepsis_i)
        last_time.append(last_i)

    sep1_labs = {'HADM_ID': hadm_id, 'sepsis1 time': sepsis_time, 'last obs': last_time}
    sep1_labs = pd.DataFrame(sep1_labs).drop_duplicates()  
    sep1_labs.to_csv(save_to + 'labels_sep1_cms.csv')

    print('have sepsis', have_sepsis, have_sepsis/len(hadm_id))
    sepsis_time = np.array(sepsis_time)
    sepsis_time = sepsis_time[sepsis_time != 'None'].astype(int)
    print('tte distr', np.percentile(sepsis_time, [0, 5, 25, 50, 75, 95, 100]))
    return sep1_labs


'''
calculate time of sepsis (sepsis3)
'''
def find_sepsis3_onset(adm_file, antibiotics, cultures):
    sofa_scores = pd.read_csv(save_to + 'lab_sofa_scores.csv')
    adms = pd.read_csv(adm_file, dtype=str, na_values='NAN')[['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME']]

    admit_time = pd.to_datetime(adms['ADMITTIME'], errors='coerce')
    dischtime = pd.to_datetime(adms['DISCHTIME'], errors='coerce')
    deathtime = pd.to_datetime(adms['DEATHTIME'], errors='coerce')
    
    disch_windows = np.floor((dischtime - admit_time)/pd.Timedelta(hours=1))
    disch_windows[disch_windows < 0] = 0
    death_windows = np.floor((deathtime - admit_time)/pd.Timedelta(hours=1))
    death_windows[death_windows < 0] = 0
    adms['disch_window'] = disch_windows
    adms['death_window'] = death_windows

    hadm_id, sepsis_time, last_time = [], [], []
    hadms = np.unique(adms['HADM_ID'])

    max_window = 336 - 1 #minus 1 for 0 indexing
    sofa_scores = sofa_scores[(sofa_scores['sofa_score'] >= 2) & (sofa_scores['window'] <= 366)]
    cultures = cultures[cultures['charttime_diff'] <= 366]
    antibiotics = antibiotics[antibiotics['startdate_diff'] <= 366]
    print('sofa >= 2', sofa_scores.shape)
    print('num with sofa >= 2', np.unique(sofa_scores['HADM_ID']).shape)
    print('num with culture', np.unique(cultures['hadm_id']).shape)
    print('num with antibiotic', np.unique(antibiotics['hadm_id']).shape)
    have_sepsis = 0

    print(np.intersect1d(np.unique(cultures['hadm_id']), hadms).shape)

    #case 1a: cultures with antibiotics 72 hours after, sofa>=2 48 hours before either
    #case 1b: cultures with antibiotics 72 hours after, sofa>=2 24 hours after either
    #case 2a: antibiotics with cultures 24 hours after, sofa>=2 48 hours before either
    #case 2b: antibiotics with cultures 24 hours after, sofa>=2 2 hours after either
    #all cases: time 0 = earlier of antibiotics or cultures
    for i in range(len(hadms)):
        if i % 1000 == 0: 
            print('iteration', i, have_sepsis)
        hadm_i = hadms[i]
        #last observed time
        last_i = adms[adms['HADM_ID'] == hadm_i]['disch_window'].iloc[0]
        if adms[adms['HADM_ID'] == hadm_i]['death_window'].isna().iloc[0] == False:
            last_i = adms[adms['HADM_ID'] == hadm_i]['death_window'].iloc[0]
        last_i = int(min(max_window, last_i))
        
        #sepsis
        sepsis_i = 'None'
        antib_i = antibiotics[(antibiotics['hadm_id'].astype(str) == hadm_i) & (~antibiotics['startdate_diff'].isna())]
        sofa_i = sofa_scores[(sofa_scores['HADM_ID'] == int(hadm_i)) & (~sofa_scores['window'].isna())]
        cult_i = cultures[(cultures['hadm_id'].astype(str) == hadm_i)  & (~cultures['charttime_diff'].isna())]

        for j in range(last_i):
            if j in cult_i['charttime_diff'].astype(int).to_numpy(): #case 1: have cultures at this window
                #antibiotics up to 72 hours after
                antib_ij = antib_i[(antib_i['startdate_diff'] >= j) & (antib_i['startdate_diff'] <= j+72)]
                #sofa 48 hours after or 24 hours before
                if antib_ij.shape[0] > 0:
                    #print('antibiotic', j, 'startdatediff', antib_ij['startdate_diff'], 'used', antib_ij['startdate_diff'].iloc[0])
                    #print(sofa_i)
                    antib_time = antib_ij['startdate_diff'].iloc[0]
                    sofa_after = sofa_i[(sofa_i['window'] >= j) & (sofa_i['window'] <= antib_time+24)]
                    sofa_before = sofa_i[(sofa_i['window'] >= j-48) & (sofa_i['window'] < j)]
                    if sofa_after.shape[0] > 0 or sofa_before.shape[0] > 0:
                        sepsis_i = j
                        have_sepsis += 1
                        break

            elif j in antib_i['startdate_diff'].astype(int).to_numpy(): #case 2: have antibiotics at this window
                #cultures up to 24 hours after
                cult_ij = cult_i[(cult_i['charttime_diff'] >= j) & (cult_i['charttime_diff'] <= j+24)]
                #sofa 48 hours after or 24 hours before
                if cult_ij.shape[0] > 0:
                    #print('culture', j, 'charttime diff', cult_ij['charttime_diff'], 'used', cult_ij['charttime_diff'].iloc[0])
                    cult_time = cult_ij['charttime_diff'].iloc[0]
                    sofa_after = sofa_i[(sofa_i['window'] >= j) & (sofa_i['window'] <= cult_time+24)]
                    sofa_before = sofa_i[(sofa_i['window'] >= j-48) & (sofa_i['window'] < j)]
                    if sofa_after.shape[0] > 0 or sofa_before.shape[0] > 0:
                        sepsis_i = j
                        have_sepsis += 1
                        break
        
        hadm_id.append(hadm_i)
        sepsis_time.append(sepsis_i)
        last_time.append(last_i)

    sep3_labs = {'HADM_ID': hadm_id, 'sepsis3 time': sepsis_time, 'last obs': last_time}
    sep3_labs = pd.DataFrame(sep3_labs).drop_duplicates()  
    sep3_labs.to_csv(save_to + 'labels_sep3.csv')

    print('have sepsis', have_sepsis, have_sepsis/len(hadm_id))
    sepsis_time = np.array(sepsis_time)
    sepsis_time = sepsis_time[sepsis_time != 'None'].astype(int)
    print('tte distr', np.percentile(sepsis_time, [0, 5, 25, 50, 75, 95, 100]))
    return sep3_labs


'''
calculate time of sepsis (sepsis3) - rhee (cdc) edition
'''
def find_sepsis3_rhee_onset(adm_file, antibiotics, cultures):
    sofa_scores = pd.read_csv(save_to + 'lab_sofa_scores.csv')
    adms = pd.read_csv(adm_file, dtype=str, na_values='NAN')[['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME']]

    admit_time = pd.to_datetime(adms['ADMITTIME'], errors='coerce')
    dischtime = pd.to_datetime(adms['DISCHTIME'], errors='coerce')
    deathtime = pd.to_datetime(adms['DEATHTIME'], errors='coerce')
    
    disch_windows = np.floor((dischtime - admit_time)/pd.Timedelta(hours=1))
    disch_windows[disch_windows < 0] = 0
    death_windows = np.floor((deathtime - admit_time)/pd.Timedelta(hours=1))
    death_windows[death_windows < 0] = 0
    adms['disch_window'] = disch_windows
    adms['death_window'] = death_windows

    hadm_id, sepsis_time, last_time = [], [], []
    hadms = np.unique(adms['HADM_ID'])

    max_window = 336 - 1 #minus 1 for 0 indexing
    sofa_scores = sofa_scores[(sofa_scores['sofa_score'] >= 2) & (sofa_scores['window'] <= 366)]
    cultures = cultures[cultures['charttime_diff'] <= 366]
    antibiotics = antibiotics[antibiotics['startdate_diff'] <= 366]
    print('sofa >= 2', sofa_scores.shape)
    print('num with sofa >= 2', np.unique(sofa_scores['HADM_ID']).shape)
    print('num with culture', np.unique(cultures['hadm_id']).shape)
    print('num with antibiotic', np.unique(antibiotics['hadm_id']).shape)
    have_sepsis = 0

    print(np.intersect1d(np.unique(cultures['hadm_id']), hadms).shape)

    #case 1a: cultures with antibiotics 48 hours before, sofa>=2 48 hours before cultures
    #case 1b: cultures with antibiotics 48 hours after, sofa>=2 48 hours after cultures
    #all cases: time 0 = cultures
    for i in range(len(hadms)):
        if i % 1000 == 0: 
            print('iteration', i, have_sepsis)
        hadm_i = hadms[i]
        #last observed time
        last_i = adms[adms['HADM_ID'] == hadm_i]['disch_window'].iloc[0]
        if adms[adms['HADM_ID'] == hadm_i]['death_window'].isna().iloc[0] == False:
            last_i = adms[adms['HADM_ID'] == hadm_i]['death_window'].iloc[0]
        last_i = int(min(max_window, last_i))
        
        #sepsis
        sepsis_i = 'None'
        antib_i = antibiotics[(antibiotics['hadm_id'].astype(str) == hadm_i) & (~antibiotics['startdate_diff'].isna())]
        sofa_i = sofa_scores[(sofa_scores['HADM_ID'] == int(hadm_i)) & (~sofa_scores['window'].isna())]
        cult_i = cultures[(cultures['hadm_id'].astype(str) == hadm_i)  & (~cultures['charttime_diff'].isna())]

        for j in range(last_i):
            if j in cult_i['charttime_diff'].astype(int).to_numpy(): #case 1: have cultures at this window
                #antibiotics up to 48 hours before or after
                antib_ij = antib_i[(antib_i['startdate_diff'] >= j-48) & (antib_i['startdate_diff'] <= j+48)]
                #sofa 48 hours after or 24 hours before
                if antib_ij.shape[0] > 0:
                    #print('antibiotic', j, 'startdatediff', antib_ij['startdate_diff'], 'used', antib_ij['startdate_diff'].iloc[0])
                    #print(sofa_i)
                    antib_time = antib_ij['startdate_diff'].iloc[0]
                    sofa_after = sofa_i[(sofa_i['window'] >= j) & (sofa_i['window'] <= j+48)]
                    sofa_before = sofa_i[(sofa_i['window'] >= j-48) & (sofa_i['window'] < j)]
                    if sofa_after.shape[0] > 0 or sofa_before.shape[0] > 0:
                        sepsis_i = j
                        have_sepsis += 1
                        break
        
        hadm_id.append(hadm_i)
        sepsis_time.append(sepsis_i)
        last_time.append(last_i)

    sep3_labs = {'HADM_ID': hadm_id, 'sepsis3 time': sepsis_time, 'last obs': last_time}
    sep3_labs = pd.DataFrame(sep3_labs).drop_duplicates()  
    sep3_labs.to_csv(save_to + 'labels_sep3_rhee.csv')

    print('have sepsis', have_sepsis, have_sepsis/len(hadm_id))
    sepsis_time = np.array(sepsis_time)
    sepsis_time = sepsis_time[sepsis_time != 'None'].astype(int)
    print('tte distr', np.percentile(sepsis_time, [0, 5, 25, 50, 75, 95, 100]))
    return sep3_labs


'''
composite sepsis definition - use sep3 if available, then use sep1 if it's there and sep3 is not
'''
def find_composite_sepsis_onset(original=True):
    sep1 = pd.read_csv(save_to + 'labels_sep1.csv')
    sep3 = pd.read_csv(save_to + 'labels_sep3.csv')
    if not original:
        sep1 = pd.read_csv(save_to + 'labels_sep1_rhee.csv') #use rhee if cms not available
        sep3 = pd.read_csv(save_to + 'labels_sep3_cms.csv') #use cms if available

    composite = sep1.merge(sep3, on=['HADM_ID'])
    hadms = composite['HADM_ID'].to_numpy()
    print(composite.shape, composite.columns)
 
    comp_lab = []
    comp_and = []
    last_obs = []
    num_diff = 0
    num_both = 0
    for i in range(len(hadms)):
        hadm = hadms[i] 
        comp_time = 'None' 
        comp_and_time = 'None'
        sep1_lab = composite['sepsis1 time'].iloc[i]
        sep3_lab = composite['sepsis3 time'].iloc[i]
        if sep3_lab != 'None':
            comp_time = sep3_lab
        elif sep1_lab != 'None':
            comp_time = sep1_lab
        if sep1_lab != 'None' and sep3_lab !='None': 
            num_both += 1
            if sep1_lab != sep3_lab:
                print(sep1_lab, sep3_lab)
                num_diff += 1
            comp_and_time = sep3_lab
        comp_lab.append(comp_time)
        comp_and.append(comp_and_time)
        last_obs.append(max(composite['last obs_x'].iloc[i], composite['last obs_y'].iloc[i]))
    composite['sepsis comp'] = comp_lab
    composite['last obs'] = last_obs
    composite['sepsis comp and'] = comp_and
    if original:
        composite[['HADM_ID', 'sepsis1 time', 'sepsis3 time', 'sepsis comp', 'last obs']].to_csv(save_to + 'labels_composite.csv')
    else:
        composite[['HADM_ID', 'sepsis1 time', 'sepsis3 time', 'sepsis comp', 'sepsis comp and', 'last obs']].to_csv(save_to + 'labels_composite_cmscdc.csv')

    print(num_both, num_diff)    

    return composite


#######################################################
'''
use inclusion exclusion based on https://proceedings.mlr.press/v106/moor19a/moor19a.pdf
'''
def get_pop(adm_file, icu_file, pat_file, lab_file):
    adm = pd.read_csv(adm_file, dtype=str, na_values='NAN')
    icu = pd.read_csv(icu_file, dtype=str, na_values='NAN')    
    pat = pd.read_csv(pat_file, dtype=str, na_values='NAN')
    
    pop = adm.merge(icu, on='HADM_ID').drop_duplicates()
    pop = pop.merge(pat, left_on='SUBJECT_ID_x', right_on='SUBJECT_ID').drop_duplicates()
    print(pop.shape)

    #keep patients >= 15 years
    pat_dob = pd.to_datetime(pop['DOB'], errors='coerce') 
    pat_admit = pd.to_datetime(pop['ADMITTIME'], errors='coerce')
    pat_age = np.floor((pat_admit.to_numpy() - pat_dob.to_numpy())/np.timedelta64(365, 'D'))
    pop['pat_age'] = pat_age
    pop = pop[pat_age >= 15]
    print('remove < 15', pop.shape)

    #exclude carevue
    pop['DBSOURCE'] = pop['DBSOURCE'].str.lower()
    pop['DBSOURCE'] = pop['DBSOURCE'].fillna('NAN')
    filter_out = ['carevue']
    for i in range(len(filter_out)): 
        pop = pop[~pop['DBSOURCE'].str.contains(filter_out[i])]
    print('exclude carevue', pop.shape)

    #exclude sepsis <= 7 hours after admission
    labs = pd.read_csv(lab_file, dtype=str, na_values='NAN')
    pop = pop.merge(labs, on='HADM_ID', how='left')
    early_sepsis = pop['sepsis comp'].to_numpy()
    keep = ''
    num_with_sepsis = 0
    for i in range(len(early_sepsis)):
        if early_sepsis[i] == 'None':
            keep += pop['HADM_ID'].iloc[i] + '|'
        elif int(early_sepsis[i]) > 7:
            keep += pop['HADM_ID'].iloc[i] + '|'
            num_with_sepsis += 1
    pop = pop[pop['HADM_ID'].str.match(keep[:-1])]
    print('exclude (composite) sepsis <= 7 hours', pop.shape)
    print('number with sepsis', num_with_sepsis)
            
    pop.to_csv(save_to + 'study_population.csv')
    return pop


'''
get features to train models
'''
def get_feats(pop, diag_file):
    #ID, t, variable_name, variable_value -> fiddle format

    #demographics
    names = ['ETHNICITY', 'pat_age', 'GENDER', 'MARITAL_STATUS']
    demographics = []
    for name in names:
        dem_i = pd.DataFrame({'ID': pop['HADM_ID'], 'variable_value': pop[name]})
        dem_i['t'] = 'NULL'
        dem_i['variable_name'] = name
        demographics.append(dem_i)
    demographics = pd.concat(demographics, axis=0).drop_duplicates()

    #vitals
    vitals = []
    part_names = ['temp_f', 'temp_c', 'heartrate', 'resprate', 'bpsys', 'bpdia', 'spo2']
    for i in range(len(part_names)):
        if i < 4:
            part = pd.read_csv(save_to + 'lab_preprocess_sirs_' + part_names[i] + '.csv', dtype=str, na_values='NAN')
        else: 
            part = pd.read_csv(save_to + 'feat_preprocess_vit_' + part_names[i] + '.csv', dtype=str, na_values='NAN')
        part['NAME'] = part_names[i]
        part = find_windows(pop, part).drop_duplicates()
        #take out odd values
        part = part[~pd.isnull(part['charttime_diff']) & ~pd.isnull(part['value'])]
        if part_names[i] == 'temp_f':
            part = part[(part['value'] > 70) & (part['value'] < 120)]
            part['value'] = (part['value'] - 32) * (5/9)
        elif part_names[i] == 'temp_c':
            part = part[(part['value'] > 10) & (part['value'] < 50)]
        elif part_names[i] == 'heartrate':
            part = part[(part['value'] > 0) & (part['value'] < 300)]
        elif part_names[i] == 'resprate':
            part = part[(part['value'] > 0) & (part['value'] < 70)]
        elif part_names[i] == 'bpsys':
            part = part[(part['value'] > 0) & (part['value'] < 400)]
        elif part_names[i] == 'bpdia':
            part = part[(part['value'] > 0) & (part['value'] < 300)]
        elif part_names[i] == 'spo2':
            part = part[(part['value'] > 0) & (part['value'] <= 100)]
        vitals_i = pd.DataFrame({'ID': part['hadm_id'], 't': part['charttime_diff'], 'variable_value': part['value']})
        vitals_i['variable_name'] = part_names[i]
        vitals.append(vitals_i)
    vitals = pd.concat(vitals, axis=0).drop_duplicates()

    #gcs, creatinine, mean bp
    sofa_feats = []
    part_names = ['gcs_motor', 'gcs_eyes', 'gcs_verbal', 'creat', 'mean_bp']
    for i in range(len(part_names)):
        part = pd.read_csv(save_to + 'lab_preprocess_sofa_' + part_names[i] + '.csv', dtype=str, na_values='NAN').drop_duplicates()
        part['NAME'] = part_names[i]
        part = find_windows(pop, part)
        #take out odd values
        part = part[~pd.isnull(part['charttime_diff']) & ~pd.isnull(part['value'])]
        if part_names[i] == 'gcs_motor':
            part = part[(part['value'] > 2) & (part['value'] < 16)]
        elif part_names[i] == 'gcs_eyes':
            part = part[(part['value'] > 2) & (part['value'] < 16)]
        elif part_names[i] == 'gcs_verbal':
            part = part[(part['value'] > 2) & (part['value'] < 16)]
        elif part_names[i] == 'mean_bp':
            part = part[(part['value'] > 0) & (part['value'] < 300)]
        elif part_names[i] == 'creat':
            part = part[(part['value'] > 0) & (part['value'] < 150)]

        sofa_i = pd.DataFrame({'ID': part['hadm_id'], 't': part['charttime_diff'], 'variable_value': part['value']})
        sofa_i['variable_name'] = part_names[i]
        sofa_feats.append(sofa_i)
    sofa_feats = pd.concat(sofa_feats, axis=0).drop_duplicates()

    print(demographics, vitals, sofa_feats)
    #put everything together
    feats = pd.concat([demographics, vitals, sofa_feats], axis=0).drop_duplicates(subset=['ID', 't', 'variable_name'])
    feats.to_csv(save_to + 'feats_for_fiddle.csv')
    feats[['ID']].drop_duplicates().to_csv(save_to + 'ids_for_fiddle.csv')
    print('done formatting features for fiddle', feats.shape)

    return feats


#######################################################
'''
main block
'''
if __name__ == '__main__':
    '''part 1: labels'''

    #get data from mimic files
    antibiotics_raw = get_antibiotics(med_file)
    cultures_raw = get_cultures(micro_file)
    sirs_parts = get_sirs_parts(chart_file, lab_file)
    sofa_parts = get_sofa_parts(chart_file, lab_file, inpm_file, inpc_file, out_file)

    #process data to get antibiotics and scores
    antibiotics = window_antibiotics(adm_file)
    cultures = window_cultures(adm_file)
    sirs_scores = combine_sirs_parts(adm_file)
    sofa_scores = combine_sofa_parts(adm_file)
 
    #combine scores to get labels
    sepsis1_times = find_sepsis1_onset(adm_file, antibiotics, cultures)
    sepsis1_cms_times = find_sepsis1_cms_onset(adm_file, antibiotics, cultures)
    sepsis3_times = find_sepsis3_onset(adm_file, antibiotics, cultures)
    sepsis3_rhee_times = find_sepsis3_rhee_onset(adm_file, antibiotics, cultures)
    sepsisc_times = find_composite_sepsis_onset()
    sepsisc_times = find_composite_sepsis_onset(original=False)

    '''part 2: population'''

    pop = get_pop(adm_file, icu_file, pat_file, save_to + 'labels_composite.csv')

    '''part 3: features - prepare for fiddle'''

    get_vitals(chart_file)
    feats = get_feats(pop, diag_file)
