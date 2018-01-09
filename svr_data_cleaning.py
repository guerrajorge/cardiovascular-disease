# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np
from functools import reduce

os.getcwd()
# os.chdir('C:\\Users\\Anesthesia\\Documents\\Research\\Cardiovascular\\PHN\\SVR Trial\\SVR Data')
os.chdir('SVR Data')

def stat_measure(a):
    b = np.unique(a.index)
    for val in a:
        kl[val, :] = np.mean(r300.loc[val, ['SBP', 'DBP', 'MBP']])


def load_filter_data():
    """
    Reading .csv files
    :return:
    """
    r100 = pd.read_csv('100_eligibility.csv', index_col='blind_id')
    r102 = pd.read_csv('102_prenorwood.csv', index_col='blind_id')
    r103 = pd.read_csv('103_norwood_hospitalization.csv', index_col='blind_id')
    r104 = pd.read_csv('104_stage2_hospital.csv', index_col='blind_id')
    r105 = pd.read_csv('105_post_stage2_history.csv', index_col='blind_id')
    r106 = pd.read_csv('106_clinic_genetic_evaluation.csv', index_col='blind_id')
    r107 = pd.read_csv('107_cardiac_catheterization_measure.csv', index_col='blind_id')
    # r108= pd.read_csv('108_primary_outcome.csv', index_col='blind_id')
    r109 = pd.read_csv('109_obser_patient_medical_history.csv', index_col='blind_id')
    r113 = pd.read_csv('113_functional_status_ii.csv', index_col='blind_id')
    r114 = pd.read_csv('114_bayley_scoring_summary.csv', index_col='blind_id')
    r144 = pd.read_csv('144_socioeconomic_status.csv', index_col='blind_id')
    r206 = pd.read_csv('206_between_visit_complicate.csv', index_col='blind_id')
    r300 = pd.read_csv('300_clinical_assess_time_echo.csv', index_col='blind_id')
    r301 = pd.read_csv('301_echocardiography_core_lab_2d.csv', index_col='blind_id')
    r302 = pd.read_csv('302_echocardiography_core_lab_3d.csv', index_col='blind_id')
    r331 = pd.read_csv('331_apo_e_core_lab.csv', index_col='blind_id')
    r340 = pd.read_csv('340_angio_core_lab.csv', index_col='blind_id')
    rkey = pd.read_csv('keyinfo.csv', index_col='blind_id')

    r100 = r100.drop(['screen_age', 'COMP_BY', 'DOB', 'RACEMR_S', 'RACEOT_S', 'HISPANIC',
                      'BWT', 'PNDXAGE', 'AGEDXHR', 'AGEDXDAY', 'ANATDX1', 'ANATDX2', 'ANATDX3',
                      'ANATDX4', 'ANDX_S', 'asso_adx', 'ASDXCODE_0', 'ASDXCODE_1', 'ASDXCODE_2',
                      'ASDXCODE_3', 'ASDXCODE_4', 'ASDX_S_0', 'ASDX_S_1', 'ASDX_S_2', 'ASDX_S_3',
                      'ASDX_S_4', 'NORWPLAN', 'LFVANOM', 'CONGOTH', 'CONOTH_S', 'CORONARY',
                      'NOSHNOTH', 'NOSHNT_S', 'DIED', 'died_age', 'OCENTFAR', 'REASON_PT_NA',
                      'REASON_PT_LB', 'REASON_PT_PO', 'REASON_PT_O', 'REASON_PT_OS',
                      'comp_age', 'ELIGIBLE', 'CONSENT', 'main549cohort', 'consent_age',
                      'NOCNST', 'NOCNST_S', 'NOFAM', 'NOFAM_S'], axis=1)

    r102 = r102.drop(['r102_age', 'main549cohort', 'LACTATE', 'LACTMETH', 'INTUBAT', 'GASCO2',
                      'GASN2', 'precath', 'pre_cath_0', 'bcath_age_0', 'BCATHNAM_0', 'BCOMP_age_20',
                      'BCOMPCOD_21', 'BCOMP_S_21', 'BCOMP_age_21', 'BCOMPCOD_22', 'BCOMP_S_22',
                      'BCOMP_age_22', 'BCOMPCOD_23', 'BCOMP_S_23', 'BCOMP_age_23',
                      'pre_cath_1', 'bcath_age_1', 'BCATHNAM_1', 'pre_cath_2', 'bcath_age_2',
                      'BCATHNAM_2', 'BSURGCOD_0', 'BSURG_S_0', 'BSURG_age_0', 'CPB_EV_0',
                      'CPB_SPC_0', 'BCOMPCOD_0', 'BCOMP_S_0', 'BCOMP_age_0', 'BCOMPCOD_1',
                      'BCOMP_S_1', 'BCOMP_age_1', 'BCOMPCOD_2', 'BCOMP_S_2', 'BCOMP_age_2',
                      'BCOMPCOD_3', 'BCOMP_S_3', 'BCOMP_age_3', 'BCOMPCOD_4', 'BCOMP_S_4',
                      'BCOMP_age_4', 'BCOMPCOD_5',
                      'BCOMP_S_5', 'BCOMP_age_5', 'BCOMPCOD_6', 'BCOMP_S_6', 'BCOMP_age_6',
                      'BCOMPCOD_7', 'BCOMP_S_7', 'BCOMP_age_7', 'BCOMPCOD_8', 'BCOMP_S_8',
                      'BCOMP_age_8', 'BCOMPCOD_9', 'BCOMP_S_9', 'BCOMP_age_9', 'BCOMPCOD_10',
                      'BCOMP_S_10', 'BCOMP_age_10', 'BCOMPCOD_11', 'BCOMP_S_11',
                      'BCOMP_age_11', 'BCOMPCOD_12', 'BCOMP_S_12', 'BCOMP_age_12',
                      'BCOMPCOD_13', 'BCOMP_S_13', 'BCOMP_age_13', 'BCOMPCOD_14',
                      'BCOMP_S_14', 'BCOMP_age_14', 'BCOMPCOD_15', 'BCOMP_S_15',
                      'BCOMP_age_15', 'BCOMPCOD_16', 'BCOMP_S_16', 'BCOMP_age_16',
                      'BCOMPCOD_17', 'BCOMP_S_17', 'BCOMP_age_17', 'BCOMPCOD_18',
                      'BCOMP_S_18', 'BCOMP_age_18', 'BCOMPCOD_19', 'BCOMP_S_19',
                      'BCOMP_age_19', 'BCOMPCOD_20', 'BCOMP_S_20'], axis=1)

    r103 = r103.drop(['VISIT', 'r103_age', 'NMEDNAME_19', 'NMEDCAT_19', 'medcode_20', 'NMEDNAME_20', 'NMEDCAT_20',
                      'medcode_21', 'NMEDNAME_21', 'NMEDCAT_21', 'norw_yr', 'nordis_age', 'NORWS_T', 'NORSTU', 'NORWE_T',
                      'NMEDCAT_16', 'medcode_17', 'NMEDNAME_17', 'NMEDCAT_17', 'medcode_18',
                      'NMEDNAME_18', 'NMEDCAT_18', 'medcode_19', 'NORETU', 'medcode_14',
                      'NMEDNAME_14', 'NMEDCAT_14', 'medcode_15', 'NMEDNAME_15',
                      'NMEDCAT_15', 'medcode_16', 'NMEDNAME_16', 'ULTRAFIL', 'CONTINU',
                      'POSTCPB', 'GRAFTORG', 'scsite', 'SHUNTONE', 'SHNTMBTS',
                      'MBTSDIA', 'MBTSLEN', 'SHNTRVPA', 'RVPADIA', 'RVPALEN', 'cross_r',
                      'CROSSCOD_0', 'CROSSCOD_1', 'CROSS_S_0', 'CROSS_S_1', 'opcode',
                      'NOPCODE_0', 'NOPCODE_1', 'NOPCODE_2', 'NOP_S_0', 'NOP_S_1',
                      'NOP_S_2', 'reticu_age', 'RETICU_T', 'RETICUTU', 'disicu_age', 'NEXTUBOR',
                      'nextub_age', 'NEXTUB_T', 'NEXTUBTU',
                      'STERN_OPEN', 'nwd_cath_0', 'ncath_age_0', 'NCATHNAM_0', 'nwd_cath_1', 'ncath_age_1',
                      'NCATHNAM_1', 'nwd_cath_2', 'ncath_age_2', 'NCATHNAM_2', 'nwd_cath_3',
                      'ncath_age_3', 'NCATHNAM_3', 'nwd_cath_4', 'ncath_age_4', 'NCATHNAM_4',
                      'nwd_cath_5', 'ncath_age_5', 'NCATHNAM_5', 'nwd_cath_6', 'ncath_age_6',
                      'NCATHNAM_6', 'nwd_cath_7', 'ncath_age_7', 'NCATHNAM_7', 'nwd_cath_8',
                      'ncath_age_8', 'NCATHNAM_8', 'ncath_age_9', 'NCATHNAM_9', 'nwd_cath_10', 'ncath_age_10',
                      'NCATHNAM_10', 'nsurgcode', 'NSURGCOD_0', 'NSURG_S_0',
                      'NSURGCOD_1', 'NSURG_S_1', 'NSURGCOD_2', 'NSURG_S_2', 'NSURGCOD_3',
                      'NSURG_S_3', 'NSURGCOD_4', 'NSURG_S_4', 'NSURGCOD_5', 'NSURG_S_5',
                      'NSURGCOD_6', 'NSURG_S_6', 'NSURGCOD_7', 'NSURG_S_7', 'NSURGCOD_8',
                      'NSURG_S_8', 'NSURGCOD_9', 'NSURG_S_9', 'NSURGCOD_10', 'NSURG_S_10',
                      'NSURGCOD_11', 'NSURG_S_11', 'NSURGCOD_12', 'NSURG_S_12', 'NSURGCOD_13',
                      'NSURG_S_13', 'NSURGCOD_14', 'NSURG_S_14', 'NPACER_D',
                      'NPACERTY', 'NPACER_S', 'NCOMPCOD_0',
                      'ncomp_age_0', 'NCOMP_S_0', 'NCOMPCOD_1', 'ncomp_age_1', 'NCOMP_S_1',
                      'NCOMPCOD_2', 'ncomp_age_2', 'NCOMP_S_2', 'NCOMPCOD_3', 'ncomp_age_3',
                      'NCOMP_S_3', 'NCOMPCOD_4', 'ncomp_age_4', 'NCOMP_S_4', 'NCOMPCOD_5',
                      'ncomp_age_5', 'NCOMP_S_5', 'NCOMPCOD_6', 'ncomp_age_6', 'NCOMP_S_6',
                      'NCOMPCOD_7', 'ncomp_age_7', 'NCOMP_S_7', 'NCOMPCOD_8', 'ncomp_age_8',
                      'NCOMP_S_8', 'NCOMPCOD_9', 'ncomp_age_9', 'NCOMP_S_9', 'NCOMPCOD_10',
                      'ncomp_age_10', 'NCOMP_S_10', 'NCOMPCOD_11', 'ncomp_age_11',
                      'NCOMP_S_11', 'ncomp_age_12', 'NCOMP_S_12', 'NCOMPCOD_13', 'ncomp_age_13',
                      'NCOMP_S_13', 'NCOMPCOD_14', 'ncomp_age_14', 'NCOMP_S_14',
                      'NCOMPCOD_15', 'ncomp_age_15', 'NCOMP_S_15', 'NCOMPCOD_16',
                      'ncomp_age_16', 'NCOMP_S_16', 'NCOMPCOD_17', 'ncomp_age_17',
                      'NCOMP_S_17', 'NCOMPCOD_18', 'ncomp_age_18', 'NCOMP_S_18',
                      'NCOMPCOD_19', 'ncomp_age_19', 'NCOMP_S_19', 'NCOMPCOD_20',
                      'ncomp_age_20', 'NCOMP_S_20', 'NCOMPCOD_21', 'ncomp_age_21',
                      'NCOMP_S_21', 'NCOMPCOD_22', 'ncomp_age_22', 'NCOMP_S_22',
                      'NCOMPCOD_23', 'ncomp_age_23', 'NCOMP_S_23', 'NCOMPCOD_24',
                      'ncomp_age_24', 'NCOMP_S_24', 'NCOMPCOD_25', 'ncomp_age_25',
                      'NCOMP_S_25', 'NCOMPCOD_26', 'ncomp_age_26', 'NCOMP_S_26',
                      'NCOMPCOD_27', 'ncomp_age_27', 'NCOMP_S_27', 'NCOMPCOD_28',
                      'ncomp_age_28', 'NCOMP_S_28', 'NCOMPCOD_29', 'ncomp_age_29',
                      'NCOMP_S_29', 'NCOMPCOD_30', 'ncomp_age_30', 'NCOMP_S_30', 'NORVITAL',
                      'medcode_0', 'NMEDNAME_0', 'NMEDCAT_0',
                      'medcode_1', 'NMEDNAME_1', 'NMEDCAT_1', 'medcode_2', 'NMEDNAME_2',
                      'NMEDCAT_2', 'medcode_3', 'NMEDCAT_3', 'medcode_4', 'NMEDNAME_4', 'NMEDCAT_4', 'medcode_5',
                      'NMEDNAME_5', 'NMEDCAT_5', 'medcode_6', 'NMEDNAME_6', 'NMEDCAT_6',
                      'medcode_7', 'NMEDNAME_7', 'NMEDCAT_7', 'medcode_8', 'NMEDNAME_8',
                      'NMEDCAT_8', 'medcode_9', 'NMEDNAME_9', 'NMEDCAT_9', 'medcode_10',
                      'NMEDNAME_10', 'NMEDCAT_10', 'medcode_11', 'NMEDNAME_11', 'NMEDCAT_11',
                      'medcode_12', 'NMEDNAME_12', 'NMEDCAT_12', 'medcode_13', 'NMEDNAME_13',
                      'NMEDCAT_13'], axis=1)

    r104 = r104.dropna(axis=1, thresh=0.7 * len(r104))
    r104 = r104.drop(['VISIT', 'r104_age',
                      'PPACERTY', 'PPACER_S', 'HYPOX', 'THRIVE', 'OCCLUSN',
                      'ARCHOBST', 'AVVINSUF', 'TIMEOTH', 'TIMEOTH_S',
                      'STG2STU', 'STG2STRT', 'STG2ETU', 'STG2END',
                      'STG2COD_0', 'STG2_S_0', 'STG2_S_1', 'STG2_S_2',
                      'sconcod', 'POSTEXOR', 'postex_age', 'POSTEX_T', 'POSTEX_TU',
                      'postsurgcd',
                      'POSTPACER', 'POSTPACTYP', 'POSTPACE_S',
                      'ST2VITAL', 'medcode_0', 'medcode_1',
                      'medcode_2', 'medcode_3', 'NMEDNAME_5', 'NMEDNAME_6', 'NMEDNAME_7',
                      'NMEDNAME_8', 'NMEDNAME_9', 'NMEDNAME_10', 'NMEDNAME_11', 'NMEDNAME_12',
                      'NMEDNAME_13', 'NMEDNAME_14', 'NMEDNAME_15', 'NMEDNAME_16',
                      'NMEDNAME_17', 'NMEDNAME_18', 'SO2SAT', 'NAIRTYPE'], axis=1)

    r105 = r105.drop(['VISIT', 'r105_age', 'cath0', 'cath_age_0',
                      'CATHNAM_0', 'cath1', 'cath_age_1', 'CATHNAM_1', 'cath2', 'cath_age_2',
                      'CATHNAM_2', 'cath3', 'cath_age_3', 'CATHNAM_3', 'cath4', 'cath_age_4',
                      'CATHNAM_4', 'SURGCOD_0', 'SURG_S_0',
                      'SURGCOD_1', 'SURG_S_1', 'SURGCOD_2', 'SURG_S_2', 'SURGCOD_3',
                      'SURG_S_3', 'SURGCOD_4', 'SURG_S_4', 'SURGCOD_5', 'SURG_S_5', 'CPB',
                      'CPB_EV_0', 'CPB_MN_0', 'CPB_SPC_0', 'DHCA_MN_0', 'DHCA_YN_0',
                      'HCT_PCT_0', 'LOW_TEMP_0', 'RCPFLOW_0', 'RCP_MN_0', 'RCP_YN_0',
                      'CPB_EV_1', 'CPB_MN_1', 'CPB_SPC_1', 'DHCA_MN_1', 'DHCA_YN_1',
                      'HCT_PCT_1', 'LOW_TEMP_1', 'RCPFLOW_1', 'RCP_MN_1', 'RCP_YN_1',
                      'pacer_age', 'PACERTY', 'PACER_S', 'INTRATACH', 'SUPRATACH',
                      'ATRIALFLUT', 'ATRIALFIB', 'JUNCTACH', 'VENTTACH', 'VENTFIB',
                      'BRADYCAR', 'AVBLK2ND', 'AVBLK3RD', 'ARRHYOTH', 'ARRHYOTH_S'], axis=1)

    r106 = r106.drop(['VISIT', 'r106_age', 'geval_age', 'HT_GEVAL',
                      'WT_GEVAL', 'HC_GEVAL', 'NORM_GEVAL',
                      'KARYOYN', 'KARYOTYP', 'KARYO_S', 'SYNDRM_S',
                      'abnorcod', 'ABNORCOD_0', 'ABNORM_S_0', 'ABNORCOD_1',
                      'ABNORM_S_1', 'ABNORCOD_2', 'ABNORM_S_2', 'ABNORCOD_3', 'ABNORM_S_3',
                      'ABNORCOD_4', 'ABNORM_S_4', 'ABNORCOD_5', 'ABNORM_S_5', 'ABNORCOD_6',
                      'ABNORM_S_6', 'ABNORCOD_7', 'ABNORM_S_7', 'ABNORCOD_8', 'ABNORM_S_8',
                      ], axis=1)

    r107 = r107.drop(['VISIT', 'r107_age', 'CATH_YN', 'cath_age', 'HT_CATH', 'cath_haz_who',
                      'WT_CATH', 'cath_waz_who', 'CATH_SZ_0',
                      'CATHSZ_S_0', 'CATH_SZ_1', 'CATHSZ_S_1', 'CATH_SZ_2', 'CATHSZ_S_2',
                      'CATH_SZ_3', 'CATHSZ_S_3'], axis=1)

    r113 = r113.loc[:, ['INCOME', 'totalall', 'total', 'genhealth', 'response']]

    r114 = r114.drop(['VISIT', 'r114_age', 'ASSESS', 'ASSESSNO', 'DAYCARE', 'HOURWK',
                      'CAREFAM', 'FAM_S', 'OTHCARE', 'OTHCARE_S', 'GENDER', 'ASSESS_age',
                      'GESTAGE', 'CAGEMON', 'CAGEDY', 'MENTAL', 'MENTNO', 'MENTQSHT', 'MENTDEV',
                      'MOTOR', 'MOTQSHT', 'MOTDEV', 'ORNTRAW', 'EMOTRAW',
                      'MOTRAW', 'MOTPERC', 'ADDRAW', 'TOTRAW',
                      'DOMLANG', 'LANGUSED', 'CHLDCOOP', 'CONFID', 'RECOMM', 'RECOMM_S',
                      'COMMS114', 'NOTIFIED_age', 'REVCOMP_age', 'STATUS', 'UPDATE_YN'], axis=1)

    r144 = r144.loc[:, ['STATE', 'MED_INSUR']]

    r206 = r206.loc[:, ['BNUMCOMP', 'bcompcode', 'BCOMPCOD_0']]

    r300 = r300.drop(['VISIT', 'PHN_CENTER', 'r300_age', 'echo_age', 'HT_ECHO',
                      'ECHO_haz_who', 'WT_ECHO', 'ECHO_waz_who', 'HC_ECHO', 'ECHO_hcaz_who',
                      'DONE3D', 'QCVISIT'], axis=1)

    r301 = r301.drop(['VISIT', 'read_age', 'ACCEPTABLE', 'UNACCEPT', 'IMGQUAL', 'BASELINE'], axis=1)

    r302 = r302.drop(['VISIT', 'read_age', 'ACCEPTABLE', 'UNACCEPT', 'IMGQUAL', 'BASELINE'], axis=1)

    r331 = r331.loc[:, 'GENOTYPE']

    r340 = r340.drop(['angio_age', 'read_age', 'ACCEPTABLE', 'QUALITY', 'HT_CATH', 'WT_CATH',
                      'lpa_z', 'LPA_STENOSIS', 'LPA_ANEURYSM', 'LPA_DI_ANEUR', 'DISTRPA',
                      'RPA_STENOSIS', 'RPA_DI_STEN', 'RPA_ANEURYSM', 'RPA_DI_ANEUR',
                      'STENO_SITE', 'STENO_ST_S', 'OTH_ABN', 'OTH_ABN_S'], axis=1)

    rkey = rkey.drop(['rand_age', 'trt', 'ctrt',
                      'transplant_age', 'death_age', 'svrend_age',
                      'time_event', 'compete_ind'], axis=1)

    tab1 = r100.merge(r102, left_index=True, right_index=True, sort=True)
    tab2 = tab1.merge(r103, left_index=True, right_index=True, sort=True)
    tab3 = pd.concat([tab2, r104, r105], axis=1)
    # tab4=tab3.merge(r106,left_index=True,right_index=True,sort=True)
    tab4 = pd.concat([tab3, r107, r113], axis=1)
    # a = np.intersect1d(tab4.index, r109.index)
    # R109 has just 1 patient overlapping the main study dataset. So we will ignore r109

    tab5 = pd.concat([tab4, r114], axis=1)
    tab6 = tab5.merge(r144, left_index=True, right_index=True, sort=True)
    # a=np.intersect1d(tab6.index,r331.index)
    # r331=r331.loc[a,:]
    # aa = r331.index
    # a = aa.difference(tab6.index)
    r331 = r331.drop(455)
    tab7 = pd.concat([tab6, r331], axis=1)
    # reduce(np.union1d, (tab3.index,r107.index,r109.index,r113.index,r114.index))
    tab8 = pd.concat([tab7, r340, rkey], axis=1)

    return tab8


def main():

    dataset = load_filter_data()


if __name__ == '__main__':
    main()

