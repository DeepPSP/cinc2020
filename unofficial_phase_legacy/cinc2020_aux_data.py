# -*- coding: utf-8 -*-
"""
auxiliary data for the PhysioNet Challenge 2020

References:
-----------
[1] https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv (removed)
[2] https://github.com/physionetchallenges/evaluation-2020/blob/master/dx_mapping_scored.csv
[3] https://github.com/physionetchallenges/evaluation-2020/blob/master/dx_mapping_unscored.csv
[4] https://github.com/physionetchallenges/evaluation-2020/blob/master/weights.csv
"""
from io import StringIO

import pandas as pd


__all__ = [
    "Dx_map",
    "dx_mapping_scored_cinc2020", "dx_mapping_unscored_cinc2020",
    "scoring_matrix_cinc2020",
]


# NOTE that 'SNR' is the 'Normal' rhythm
Dx_map = pd.read_csv(StringIO("""dx,SNOMED code,Abbreviation
1st degree av block,270492004,IAVB
2nd degree av block,195042002,IIAVB
accelerated idioventricular rhythm,61277005,AIVR
accelerated junctional rhythm,426664006,AJR
acute myocardial ischemia,413444003,AMIs
anterior ischemia,426434006,AnMIs
anterior myocardial infarction,54329005,AnMI
atrial fibrillation,164889003,AF
atrial fibrillation and flutter,195080001,AFAFL
atrial flutter,164890007,AFL
atrial hypertrophy,195126007,AH
atrial pacing pattern,251268003,AP
atrial tachycardia,713422000,ATach
av block,233917008,AVB
blocked premature atrial contraction,251170000,BPAC
brady tachy syndrome,74615001,BTS
bradycardia,426627000,Brady
brugada syndrome,418818005,Brug
bundle branch block,6374002,BBB
chronic atrial fibrillation,426749004,CAF
chronic myocardial ischemia,413844008,CMIs
chronic rheumatic pericarditis,78069008,CRPC
complete heart block,27885002,CHB
complete right bundle branch block,713427006,CRBBB
decreased qt interval,77867006,SQT
diffuse intraventricular block,82226007,DIVB
early repolarization,428417006,ERe
ecg artefacts,251143007,ART
ectopic rhythm,29320008,ER
electrical alternans,423863005,EA
high t voltage,251259000,HTV
incomplete left bundle branch block,251120003,ILBBB
incomplete right bundle branch block,713426002,IRBBB
indeterminate cardiac axis,251200008,ICA
inferior ischaemia,425419005,IIs
inferior st segment depression,704997005,ISTD
isorhythmic dissociation,50799005,IAVD
junctional escape,426995002,JE
junctional premature complex,251164006,JPC
junctional tachycardia,426648003,JTach
lateral ischaemia,425623009,LIs
left anterior fascicular block,445118002,LAnFB
left atrial abnormality,253352002,LAA
left atrial enlargement,67741000119109,LAE
left atrial hypertrophy,446813000,LAH
left axis deviation,39732003,LAD
left bundle branch block,164909002,LBBB
left posterior fascicular block,445211001,LPFB
left ventricular hypertrophy,164873001,LVH
left ventricular strain,370365005,LVS
low qrs voltages,251146004,LQRSV
low qrs voltages in the limb leads,251147008,LQRSVLL
low qrs voltages in the precordial leads,251148003,LQRSP
mobitz type 2 second degree atrioventricular block,28189009,MoII
mobitz type i wenckebach atrioventricular block,54016002,MoI
multifocal atrial tachycardia,713423005,MATach
myocardial infarction,164865005,MI
myocardial ischemia,164861001,MIs
non-specific interatrial conduction block,65778007,NSIACB
nonspecific intraventricular conduction disorder,698252002,NSIVCB
nonspecific st t abnormality,428750005,NSSTTA
old myocardial infarction,164867002,OldMI
pacing rhythm,10370003,PR
paroxysmal supraventricular tachycardia,67198005,PSVT
partial atrioventricular block 2:1,164903001,PAVB21
premature atrial contraction,284470004,PAC
premature ventricular complexes,164884008,PVC
premature ventricular contractions,427172004,PVC
prolonged qt interval,111975006,LQT
qwave abnormal,164917005,QAb
r wave abnormal,164921003,RAb
right atrial abnormality,253339007,RAAb
right atrial hypertrophy,446358003,RAH
right axis deviation,47665007,RAD
right bundle branch block,59118001,RBBB
right ventricular hypertrophy,89792004,RVH
s t changes,55930002,STC
shortened pr interval,49578007,SPRI
sinus arrhythmia,427393009,SA
sinus bradycardia,426177001,SB
sinus rhythm,426783006,SNR
sinus tachycardia,427084000,STach
st depression,429622005,STD
st elevation,164931005,STE
st interval abnormal,164930006,STIAb
supraventricular premature beats,63593006,SVPB
supraventricular tachycardia,426761007,SVT
suspect arm ecg leads reversed,251139008,ALR
t wave abnormal,164934002,TAb
t wave inversion,59931005,TInv
tall u wave,251242005,UTall
u wave abnormal,164937009,UAb
ventricular bigeminy,11157007,VBig
ventricular ectopic beats,17338001,VEB
ventricular escape beat,75532003,VEsB
ventricular escape rhythm,81898007,VEsR
ventricular fibrillation,164896001,VF
ventricular flutter,111288001,VFL
ventricular hypertrophy,266249003,VH
ventricular pacing pattern,251266004,VPP
ventricular pre excitation,195060002,VPEx
ventricular tachycardia,164895002,VTach
ventricular trigeminy,251180001,VTrig
wandering atrial pacemaker,195101003,WAP
wolff parkinson white pattern,74390002,WPW"""))

dx_mapping_scored_cinc2020 = pd.read_csv(StringIO("""Dx,SNOMED CT Code,Abbreviation,CPSC,CPSC-Extra,StPetersburg,PTB,PTB-XL,Georgia,Total,Notes
1st degree av block,270492004,IAVB,722,106,0,0,797,769,2394,
atrial fibrillation,164889003,AF,1221,153,2,15,1514,570,3475,
atrial flutter,164890007,AFL,0,54,0,1,73,186,314,
bradycardia,426627000,Brady,0,271,11,0,0,6,288,
complete right bundle branch block,713427006,CRBBB,0,113,0,0,542,28,683,We score 713427006 and 59118001 as the same diagnosis.
incomplete right bundle branch block,713426002,IRBBB,0,86,0,0,1118,407,1611,
left anterior fascicular block,445118002,LAnFB,0,0,0,0,1626,180,1806,
left axis deviation,39732003,LAD,0,0,0,0,5146,940,6086,
left bundle branch block,164909002,LBBB,236,38,0,0,536,231,1041,
low qrs voltages,251146004,LQRSV,0,0,0,0,182,374,556,
nonspecific intraventricular conduction disorder,698252002,NSIVCB,0,4,1,0,789,203,997,
pacing rhythm,10370003,PR,0,3,0,0,296,0,299,
premature atrial contraction,284470004,PAC,616,73,3,0,398,639,1729,We score 284470004 and 63593006 as the same diagnosis.
premature ventricular contractions,427172004,PVC,0,188,0,0,0,0,188,We score 427172004 and 17338001 as the same diagnosis.
prolonged pr interval,164947007,LPR,0,0,0,0,340,0,340,
prolonged qt interval,111975006,LQT,0,4,0,0,118,1391,1513,
qwave abnormal,164917005,QAb,0,1,0,0,548,464,1013,
right axis deviation,47665007,RAD,0,1,0,0,343,83,427,
right bundle branch block,59118001,RBBB,1857,1,2,0,0,542,2402,We score 713427006 and 59118001 as the same diagnosis.
sinus arrhythmia,427393009,SA,0,11,2,0,772,455,1240,
sinus bradycardia,426177001,SB,0,45,0,0,637,1677,2359,
sinus rhythm,426783006,SNR,918,4,0,80,18092,1752,20846,
sinus tachycardia,427084000,STach,0,303,11,1,826,1261,2402,
supraventricular premature beats,63593006,SVPB,0,53,4,0,157,1,215,We score 284470004 and 63593006 as the same diagnosis.
t wave abnormal,164934002,TAb,0,22,0,0,2345,2306,4673,
t wave inversion,59931005,TInv,0,5,1,0,294,812,1112,
ventricular premature beats,17338001,VPB,0,8,0,0,0,357,365,We score 427172004 and 17338001 as the same diagnosis."""))

dx_mapping_unscored_cinc2020 = pd.read_csv(StringIO("""Dx,SNOMED CT Code,Abbreviation,CPSC,CPSC-Extra,StPetersburg,PTB,PTB-XL,Georgia,Total
2nd degree av block,195042002,IIAVB,0,21,0,0,14,23,58
abnormal QRS,164951009,abQRS,0,0,0,0,3389,0,3389
accelerated junctional rhythm,426664006,AJR,0,0,0,0,0,19,19
acute myocardial infarction,57054005,AMI,0,0,6,0,0,0,6
acute myocardial ischemia,413444003,AMIs,0,1,0,0,0,1,2
anterior ischemia,426434006,AnMIs,0,0,0,0,44,281,325
anterior myocardial infarction,54329005,AnMI,0,62,0,0,354,0,416
atrial bigeminy,251173003,AB,0,0,3,0,0,0,3
atrial fibrillation and flutter,195080001,AFAFL,0,39,0,0,0,2,41
atrial hypertrophy,195126007,AH,0,2,0,0,0,60,62
atrial pacing pattern,251268003,AP,0,0,0,0,0,52,52
atrial tachycardia,713422000,ATach,0,15,0,0,0,28,43
atrioventricular junctional rhythm,29320008,AVJR,0,6,0,0,0,0,6
av block,233917008,AVB,0,5,0,0,0,74,79
blocked premature atrial contraction,251170000,BPAC,0,2,3,0,0,0,5
brady tachy syndrome,74615001,BTS,0,1,1,0,0,0,2
bundle branch block,6374002,BBB,0,0,1,20,0,116,137
cardiac dysrhythmia,698247007,CD,0,0,0,16,0,0,16
chronic atrial fibrillation,426749004,CAF,0,1,0,0,0,0,1
chronic myocardial ischemia,413844008,CMI,0,161,0,0,0,0,161
complete heart block,27885002,CHB,0,27,0,0,16,8,51
congenital incomplete atrioventricular heart block,204384007,CIAHB,0,0,0,2,0,0,2
coronary heart disease,53741008,CHD,0,0,16,21,0,0,37
decreased qt interval,77867006,SQT,0,1,0,0,0,0,1
diffuse intraventricular block,82226007,DIB,0,1,0,0,0,0,1
early repolarization,428417006,ERe,0,0,0,0,0,140,140
fusion beats,13640000,FB,0,0,7,0,0,0,7
heart failure,84114007,HF,0,0,0,7,0,0,7
heart valve disorder,368009,HVD,0,0,0,6,0,0,6
high t-voltage,251259000,HTV,0,1,0,0,0,0,1
idioventricular rhythm,49260003,IR,0,0,2,0,0,0,2
incomplete left bundle branch block,251120003,ILBBB,0,42,0,0,77,86,205
indeterminate cardiac axis,251200008,ICA,0,0,0,0,156,0,156
inferior ischaemia,425419005,IIs,0,0,0,0,219,451,670
inferior ST segment depression,704997005,ISTD,0,1,0,0,0,0,1
junctional escape,426995002,JE,0,4,0,0,0,5,9
junctional premature complex,251164006,JPC,0,2,0,0,0,0,2
junctional tachycardia,426648003,JTach,0,2,0,0,0,4,6
lateral ischaemia,425623009,LIs,0,0,0,0,142,903,1045
left atrial abnormality,253352002,LAA,0,0,0,0,0,72,72
left atrial enlargement,67741000119109,LAE,0,1,0,0,427,870,1298
left atrial hypertrophy,446813000,LAH,0,40,0,0,0,0,40
left posterior fascicular block,445211001,LPFB,0,0,0,0,177,25,202
left ventricular hypertrophy,164873001,LVH,0,158,10,0,2359,1232,3759
left ventricular strain,370365005,LVS,0,1,0,0,0,0,1
mobitz type i wenckebach atrioventricular block,54016002,MoI,0,0,3,0,0,0,3
myocardial infarction,164865005,MI,0,376,9,368,5261,7,6021
myocardial ischemia,164861001,MIs,0,384,0,0,2175,0,2559
nonspecific st t abnormality,428750005,NSSTTA,0,1290,0,0,381,1883,3554
old myocardial infarction,164867002,OldMI,0,1168,0,0,0,0,1168
paired ventricular premature complexes,251182009,VPVC,0,0,23,0,0,0,23
paroxysmal atrial fibrillation,282825002,PAF,0,0,1,1,0,0,2
paroxysmal supraventricular tachycardia,67198005,PSVT,0,0,3,0,24,0,27
paroxysmal ventricular tachycardia,425856008,PVT,0,0,15,0,0,0,15
r wave abnormal,164921003,RAb,0,1,0,0,0,10,11
rapid atrial fibrillation,314208002,RAF,0,0,0,2,0,0,2
right atrial abnormality,253339007,RAAb,0,0,0,0,0,14,14
right atrial hypertrophy,446358003,RAH,0,18,0,0,99,0,117
right ventricular hypertrophy,89792004,RVH,0,20,0,0,126,86,232
s t changes,55930002,STC,0,1,0,0,770,6,777
shortened pr interval,49578007,SPRI,0,3,0,0,0,2,5
sinoatrial block,65778007,SAB,0,9,0,0,0,0,9
sinus node dysfunction,60423000,SND,0,0,2,0,0,0,2
st depression,429622005,STD,869,57,4,0,1009,38,1977
st elevation,164931005,STE,220,66,4,0,28,134,452
st interval abnormal,164930006,STIAb,0,481,2,0,0,992,1475
supraventricular bigeminy,251168009,SVB,0,0,1,0,0,0,1
supraventricular tachycardia,426761007,SVT,0,3,1,0,27,32,63
suspect arm ecg leads reversed,251139008,ALR,0,0,0,0,0,12,12
transient ischemic attack,266257000,TIA,0,0,7,0,0,0,7
u wave abnormal,164937009,UAb,0,1,0,0,0,0,1
ventricular bigeminy,11157007,VBig,0,5,9,0,82,2,98
ventricular ectopics,164884008,VEB,700,0,49,0,1154,41,1944
ventricular escape beat,75532003,VEsB,0,3,1,0,0,0,4
ventricular escape rhythm,81898007,VEsR,0,1,0,0,0,1,2
ventricular fibrillation,164896001,VF,0,10,0,25,0,3,38
ventricular flutter,111288001,VFL,0,1,0,0,0,0,1
ventricular hypertrophy,266249003,VH,0,5,0,13,30,71,119
ventricular pacing pattern,251266004,VPP,0,0,0,0,0,46,46
ventricular pre excitation,195060002,VPEx,0,6,0,0,0,2,8
ventricular tachycardia,164895002,VTach,0,1,1,10,0,0,12
ventricular trigeminy,251180001,VTrig,0,4,4,0,20,1,29
wandering atrial pacemaker,195101003,WAP,0,0,0,0,0,7,7
wolff parkinson white pattern,74390002,WPW,0,0,4,2,80,2,88"""))

weights_cinc2020 = pd.read_csv(StringIO(""",270492004,164889003,164890007,426627000,713427006,713426002,445118002,39732003,164909002,251146004,698252002,10370003,284470004,427172004,164947007,111975006,164917005,47665007,59118001,427393009,426177001,426783006,427084000,63593006,164934002,59931005,17338001
270492004,1.0,0.3,0.3,0.5,0.4,0.5,0.45,0.45,0.325,0.375,0.45,0.425,0.4625,0.425,0.5,0.35,0.2,0.45,0.4,0.5,0.5,0.45,0.425,0.4625,0.3,0.3,0.425
164889003,0.3,1.0,0.5,0.3,0.4,0.3,0.35,0.35,0.475,0.425,0.35,0.375,0.3375,0.375,0.3,0.45,0.4,0.35,0.4,0.3,0.3,0.25,0.375,0.3375,0.5,0.5,0.375
164890007,0.3,0.5,1.0,0.3,0.4,0.3,0.35,0.35,0.475,0.425,0.35,0.375,0.3375,0.375,0.3,0.45,0.4,0.35,0.4,0.3,0.3,0.25,0.375,0.3375,0.5,0.5,0.375
426627000,0.5,0.3,0.3,1.0,0.4,0.5,0.45,0.45,0.325,0.375,0.45,0.425,0.4625,0.425,0.5,0.35,0.2,0.45,0.4,0.5,0.5,0.45,0.425,0.4625,0.3,0.3,0.425
713427006,0.4,0.4,0.4,0.4,1.0,0.4,0.45,0.45,0.425,0.475,0.45,0.475,0.4375,0.475,0.4,0.45,0.3,0.45,1.0,0.4,0.4,0.35,0.475,0.4375,0.4,0.4,0.475
713426002,0.5,0.3,0.3,0.5,0.4,1.0,0.45,0.45,0.325,0.375,0.45,0.425,0.4625,0.425,0.5,0.35,0.2,0.45,0.4,0.5,0.5,0.45,0.425,0.4625,0.3,0.3,0.425
445118002,0.45,0.35,0.35,0.45,0.45,0.45,1.0,0.5,0.375,0.425,0.5,0.475,0.4875,0.475,0.45,0.4,0.25,0.5,0.45,0.45,0.45,0.4,0.475,0.4875,0.35,0.35,0.475
39732003,0.45,0.35,0.35,0.45,0.45,0.45,0.5,1.0,0.375,0.425,0.5,0.475,0.4875,0.475,0.45,0.4,0.25,0.5,0.45,0.45,0.45,0.4,0.475,0.4875,0.35,0.35,0.475
164909002,0.325,0.475,0.475,0.325,0.425,0.325,0.375,0.375,1.0,0.45,0.375,0.4,0.3625,0.4,0.325,0.475,0.375,0.375,0.425,0.325,0.325,0.275,0.4,0.3625,0.475,0.475,0.4
251146004,0.375,0.425,0.425,0.375,0.475,0.375,0.425,0.425,0.45,1.0,0.425,0.45,0.4125,0.45,0.375,0.475,0.325,0.425,0.475,0.375,0.375,0.325,0.45,0.4125,0.425,0.425,0.45
698252002,0.45,0.35,0.35,0.45,0.45,0.45,0.5,0.5,0.375,0.425,1.0,0.475,0.4875,0.475,0.45,0.4,0.25,0.5,0.45,0.45,0.45,0.4,0.475,0.4875,0.35,0.35,0.475
10370003,0.425,0.375,0.375,0.425,0.475,0.425,0.475,0.475,0.4,0.45,0.475,1.0,0.4625,0.5,0.425,0.425,0.275,0.475,0.475,0.425,0.425,0.375,0.5,0.4625,0.375,0.375,0.5
284470004,0.4625,0.3375,0.3375,0.4625,0.4375,0.4625,0.4875,0.4875,0.3625,0.4125,0.4875,0.4625,1.0,0.4625,0.4625,0.3875,0.2375,0.4875,0.4375,0.4625,0.4625,0.4125,0.4625,1.0,0.3375,0.3375,0.4625
427172004,0.425,0.375,0.375,0.425,0.475,0.425,0.475,0.475,0.4,0.45,0.475,0.5,0.4625,1.0,0.425,0.425,0.275,0.475,0.475,0.425,0.425,0.375,0.5,0.4625,0.375,0.375,1.0
164947007,0.5,0.3,0.3,0.5,0.4,0.5,0.45,0.45,0.325,0.375,0.45,0.425,0.4625,0.425,1.0,0.35,0.2,0.45,0.4,0.5,0.5,0.45,0.425,0.4625,0.3,0.3,0.425
111975006,0.35,0.45,0.45,0.35,0.45,0.35,0.4,0.4,0.475,0.475,0.4,0.425,0.3875,0.425,0.35,1.0,0.35,0.4,0.45,0.35,0.35,0.3,0.425,0.3875,0.45,0.45,0.425
164917005,0.2,0.4,0.4,0.2,0.3,0.2,0.25,0.25,0.375,0.325,0.25,0.275,0.2375,0.275,0.2,0.35,1.0,0.25,0.3,0.2,0.2,0.15,0.275,0.2375,0.4,0.4,0.275
47665007,0.45,0.35,0.35,0.45,0.45,0.45,0.5,0.5,0.375,0.425,0.5,0.475,0.4875,0.475,0.45,0.4,0.25,1.0,0.45,0.45,0.45,0.4,0.475,0.4875,0.35,0.35,0.475
59118001,0.4,0.4,0.4,0.4,1.0,0.4,0.45,0.45,0.425,0.475,0.45,0.475,0.4375,0.475,0.4,0.45,0.3,0.45,1.0,0.4,0.4,0.35,0.475,0.4375,0.4,0.4,0.475
427393009,0.5,0.3,0.3,0.5,0.4,0.5,0.45,0.45,0.325,0.375,0.45,0.425,0.4625,0.425,0.5,0.35,0.2,0.45,0.4,1.0,0.5,0.45,0.425,0.4625,0.3,0.3,0.425
426177001,0.5,0.3,0.3,0.5,0.4,0.5,0.45,0.45,0.325,0.375,0.45,0.425,0.4625,0.425,0.5,0.35,0.2,0.45,0.4,0.5,1.0,0.45,0.425,0.4625,0.3,0.3,0.425
426783006,0.45,0.25,0.25,0.45,0.35,0.45,0.4,0.4,0.275,0.325,0.4,0.375,0.4125,0.375,0.45,0.3,0.15,0.4,0.35,0.45,0.45,1.0,0.375,0.4125,0.25,0.25,0.375
427084000,0.425,0.375,0.375,0.425,0.475,0.425,0.475,0.475,0.4,0.45,0.475,0.5,0.4625,0.5,0.425,0.425,0.275,0.475,0.475,0.425,0.425,0.375,1.0,0.4625,0.375,0.375,0.5
63593006,0.4625,0.3375,0.3375,0.4625,0.4375,0.4625,0.4875,0.4875,0.3625,0.4125,0.4875,0.4625,1.0,0.4625,0.4625,0.3875,0.2375,0.4875,0.4375,0.4625,0.4625,0.4125,0.4625,1.0,0.3375,0.3375,0.4625
164934002,0.3,0.5,0.5,0.3,0.4,0.3,0.35,0.35,0.475,0.425,0.35,0.375,0.3375,0.375,0.3,0.45,0.4,0.35,0.4,0.3,0.3,0.25,0.375,0.3375,1.0,0.5,0.375
59931005,0.3,0.5,0.5,0.3,0.4,0.3,0.35,0.35,0.475,0.425,0.35,0.375,0.3375,0.375,0.3,0.45,0.4,0.35,0.4,0.3,0.3,0.25,0.375,0.3375,0.5,1.0,0.375
17338001,0.425,0.375,0.375,0.425,0.475,0.425,0.475,0.475,0.4,0.45,0.475,0.5,0.4625,1.0,0.425,0.425,0.275,0.475,0.475,0.425,0.425,0.375,0.5,0.4625,0.375,0.375,1.0"""))

# alias
scoring_matrix_cinc2020 = weights_cinc2020
