# -*- coding: utf-8 -*-
"""
knowledge about ECG arrhythmia, and corresponding Dx maps

Standard_12Leads_ECG:
---------------------
    Inferior leads: II, III, aVF
    Lateral leads: I, aVL, V5-6
    Septal leads: V1, aVR
    Anterior leads: V2-4
    -----------------------------------
    Chest (precordial) leads: V1-6
    Limb leads: I, II, III, aVF, aVR, aVL
"""
from easydict import EasyDict as ED


__all__ = [
    "AF", "AFL",  # atrial
    "IAVB", "LBBB", "RBBB", "CRBBB", "IRBBB", "LAnFB", "NSIVCB",  # conduction block
    "PAC", "PJC", "PVC", "SPB",  # premature: qrs, morphology
    "LPR", "LQT", "QAb", "TAb", "TInv",  # wave morphology
    "LAD", "RAD",  # axis
    "Brady", "LQRSV",  # qrs (RR interval, amplitude)
    "SA", "SB", "SNR", "STach",  # sinus
    "PR",  # pacer
    "STD", "STE",  # ST segments
]


AF = ED({  # rr, morphology
    "fullname": "atrial fibrillation",
    "url": [
        "https://litfl.com/atrial-fibrillation-ecg-library/",
        "https://en.wikipedia.org/wiki/Atrial_fibrillation#Screening",
    ],
    "knowledge": [
        "irregularly irregular rhythm",
        "no P waves",
        "absence of an isoelectric baseline",
        "variable ventricular rate",
        "QRS complexes usually < 120 ms unless pre-existing bundle branch block, accessory pathway, or rate related aberrant conduction",
        "fibrillatory waves (f-wave) may be present and can be either fine (amplitude < 0.5mm) or coarse (amplitude >0.5mm)",
        "fibrillatory waves (f-wave) may mimic P waves leading to misdiagnosis",
    ],
})

AFL = ED({  # rr, morphology
    "fullname": "atrial flutter",
    "url": [
        "https://litfl.com/atrial-flutter-ecg-library/",
        "https://en.wikipedia.org/wiki/Atrial_flutter",
    ],
    "knowledge": [
        "a type of supraventricular tachycardia caused by a re-entry circuit within the right atrium",
        "fairly predictable atrial rate (NOT equal to ventricular rate for AFL) of around 300 bpm (range 200-400)",
        "fixed AV blocks, with ventricular rate a fraction (1/2,1/3,etc.) of atrial rate",
        "narrow complex tachycardia (ref. supraventricular & ventricular rate)",
        "flutter waves ('saw-tooth' pattern) best seen in leads II, III, aVF (may be more easily spotted by turning the ECG upside down), may resemble P waves in V1",
        "loss of the isoelectric baseline",  # important
    ],
})

Brady = ED({  # rr
    "fullname": "bradycardia",
    "url": [
        "https://litfl.com/bradycardia-ddx/",
        "https://en.wikipedia.org/wiki/Bradycardia"
    ],
    "knowledge": [
        "heart rate <60/min in an adult",
    ],
})

IAVB = {  # morphology
    "fullname": "1st degree av block",
    "url": [
        "https://litfl.com/first-degree-heart-block-ecg-library/",
        "https://en.wikipedia.org/wiki/Atrioventricular_block#First-degree_Atrioventricular_Block"
    ],
    "knowledge": [
        "PR interval > 200ms",
        "Marked’ first degree block if PR interval > 300ms",
        "P waves might be buried in the preceding T wave",
        "there are no dropped, or skipped, beats",
    ],
}

LBBB = ED({  # morphology
    "fullname": "left bundle branch block",
    "url": [
        "https://litfl.com/left-bundle-branch-block-lbbb-ecg-library/",
        "https://en.wikipedia.org/wiki/Left_bundle_branch_block",
    ],
    "knowledge": [
        "heart rhythm must be supraventricular",
        "QRS duration of > 120 ms",
        "lead V1: Dominant S wave, with QS or rS complex",
        "lateral leads: M-shaped, or notched, or broad monophasic R wave or RS complex; absence of Q waves (small Q waves are still allowed in aVL)",
        "chest (precordial) leads: poor R wave progression",
        "left precordial leads (V5-6): prolonged R wave peak time > 60ms",
        "ST segments and T waves always go in the opposite direction to the main vector of the QRS complex",
    ],
})

RBBB = ED({  # morphology
    "fullname": "right bundle branch block",
    "url": [
        "https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/",
        "https://en.wikipedia.org/wiki/Right_bundle_branch_block",
    ],
    "knowledge": [
        "broad QRS > 100 ms (incomplete block) or > 120 ms (complete block)",
        "leads V1-3: RSR’ pattern (‘M-shaped’ QRS complex); sometimes a broad monophasic R wave or a qR complex in V1",
        "lateral leads: wide, slurred S wave",
    ],
})

CRBBB = ED({  # morphology
    "fullname": "complete right bundle branch block",
    "url": [
        "https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/",
        "https://en.wikipedia.org/wiki/Right_bundle_branch_block",
    ],
    "knowledge": [
        "broad QRS > 120 ms",
        "leads V1-3: RSR’ pattern (‘M-shaped’ QRS complex); sometimes a broad monophasic R wave or a qR complex in V1",
        "lateral leads: wide, slurred S wave",
    ],
})

IRBBB = ED({  # morphology
    "fullname": "incomplete right bundle branch block",
    "url": [
        "https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/",
        "https://en.wikipedia.org/wiki/Right_bundle_branch_block#Diagnosis",
    ],
    "knowledge": [
        "defined as an RSR’ pattern in V1-3 with QRS duration < 120ms (and > 100ms?)",
        "normal variant, commonly seen in children (of no clinical significance)",
    ],
})

LAnFB = ED({  # morphology
    "fullname": "left anterior fascicular block",
    "url": [
        "https://litfl.com/left-anterior-fascicular-block-lafb-ecg-library/",
        "https://en.wikipedia.org/wiki/Left_anterior_fascicular_block",
    ],
    "knowledge": [
        "inferior leads (II, III, aVF): small R waves, large negative voltages (deep S waves), i.e. 'rS complexes'",
        "left-sided leads (I, aVL): small Q waves, large positive voltages (tall R waves), i.e. 'qR complexes'",
        "slight widening of the QRS",
        "increased R wave peak time in aVL",
        "LAD of degree (-45°, -90°)"
    ],
})

LAD = ED({  # morphology
    "fullname": "left axis deviation",
    "url": [
        "https://litfl.com/left-axis-deviation-lad-ecg-library/",
        "https://en.wikipedia.org/wiki/Left_axis_deviation",
    ],
    "knowledge": [
        "QRS axis (-30°, -90°)",
        "leads I and aVL are positive; leads II, III and aVF are negative",  # important
        "LAnFB, LBBB, PR, ventricular ectopics are causes of LAD",
    ],
})

LQRSV = ED({  # voltage
    "fullname": "low qrs voltages",
    "url": [
        "https://litfl.com/low-qrs-voltage-ecg-library/",
    ],
    "knowledge": [
        "amplitudes of all the QRS complexes in the limb leads are < 5mm (0.5mV); or  amplitudes of all the QRS complexes in the precordial leads are < 10mm (1mV)",
    ],
})

NSIVCB = ED({
    "fullname": "nonspecific intraventricular conduction disorder",
    "url": [

    ],
    "knowledge": [

    ],
})

PR = ED({
    "fullname": "pacing rhythm",
    "url": [

    ],
    "knowledge": [

    ],
})

PAC = ED({
    "fullname": "premature atrial contraction",
    "url": [
        "https://litfl.com/premature-atrial-complex-pac/",
        "https://en.wikipedia.org/wiki/Premature_atrial_contraction",
    ],
    "knowledge": [
        "An abnormal (non-sinus) P wave is followed by a QRS complex",
        "P wave typically has a different morphology and axis to the sinus P waves",
        "Abnormal P wave may be hidden in the preceding T wave, producing a “peaked” or “camel hump” appearance",
        # to add more
    ],
})

PJC = ED({
    "fullname": "premature junctional contraction",
    "url": [
        "https://litfl.com/premature-junctional-complex-pjc/",
        "https://en.wikipedia.org/wiki/Premature_junctional_contraction",
    ],
    "knowledge": [
        "Narrow QRS complex, either (1) without a preceding P wave or (2) with a retrograde P wave which may appear before, during, or after the QRS complex. If before, there is a short PR interval of < 120 ms and the  “retrograde” P waves are usually inverted in leads II, III and aVF",
        "Occurs sooner than would be expected for the next sinus impulse",
        "Followed by a compensatory pause",
    ],
})

PVC = ED({
    "fullname": "premature ventricular contractions",
    "url": [
        "https://litfl.com/premature-ventricular-complex-pvc-ecg-library/",
        "https://en.wikipedia.org/wiki/Premature_ventricular_contraction",
    ],
    "knowledge": [
        "Broad QRS complex (≥ 120 ms) with abnormal morphology",
        "Premature — i.e. occurs earlier than would be expected for the next sinus impulse",
        "Discordant ST segment and T wave changes",
        "Usually followed by a full compensatory pause",
        "Retrograde capture of the atria may or may not occur",
    ],
})

LPR = ED({
    "fullname": "prolonged pr interval",
    "url": [

    ],
    "knowledge": [

    ],
})

LQT = ED({
    "fullname": "prolonged qt interval",
    "url": [

    ],
    "knowledge": [

    ],
})

QAb = ED({
    "fullname": "qwave abnormal",
    "url": [

    ],
    "knowledge": [

    ],
})

RAD = ED({
    "fullname": "right axis deviation",
    "url": [

    ],
    "knowledge": [

    ],
})

SA = ED({
    "fullname": "sinus arrhythmia",
    "url": [

    ],
    "knowledge": [

    ],
})

SB = ED({
    "fullname": "sinus bradycardia",
    "url": [

    ],
    "knowledge": [

    ],
})

SNR = ED({
    "fullname": "sinus rhythm",  # the NORMAL rhythm
    "url": [

    ],
    "knowledge": [

    ],
})

STach = ED({
    "fullname": "sinus tachycardia",
    "url": [

    ],
    "knowledge": [

    ],
})

SVPB = ED({
    "fullname": "supraventricular premature beats",
    "url": [
        "https://en.wikipedia.org/wiki/Premature_atrial_contraction#Supraventricular_extrasystole",
    ],
    "knowledge": PAC["knowledge"] + PJC["knowledge"],
})

TAb = ED({
    "fullname": "t wave abnormal",
    "url": [

    ],
    "knowledge": [

    ],
})

TInv = ED({
    "fullname": "t wave inversion",
    "url": [

    ],
    "knowledge": [

    ],
})

VPB = ED({
    "fullname": "ventricular premature beats",
    "url": [

    ],
    "knowledge": [

    ],
})

SPB = SVPB  # alias

STD = ED({
    "fullname": "st depression",
    "url": [
        "",
    ],
    "knowledge": [
        "",
    ],
})

STE = ED({
    "fullname": "st elevation",
    "url": [
        "",
    ],
    "knowledge": [
        "",
    ],
})
