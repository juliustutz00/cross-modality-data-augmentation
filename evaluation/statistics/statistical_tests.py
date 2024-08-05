import numpy as np
from scipy.stats import friedmanchisquare, ttest_rel
import scikit_posthocs as sp



b_acc_resnet_no = [0.630231, 0.580443, 0.655755, 0.625984, 0.604331]
b_acc_resnet_imgaug = [0.676281, 0.626559, 0.636667, 0.636911, 0.641843]
b_acc_resnet_albumentations = [0.680173, 0.649562, 0.656662, 0.628771, 0.643037]
b_acc_resnet_v2 = [0.681014, 0.605636, 0.664713, 0.601876, 0.589069]
b_acc_resnet_RandAugment = [0.649474, 0.585862, 0.658409, 0.622843, 0.624547]
b_acc_resnet_cross = [0.622291, 0.649850, 0.641467, 0.637906, 0.577811]

b_acc_vit_no = [0.735557, 0.720096, 0.689242, 0.719234, 0.724011]
b_acc_vit_imgaug = [0.719543, 0.724874, 0.696828, 0.730470, 0.729629]
b_acc_vit_albumentations = [0.727130, 0.729939, 0.695147, 0.714722, 0.729917]
b_acc_vit_v2 = [0.708617, 0.710829, 0.696253, 0.713328, 0.718703]
b_acc_vit_RandAugment = [0.723458, 0.685305, 0.708794, 0.721512, 0.706936]
b_acc_vit_cross = [0.733588, 0.728523, 0.696275, 0.731885, 0.724011]

b_acc_resnet_imgaug_cross = [0.679068, 0.637220, 0.637552, 0.677740, 0.609462]
b_acc_resnet_albumentations_cross = [0.675396, 0.647903, 0.664536, 0.679643, 0.660953]
b_acc_resnet_v2_cross = [0.639344, 0.651155, 0.661926, 0.672653, 0.622069]
b_acc_resnet_randaugment_cross = [0.652570, 0.578585, 0.583363, 0.651464, 0.592896]

b_acc_vit_imgaug_cross = [0.724874, 0.730779, 0.699637, 0.731598, 0.730470]
b_acc_vit_albumentations_cross = [0.724321, 0.719256, 0.674931, 0.723724, 0.738631]
b_acc_vit_v2_cross = [0.712842, 0.715031, 0.702159, 0.704326, 0.724896]
b_acc_vit_randaugment_cross = [0.714456, 0.672100, 0.713572, 0.723480, 0.695988]

precision_resnet_no = [0.655945, 0.594842, 0.668460, 0.642333, 0.618410]
precision_resnet_imgaug = [0.690877, 0.646051, 0.654015, 0.648148, 0.671148]
precision_resnet_albumentations = [0.688887, 0.660950, 0.692164, 0.641292, 0.654724]
precision_resnet_v2 = [0.689607, 0.621147, 0.673896, 0.636824, 0.603019]
precision_resnet_RandAugment = [0.665564, 0.620096, 0.696774, 0.633709, 0.636271]
precision_resnet_cross = [0.633552, 0.662069, 0.665940, 0.667909, 0.593607]

precision_vit_no = [0.744596, 0.728944, 0.716754, 0.726374, 0.730970]
precision_vit_imgaug = [0.729981, 0.733577, 0.724126, 0.737074, 0.736301]
precision_vit_albumentations = [0.737129, 0.739613, 0.725207, 0.722155, 0.736835]
precision_vit_v2 = [0.729569, 0.720985, 0.717412, 0.720604, 0.729703]
precision_vit_RandAugment = [0.730989, 0.711269, 0.719405, 0.732174, 0.730173]
precision_vit_cross = [0.742453, 0.736574, 0.728762, 0.738812, 0.730970]

precision_resnet_imgaug_cross = [0.689308, 0.652525, 0.670611, 0.716847, 0.668294]
precision_resnet_albumentations_cross = [0.684289, 0.664089, 0.704532, 0.691923, 0.692121]
precision_resnet_v2_cross = [0.658436, 0.668482, 0.672819, 0.695651, 0.643247]
precision_resnet_randaugment_cross = [0.667550, 0.628666, 0.614733, 0.663411, 0.632957]

precision_vit_imgaug_cross = [0.733577, 0.739949, 0.726413, 0.738226, 0.737074]
precision_vit_albumentations_cross = [0.734649, 0.728526, 0.714581, 0.730584, 0.745403]
precision_vit_v2_cross = [0.738259, 0.723228, 0.725217, 0.712045, 0.738458]
precision_vit_randaugment_cross = [0.721779, 0.699146, 0.724071, 0.734383, 0.724639]

recall_resnet_no = [0.662037, 0.599537, 0.671296, 0.648148, 0.622685]
recall_resnet_imgaug = [0.694444, 0.652778, 0.659722, 0.648148, 0.613426]
recall_resnet_albumentations = [0.685185, 0.662037, 0.692130, 0.643519, 0.636574]
recall_resnet_v2 = [0.685185, 0.592593, 0.668981, 0.643519, 0.581019]
recall_resnet_RandAugment = [0.634259, 0.629630, 0.625000, 0.629630, 0.636574]
recall_resnet_cross = [0.631944, 0.664352, 0.671296, 0.608796, 0.564815]

recall_vit_no = [0.745370, 0.729167, 0.717593, 0.722222, 0.726852]
recall_vit_imgaug = [0.731481, 0.733796, 0.724537, 0.731481, 0.731481]
recall_vit_albumentations = [0.738426, 0.740741, 0.724537, 0.712963, 0.733796]
recall_vit_v2 = [0.731481, 0.722222, 0.719907, 0.715278, 0.731481]
recall_vit_RandAugment = [0.729167, 0.712963, 0.699074, 0.733796, 0.731481]
recall_vit_cross = [0.743056, 0.736111, 0.726852, 0.736111, 0.726852]

recall_resnet_imgaug_cross = [0.689815, 0.657407, 0.673611, 0.712963, 0.562500]
recall_resnet_albumentations_cross = [0.680556, 0.668981, 0.701389, 0.694444, 0.631944]
recall_resnet_v2_cross = [0.620370, 0.634259, 0.673611, 0.699074, 0.650463]
recall_resnet_randaugment_cross = [0.638889, 0.631944, 0.548611, 0.643519, 0.638889]

recall_vit_imgaug_cross = [0.733796, 0.740741, 0.726852, 0.733796, 0.731481]
recall_vit_albumentations_cross = [0.736111, 0.729167, 0.710648, 0.724537, 0.743056]
recall_vit_v2_cross = [0.738426, 0.722222, 0.726852, 0.703704, 0.740741]
recall_vit_randaugment_cross = [0.717593, 0.701389, 0.703704, 0.736111, 0.724537]

f1_resnet_no = [0.649567, 0.596472, 0.669356, 0.642853, 0.619797]
f1_resnet_imgaug = [0.691138, 0.644735, 0.653822, 0.648148, 0.609196]
f1_resnet_albumentations = [0.686510, 0.661437, 0.676802, 0.642184, 0.639354]
f1_resnet_v2 = [0.686680, 0.594489, 0.670637, 0.620457, 0.583891]
f1_resnet_RandAugment = [0.635766, 0.603465, 0.618017, 0.631188, 0.636419]
f1_resnet_cross = [0.632664, 0.662921, 0.660555, 0.604010, 0.566607]

f1_vit_no = [0.744918, 0.729051, 0.708364, 0.723541, 0.728149]
f1_vit_imgaug = [0.730477, 0.733683, 0.715816, 0.733011, 0.732890]
f1_vit_albumentations = [0.737580, 0.740032, 0.714670, 0.714990, 0.734840]
f1_vit_v2 = [0.725727, 0.721463, 0.713664, 0.716837, 0.730193]
f1_vit_RandAugment = [0.729886, 0.704163, 0.701310, 0.732661, 0.724782]
f1_vit_cross = [0.742718, 0.736327, 0.716172, 0.737067, 0.728149]

f1_resnet_imgaug_cross = [0.689547, 0.653235, 0.657361, 0.698308, 0.538431]
f1_resnet_albumentations_cross = [0.681900, 0.664244, 0.684976, 0.692641, 0.627918]
f1_resnet_v2_cross = [0.620671, 0.635326, 0.673182, 0.691011, 0.640767]
f1_resnet_randaugment_cross = [0.640749, 0.591069, 0.536903, 0.646167, 0.609948]

f1_vit_imgaug_cross = [0.733683, 0.740280, 0.718478, 0.735128, 0.733011]
f1_vit_albumentations_cross = [0.735124, 0.728811, 0.695503, 0.726164, 0.743906]
f1_vit_v2_cross = [0.731170, 0.722663, 0.720037, 0.705714, 0.738273]
f1_vit_randaugment_cross = [0.718934, 0.691323, 0.705905, 0.734845, 0.715251]

auc_resnet_no = [0.729121, 0.652836, 0.697912, 0.671017, 0.649805]
auc_resnet_imgaug = [0.742701, 0.679864, 0.702446, 0.701009, 0.712842]
auc_resnet_albumentations = [0.746550, 0.706317, 0.729607, 0.714456, 0.720362]
auc_resnet_v2 = [0.739229, 0.671857, 0.715673, 0.725294, 0.660687]
auc_resnet_RandAugment = [0.711293, 0.698155, 0.709635, 0.670729, 0.699792]
auc_resnet_cross = [0.686300, 0.676635, 0.724653, 0.725648, 0.626957]

auc_vit_no = [0.782757, 0.774020, 0.777581, 0.780169, 0.778776]
auc_vit_imgaug = [0.782314, 0.775458, 0.780921, 0.783177, 0.781540]
auc_vit_albumentations = [0.783111, 0.775015, 0.778024, 0.783133, 0.779483]
auc_vit_v2 = [0.784902, 0.770990, 0.782314, 0.775989, 0.786097]
auc_vit_RandAugment = [0.794280, 0.776719, 0.779041, 0.795895, 0.790454]
auc_vit_cross = [0.784747, 0.775015, 0.780678, 0.785721, 0.783221]

auc_resnet_imgaug_cross = [0.733279, 0.707533, 0.734628, 0.767363, 0.713218]
auc_resnet_albumentations_cross = [0.746594, 0.736464, 0.746328, 0.739560, 0.727440]
auc_resnet_v2_cross = [0.691940, 0.700058, 0.757056, 0.747257, 0.718393]
auc_resnet_randaugment_cross = [0.704216, 0.685946, 0.654229, 0.707755, 0.702999]

auc_vit_imgaug_cross = [0.783597, 0.777581, 0.782137, 0.786627, 0.785057]
auc_vit_albumentations_cross = [0.786804, 0.771123, 0.780213, 0.783066, 0.783111]
auc_vit_v2_cross = [0.787402, 0.772273, 0.783907, 0.773954, 0.786539]
auc_vit_randaugment_cross = [0.791582, 0.776409, 0.779063, 0.793794, 0.786163]


b_accs = [b_acc_resnet_no, b_acc_resnet_imgaug, b_acc_resnet_albumentations, b_acc_resnet_v2, b_acc_resnet_RandAugment, b_acc_resnet_cross,
          b_acc_vit_no, b_acc_vit_imgaug, b_acc_vit_albumentations, b_acc_vit_v2, b_acc_vit_RandAugment, b_acc_vit_cross,
          b_acc_resnet_imgaug_cross, b_acc_resnet_albumentations_cross, b_acc_resnet_v2_cross, b_acc_resnet_randaugment_cross,
          b_acc_vit_imgaug_cross, b_acc_vit_albumentations_cross, b_acc_vit_v2_cross, b_acc_vit_randaugment_cross]

precisions = [precision_resnet_no, precision_resnet_imgaug, precision_resnet_albumentations, precision_resnet_v2, precision_resnet_RandAugment, precision_resnet_cross,
              precision_vit_no, precision_vit_imgaug, precision_vit_albumentations, precision_vit_v2, precision_vit_RandAugment, precision_vit_cross,
              precision_resnet_imgaug_cross, precision_resnet_albumentations_cross, precision_resnet_v2_cross, precision_resnet_randaugment_cross,
              precision_vit_imgaug_cross, precision_vit_albumentations_cross, precision_vit_v2_cross, precision_vit_randaugment_cross]

recalls = [recall_resnet_no, recall_resnet_imgaug, recall_resnet_albumentations, recall_resnet_v2, recall_resnet_RandAugment, recall_resnet_cross,
           recall_vit_no, recall_vit_imgaug, recall_vit_albumentations, recall_vit_v2, recall_vit_RandAugment, recall_vit_cross,
           recall_resnet_imgaug_cross, recall_resnet_albumentations_cross, recall_resnet_v2_cross, recall_resnet_randaugment_cross,
           recall_vit_imgaug_cross, recall_vit_albumentations_cross, recall_vit_v2_cross, recall_vit_randaugment_cross]

f1_scores = [f1_resnet_no, f1_resnet_imgaug, f1_resnet_albumentations, f1_resnet_v2, f1_resnet_RandAugment, f1_resnet_cross,
             f1_vit_no, f1_vit_imgaug, f1_vit_albumentations, f1_vit_v2, f1_vit_RandAugment, f1_vit_cross,
             f1_resnet_imgaug_cross, f1_resnet_albumentations_cross, f1_resnet_v2_cross, f1_resnet_randaugment_cross,
             f1_vit_imgaug_cross, f1_vit_albumentations_cross, f1_vit_v2_cross, f1_vit_randaugment_cross]

aucs = [auc_resnet_no, auc_resnet_imgaug, auc_resnet_albumentations, auc_resnet_v2, auc_resnet_RandAugment, auc_resnet_cross,
        auc_vit_no, auc_vit_imgaug, auc_vit_albumentations, auc_vit_v2, auc_vit_RandAugment, auc_vit_cross,
        auc_resnet_imgaug_cross, auc_resnet_albumentations_cross, auc_resnet_v2_cross, auc_resnet_randaugment_cross,
        auc_vit_imgaug_cross, auc_vit_albumentations_cross, auc_vit_v2_cross, auc_vit_randaugment_cross]

print("Friedman Tests")
b_acc_stat, b_acc_p_value = friedmanchisquare(*(np.array(b_accs).T))
print('Friedman Test for balanced accuracy: Statistics=%.3f, p-Value=%.3f' % (b_acc_stat, b_acc_p_value))

precision_stat, precision_p_value = friedmanchisquare(*(np.array(precisions).T))
print('Friedman Test for precision: Statistics=%.3f, p-Value=%.3f' % (precision_stat, precision_p_value))

recall_stat, recall_p_value = friedmanchisquare(*(np.array(recalls).T))
print('Friedman Test for recall: Statistics=%.3f, p-Value=%.3f' % (recall_stat, recall_p_value))

f1_score_stat, f1_score_p_value = friedmanchisquare(*(np.array(f1_scores).T))
print('Friedman Test for f1-score: Statistics=%.3f, p-Value=%.3f' % (f1_score_stat, f1_score_p_value))

auc_stat, auc_p_value = friedmanchisquare(*(np.array(aucs).T))
print('Friedman Test for AUC-ROC: Statistics=%.3f, p-Value=%.3f' % (auc_stat, auc_p_value))


print("Balanced Accuracies")
t5_stat, p5_value = ttest_rel(b_acc_resnet_imgaug, b_acc_resnet_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(b_acc_resnet_albumentations, b_acc_resnet_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(b_acc_resnet_v2, b_acc_resnet_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(b_acc_resnet_RandAugment, b_acc_resnet_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(b_acc_vit_imgaug, b_acc_vit_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(b_acc_vit_albumentations, b_acc_vit_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(b_acc_vit_v2, b_acc_vit_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(b_acc_vit_RandAugment, b_acc_vit_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))


print("Precisions")
t5_stat, p5_value = ttest_rel(precision_resnet_imgaug, precision_resnet_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(precision_resnet_albumentations, precision_resnet_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(precision_resnet_v2, precision_resnet_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(precision_resnet_RandAugment, precision_resnet_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(precision_vit_imgaug, precision_vit_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(precision_vit_albumentations, precision_vit_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(precision_vit_v2, precision_vit_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(precision_vit_RandAugment, precision_vit_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))


print("Recalls")
t5_stat, p5_value = ttest_rel(recall_resnet_imgaug, recall_resnet_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(recall_resnet_albumentations, recall_resnet_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(recall_resnet_v2, recall_resnet_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(recall_resnet_RandAugment, recall_resnet_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(recall_vit_imgaug, recall_vit_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(recall_vit_albumentations, recall_vit_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(recall_vit_v2, recall_vit_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(recall_vit_RandAugment, recall_vit_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))


print("F1-Scores")
t5_stat, p5_value = ttest_rel(f1_resnet_imgaug, f1_resnet_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(f1_resnet_albumentations, f1_resnet_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(f1_resnet_v2, f1_resnet_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(f1_resnet_RandAugment, f1_resnet_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(f1_vit_imgaug, f1_vit_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(f1_vit_albumentations, f1_vit_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(f1_vit_v2, f1_vit_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(f1_vit_RandAugment, f1_vit_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))


print("AUC-ROC")
t5_stat, p5_value = ttest_rel(auc_resnet_imgaug, auc_resnet_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(auc_resnet_albumentations, auc_resnet_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(auc_resnet_v2, auc_resnet_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for resnet v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(auc_resnet_RandAugment, auc_resnet_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for resnet RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(auc_vit_imgaug, auc_vit_imgaug_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer imgaug and imgaug+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(auc_vit_albumentations, auc_vit_albumentations_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer albumentations and albumentations+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))

t5_stat, p5_value = ttest_rel(auc_vit_v2, auc_vit_v2_cross)
if p5_value < 0.05:
    print('Paired t-test for vision transformer v2 and v2+cross: t-Statistic=%.3f, p-Value=%.3f' % (t5_stat, p5_value))

t6_stat, p6_value = ttest_rel(auc_vit_RandAugment, auc_vit_randaugment_cross)
if p6_value < 0.05:
    print('Paired t-test for vision transformer RandAugment and RandAugment+cross: t-Statistic=%.3f, p-Value=%.3f' % (t6_stat, p6_value))
