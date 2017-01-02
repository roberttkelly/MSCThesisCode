library(glmnet)
library(ggplot2)
library(caret)
library(plyr)
# set.seed(2)

img_dir = '../../../../kaggle-eye/data/crop256x256/train'
args = commandArgs(TRUE)
if (length(args)) {
  img_dir = args[1]
}

set.seed(6)


left_right_join = function(dat, vv) {
  tmp = dat
  tmp = merge(tmp[tmp$side == 'left', ], 
              tmp[tmp$side == 'right', ], 
              by = 'subj_id')
  tmp1 = tmp[, c('subj_id', 
                 'side.x', 
                 'level.x', 
                 paste(vv, '.x', sep=''), 
                 paste(vv, '.y', sep=''))]
  names(tmp1) = gsub('\\.x', '_1', names(tmp1))
  names(tmp1) = gsub('\\.y', '_2', names(tmp1))
  names(tmp1)[1:3] = c('subj_id', 'side', 'level')
  tmp2 = tmp[, c('subj_id', 
                 'side.y', 
                 'level.y', 
                 paste(vv, '.y', sep=''), 
                 paste(vv, '.x', sep=''))]
  names(tmp2) = gsub('\\.y', '_1', names(tmp2))
  names(tmp2) = gsub('\\.x', '_2', names(tmp2))
  names(tmp2)[1:3] = c('subj_id', 'side', 'level')
  tmp = rbind(tmp1, tmp2)
  tmp = cbind(tmp, xxx = pmax(tmp[, paste(vv, '1', sep='_')], 
                              tmp[, paste(vv, '2', sep='_')]))
  names(tmp)[names(tmp) == 'xxx'] = paste(vv, 'max', sep='_')
  tmp = cbind(tmp, xxx = pmin(tmp[, paste(vv, '1', sep='_')], 
                              tmp[, paste(vv, '2', sep='_')]))
  names(tmp)[names(tmp) == 'xxx'] = paste(vv, 'min', sep='_')
  tmp$level = tmp$level + 1
  tmp$side = as.numeric(tmp$side) - 1
  tmp
}

predictor_file_name = 'ens-tiered3-predictors.csv'
predictor_dat = read.csv(predictor_file_name, stringsAsFactors=F)
print(nrow(predictor_dat))

train = dir(img_dir)
train = data.frame(
  subj_id = gsub('_left|_right|\\.jpeg$', '', train),
  side = gsub('^[0-9]*_|\\.jpeg$', '', train))
labels = read.csv('trainLabels.csv')
labels = transform(labels, 
  subj_id = gsub('_left|_right|\\.jpeg$', '', image),
  side = gsub('^[0-9]*_|\\.jpeg$', '', image))
train = merge(train, labels)
for (i in 1:nrow(predictor_dat)) {
  print(predictor_dat$id[i])
  dat = read.csv(predictor_dat$val[i], header=F)
  dat = transform(dat,
    subj_id = gsub('_left|_right|\\.jpeg$', '', V1),
    side = gsub('^[0-9]*_|\\.jpeg$', '', V1),
    pred = V2)
  dat = dat[,c('subj_id', 'side', 'pred')]
  names(dat)[3] = predictor_dat$id[i]
  train = merge(train, dat)
  print(head(train))
}

set.seed(5)
for (i in 1:nrow(predictor_dat)) {
  id = predictor_dat$id[i]
  train_tmp = train[, c('subj_id', 'side', 'image', 'level', id)]
  train_tmp = left_right_join(train_tmp, id)
  train_y = train_tmp$level
  train_x = train_tmp[, !names(train_tmp) %in% c('subj_id', 'level', 'sz')]
  train_x = model.matrix(~(0+.)^2, data=train_x)
  fit = cv.glmnet(y=train_y, 
                  x=train_x,
                  type.measure='mse',
                  nfolds=30,
                  alpha=0.5,
                  family='gaussian',
                  standardize=T, 
                  nlambda=300, 
                  lambda.min.ratio=0.001)
  saveRDS(fit, paste0('../../models/output3/models/lr-', id, '.rds'))
  fit = readRDS(paste0('../../models/output3/models/lr-', id, '.rds'))
  coefs = as.matrix(coef(fit))[as.matrix(coef(fit)) != 0]
  names(coefs) = rownames(coef(fit))[as.matrix(coef(fit)) != 0]
  bestIndx = which(fit$cvm == min(fit$cvm))
  tmp = data.frame(var=names(coefs), coef=coefs)
  rownames(tmp) = NULL
  print(id)
  print(tmp[rev(order(tmp$coef)), ])
  print(bestIndx)
  print(fit$cvm[bestIndx])
  preds = predict(fit, train_x, type='response')[, 1]
  rslt = data.frame(pred = preds, 
                    actual = train_y - 1)
  write.table(rslt, paste0('kappascan-', id, '.tsv'), 
              sep='\t', quote=F, na='', row.names=F, col.names=F)
}

# [1] "m42"
#               var        coef
# 1     (Intercept) 0.416689016
# 4         m42_max 0.243213575
# 2           m42_1 0.228887795
# 5         m42_min 0.048317185
# 7   m42_1:m42_max 0.038474209
# 3           m42_2 0.032781294
# 8   m42_2:m42_max 0.015593009
# 9 m42_max:m42_min 0.009588521
# 6     m42_1:m42_2 0.009053049
# [1] 289
# [1] 0.2957058
# [1] "m41"
#                var        coef
# 1      (Intercept) 0.426780800
# 2            m41_1 0.231733521
# 4          m41_max 0.223381661
# 5          m41_min 0.065797750
# 7    m41_1:m41_max 0.031865395
# 3            m41_2 0.028439166
# 10 m41_max:m41_min 0.012977646
# 6      m41_1:m41_2 0.012430827
# 9    m41_2:m41_max 0.012225833
# 8    m41_1:m41_min 0.003471466
# [1] 268
# [1] 0.2977783
# [1] "cyc28"
#                   var        coef
# 1         (Intercept) 0.392364560
# 4           cyc28_max 0.264141218
# 2             cyc28_1 0.236201733
# 3             cyc28_2 0.053673378
# 5           cyc28_min 0.046219630
# 7   cyc28_1:cyc28_max 0.030520646
# 8   cyc28_2:cyc28_max 0.015477485
# 9 cyc28_max:cyc28_min 0.008893216
# 6     cyc28_1:cyc28_2 0.008303821
# [1] 200
# [1] 0.3277148
# [1] "m46"
#               var        coef
# 1     (Intercept) 0.354145415
# 2           m46_1 0.283782923
# 4         m46_max 0.267375620
# 5         m46_min 0.045921059
# 7   m46_1:m46_max 0.035804841
# 3           m46_2 0.024297235
# 8   m46_2:m46_max 0.011915412
# 9 m46_max:m46_min 0.007476839
# 6     m46_1:m46_2 0.006857659
# [1] 216
# [1] 0.248198
# [1] "m47"
#               var        coef
# 1     (Intercept) 0.376706135
# 2           m47_1 0.286382113
# 4         m47_max 0.275643901
# 7   m47_1:m47_max 0.038954636
# 5         m47_min 0.032840757
# 3           m47_2 0.021225881
# 8   m47_2:m47_max 0.015273740
# 9 m47_max:m47_min 0.006550669
# 6     m47_1:m47_2 0.005837703
# [1] 219
# [1] 0.2525262
# [1] "m52_no_bg"
#                           var        coef
# 1                 (Intercept) 0.370204005
# 4               m52_no_bg_max 0.288083622
# 2                 m52_no_bg_1 0.287836898
# 7   m52_no_bg_1:m52_no_bg_max 0.039973626
# 3                 m52_no_bg_2 0.024908873
# 5               m52_no_bg_min 0.021454362
# 8   m52_no_bg_2:m52_no_bg_max 0.015691572
# 9 m52_no_bg_max:m52_no_bg_min 0.007455360
# 6     m52_no_bg_1:m52_no_bg_2 0.006919413
# [1] 216
# [1] 0.2428971
# [1] "m51_no_bg"
#                           var        coef
# 1                 (Intercept) 0.381435609
# 2                 m51_no_bg_1 0.280782751
# 4               m51_no_bg_max 0.258988214
# 5               m51_no_bg_min 0.063472451
# 3                 m51_no_bg_2 0.032434435
# 7   m51_no_bg_1:m51_no_bg_max 0.030214024
# 8   m51_no_bg_2:m51_no_bg_max 0.009065933
# 9 m51_no_bg_max:m51_no_bg_min 0.007779661
# 6     m51_no_bg_1:m51_no_bg_2 0.007299311
# [1] 193
# [1] 0.2504375
# [1] "m53"
#               var        coef
# 1     (Intercept) 0.357204326
# 2           m53_1 0.284671518
# 4         m53_max 0.268788016
# 7   m53_1:m53_max 0.037391862
# 5         m53_min 0.033715165
# 8   m53_2:m53_max 0.014667630
# 3           m53_2 0.014591979
# 9 m53_max:m53_min 0.008611662
# 6     m53_1:m53_2 0.008024322
# [1] 299
# [1] 0.2420896
# [1] "m51_1"
#                   var        coef
# 1         (Intercept) 0.338811856
# 2             m51_1_1 0.292829034
# 4           m51_1_max 0.279876781
# 5           m51_1_min 0.036769033
# 7   m51_1_1:m51_1_max 0.035794808
# 3             m51_1_2 0.022960666
# 8   m51_1_2:m51_1_max 0.012619759
# 9 m51_1_max:m51_1_min 0.006902321
# 6     m51_1_1:m51_1_2 0.006234654
# [1] 297
# [1] 0.2446359
# [1] "m53_2_psl"
#                           var        coef
# 2                 m53_2_psl_1 0.310075091
# 4               m53_2_psl_max 0.287599746
# 1                 (Intercept) 0.285341229
# 5               m53_2_psl_min 0.058529409
# 3                 m53_2_psl_2 0.032852965
# 7   m53_2_psl_1:m53_2_psl_max 0.031808875
# 8   m53_2_psl_2:m53_2_psl_max 0.009516492
# 9 m53_2_psl_max:m53_2_psl_min 0.006464884
# 6     m53_2_psl_1:m53_2_psl_2 0.005847056
# [1] 195
# [1] 0.2490177

# [1] "m58_1"
#                    var         coef
# 1          (Intercept) 3.255468e-01
# 2              m58_1_1 2.891454e-01
# 4            m58_1_max 2.889516e-01
# 7    m58_1_1:m58_1_max 4.515542e-02
# 5            m58_1_min 2.005132e-02
# 9    m58_1_2:m58_1_max 1.811584e-02
# 3              m58_1_2 1.660538e-02
# 10 m58_1_max:m58_1_min 1.082240e-02
# 6      m58_1_1:m58_1_2 1.026391e-02
# 8    m58_1_1:m58_1_min 1.120973e-06
# [1] 224
# [1] 0.2488325
# [1] "m55_bg_small"
#                                 var        coef
# 1                       (Intercept) 0.375004626
# 4                  m55_bg_small_max 0.291868563
# 2                    m55_bg_small_1 0.289492819
# 7   m55_bg_small_1:m55_bg_small_max 0.038729316
# 3                    m55_bg_small_2 0.020618985
# 8   m55_bg_small_2:m55_bg_small_max 0.016291012
# 5                  m55_bg_small_min 0.008931466
# 9 m55_bg_small_max:m55_bg_small_min 0.005908733
# 6     m55_bg_small_1:m55_bg_small_2 0.005279540
# [1] 218
# [1] 0.2472792

# [1] "m62_std"
#                        var         coef
# 1              (Intercept) 3.747926e-01
# 2                m62_std_1 2.741062e-01
# 4              m62_std_max 2.534453e-01
# 5              m62_std_min 5.357088e-02
# 7    m62_std_1:m62_std_max 3.517945e-02
# 3                m62_std_2 2.179335e-02
# 9    m62_std_2:m62_std_max 1.210529e-02
# 10 m62_std_max:m62_std_min 8.588284e-03
# 6      m62_std_1:m62_std_2 8.059235e-03
# 8    m62_std_1:m62_std_min 7.467923e-05
# [1] 216
# [1] 0.2444287