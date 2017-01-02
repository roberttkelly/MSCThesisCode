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

predictor_file_name = 'ens-105-predictors.csv'
predictor_dat = read.csv(predictor_file_name, stringsAsFactors=F)
print(nrow(predictor_dat))

left_right_join = function(dat) {
  tmp = dat
  tmp = merge(tmp[tmp$side == 'left', ], 
              tmp[tmp$side == 'right', ], 
              by = 'subj_id')
  vv = predictor_dat$id
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
  for (i in 1:nrow(predictor_dat)) {
    current_id = predictor_dat$id[i]
    tmp = cbind(tmp, xxx = pmax(tmp[, paste(current_id, '1', sep='_')], 
                                tmp[, paste(current_id, '2', sep='_')]))
    names(tmp)[names(tmp) == 'xxx'] = paste(current_id, 'max', sep='_')
    tmp = cbind(tmp, xxx = pmin(tmp[, paste(current_id, '1', sep='_')], 
                                tmp[, paste(current_id, '2', sep='_')]))
    names(tmp)[names(tmp) == 'xxx'] = paste(current_id, 'min', sep='_')
  }
  tmp$level = tmp$level + 1
  tmp$side = as.numeric(tmp$side) - 1
  tmp
}

max_min = function(dat) {
  tmp = dat
  tmp = cbind(tmp, xxx = apply(tmp[, grep('_1', names(tmp))], 1, max))
  names(tmp)[names(tmp) == 'xxx'] = 'max_1'
  tmp = cbind(tmp, xxx = apply(tmp[, grep('_1', names(tmp))], 1, min))
  names(tmp)[names(tmp) == 'xxx'] = 'min_1'
  tmp
}

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
  dat = read.csv(predictor_dat$val[i], header=F)
  dat = transform(dat,
    subj_id = gsub('_left|_right|\\.jpeg$', '', V1),
    side = gsub('^[0-9]*_|\\.jpeg$', '', V1),
    pred = V2)
  dat = dat[,c('subj_id', 'side', 'pred')]
  names(dat)[3] = predictor_dat$id[i]
  train = merge(train, dat)
  print(dim(train))
}

train$m46 = 0.5 * (train$m46 + train$m46_2)
train$m46_2 = NULL
predictor_dat = predictor_dat[predictor_dat$id != 'm46_2', ]

train = left_right_join(train)
train = max_min(train)


write.csv(train, 'ensemble_fitting_matrix.csv', row.names=F)

train_y = train$level
train_x = train[, !names(train) %in% c('subj_id', 'level', 'sz')]
train_x = model.matrix(~(0+.)^2, data=train_x)
dim(train_x)

set.seed(5)
fit = cv.glmnet(y=train_y, 
                x=train_x,
                type.measure='mse',
                nfolds=30,
                alpha=0.6,
                family='gaussian',
                standardize=T, 
                nlambda=300, 
                lambda.min.ratio=0.001)
saveRDS(fit, '../../models/output3/models/ens105.rds')
fit = readRDS('../../models/output3/models/ens105.rds')
coefs = as.matrix(coef(fit))[as.matrix(coef(fit)) != 0]
names(coefs) = rownames(coef(fit))[as.matrix(coef(fit)) != 0]
bestIndx = which(fit$cvm == min(fit$cvm))
tmp = data.frame(var=names(coefs), coef=coefs)
rownames(tmp) = NULL
tmp[rev(order(tmp$coef)), ]

bestIndx
fit$cvm[bestIndx]
fit$lambda[bestIndx]
summary(fit$lambda)

preds = predict(fit, train_x, type='response')[, 1]
rslt = data.frame(pred = preds, 
                  actual = train_y - 1)
table(rslt$actual)
prop.table(table(rslt$actual))
write.table(rslt, 'kappascan.tsv', 
            sep='\t', quote=F, na='', row.names=F, col.names=F)

#                        var         coef
# 1              (Intercept) 0.3583172063
# 6                    m53_1 0.0921488777
# 9            m52_no_bg_max 0.0885712307
# 4              m52_no_bg_1 0.0828727706
# 5              m51_no_bg_1 0.0820158333
# 11                 m53_max 0.0789745074
# 3                    m47_1 0.0507662479
# 10           m51_no_bg_max 0.0395980269
# 2                    m46_1 0.0368135557
# 8                  m47_max 0.0315050330
# 7                  m46_max 0.0140580797
# 12                   max_1 0.0130081083
# 23   m41_max:m52_no_bg_max 0.0058689359
# 20         m42_max:m53_max 0.0054791081
# 24         m41_max:m53_max 0.0049071226
# 19   m42_max:m52_no_bg_max 0.0045980725
# 29         m47_max:m53_max 0.0043879388
# 32     m52_no_bg_max:max_1 0.0043180213
# 28   m47_max:m52_no_bg_max 0.0042620666
# 33           m53_max:max_1 0.0037329151
# 27       cyc28_max:m53_max 0.0036576947
# 26 cyc28_max:m52_no_bg_max 0.0035159808
# 22         m41_max:m47_max 0.0033607143
# 31   m52_no_bg_max:m53_max 0.0032091486
# 18         m42_max:m47_max 0.0029636488
# 25       cyc28_max:m47_max 0.0027404610
# 13                   min_1 0.0026252536
# 30           m47_max:max_1 0.0012280575
# 16           m53_1:m42_max 0.0012112265
# 14           m47_1:m42_max 0.0008127055
# 21           m42_max:max_1 0.0008016348
# 15           m47_1:m41_max 0.0006854084
# 17           m53_1:m41_max 0.0004574470
# [1] 288
# [1] 0.238064
# [1] 0.001754512

# best score is 0.84872806
# best cut off
# 2573 1.5064173
# 2927 2.200046
# 3262 2.9302843
# 3485 4.0473447
