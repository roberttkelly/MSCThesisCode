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


cutoffs_file_name = 'ens-tiered3-cutoffs.csv'
cutoffs_dat = read.csv(cutoffs_file_name, stringsAsFactors=F)
head(cutoffs_dat)

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
}

dat = train[, c('subj_id', 'side', 'image', 'level')]

for (i in 1:nrow(predictor_dat)) {
  id = predictor_dat$id[i]
  cuts = as.numeric(cutoffs_dat[cutoffs_dat$id == id, -(1:2)])
  train_tmp = train[, c('subj_id', 'side', 'image', 'level', id)]
  train_tmp = left_right_join(train_tmp, id)
  train_y = train_tmp$level
  train_x = train_tmp[, !names(train_tmp) %in% c('subj_id', 'level', 'sz')]
  train_x = model.matrix(~(0+.)^2, data=train_x)
  fit = readRDS(paste0('models/lr-', id, '.rds'))
  scores = predict(fit, train_x, type='response')[, 1]
  preds = rep(0, nrow(train_x))
  preds[scores > cuts[1]] = 1
  preds[scores > cuts[2]] = 2
  preds[scores > cuts[3]] = 3
  preds[scores > cuts[4]] = 4
  new_dat = data.frame(subj_id = train_tmp$subj_id, 
                       side = train_tmp$side, 
                       x = factor(preds))
  new_dat$side = factor(new_dat$side, 
                  levels=c(0, 1), 
                  labels=c('left', 'right'))
  names(new_dat)[3] = id
  dat = merge(dat, new_dat)
}



train_y = dat$level
train_x = dat[, !names(dat) %in% c('subj_id', 'level', 'image', 'side')]
# train_x = model.matrix(~(0+.), data=train_x)
dv = dummyVars(~., data=train_x)
train_x = predict(dv, train_x)
summary(train_x)
set.seed(2)

tmp = data.frame(subj_id = dat$subj_id, 
                 side = as.numeric(dat$side) - 1, 
                 level = dat$level, 
                 train_x)
write.csv(tmp, 'ensemble_fitting_matrix.csv', row.names=F)


fit = cv.glmnet(y=train_y, 
                x=train_x,
                nfolds=30,
                alpha=0.01,
                family='gaussian', 
                standardize=T, 
                nlambda=300, 
                lambda.min.ratio=0.001)
saveRDS(fit, '../../models/output3/models/ens-tiered3.rds')
# fit = readRDS('models/best.rds')

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
                  actual = train_y)
# table(rslt)
write.table(rslt, 'kappascan.tsv', 
            sep='\t', quote=F, na='', row.names=F, col.names=F)



#               var        coef
# 1     (Intercept)  1.00703693
# 6           m42.4  0.21580259
# 35          m53.4  0.20662566
# 43    m53._2_psl4  0.20063436
# 23          m47.4  0.19667538
# 19          m46.4  0.19301233
# 10          m41.4  0.19283115
# 15        cyc28.4  0.18506147
# 27    m52_no_bg.4  0.18209339
# 39        m51_1.4  0.16774925
# 31    m51_no_bg.4  0.16336390
# 51 m55_bg_small.4  0.15757200
# 47        m58_1.4  0.14043323
# 14        cyc28.3  0.12226221
# 26    m52_no_bg.3  0.11398765
# 46        m58_1.3  0.11031648
# 18          m46.3  0.11002835
# 34          m53.3  0.10300322
# 38        m51_1.3  0.10169496
# 42    m53._2_psl3  0.10056483
# 50 m55_bg_small.3  0.09909630
# 5           m42.3  0.09768132
# 9           m41.3  0.09288147
# 22          m47.3  0.08618789
# 30    m51_no_bg.3  0.08611033
# 49 m55_bg_small.2  0.07212585
# 33          m53.2  0.06443404
# 29    m51_no_bg.2  0.06262744
# 37        m51_1.2  0.06241895
# 41    m53._2_psl2  0.05523964
# 21          m47.2  0.05352036
# 45        m58_1.2  0.05326154
# 25    m52_no_bg.2  0.05204018
# 8           m41.2  0.04928406
# 17          m46.2  0.04867028
# 4           m42.2  0.04551168
# 13        cyc28.2  0.03034281
# 3           m42.1 -0.01101457
# 12        cyc28.1 -0.01953295
# 2           m42.0 -0.05911802
# 11        cyc28.0 -0.06543296
# 16          m46.0 -0.07257809
# 36        m51_1.0 -0.07270861
# 40    m53._2_psl0 -0.07282505
# 20          m47.0 -0.07292000
# 7           m41.0 -0.07349456
# 48 m55_bg_small.0 -0.07429728
# 32          m53.0 -0.07618830
# 24    m52_no_bg.0 -0.07789236
# 44        m58_1.0 -0.08176862
# 28    m51_no_bg.0 -0.08553461
# [1] 239
# [1] 0.2359427