library(glmnet)
library(ggplot2)
library(caret)
library(plyr)
# set.seed(2)

img_dir = '../../../../kaggle-eye/data/crop256x256/test'
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
                 paste(vv, '.x', sep=''), 
                 paste(vv, '.y', sep=''))]
  names(tmp1) = gsub('\\.x', '_1', names(tmp1))
  names(tmp1) = gsub('\\.y', '_2', names(tmp1))
  names(tmp1)[1:2] = c('subj_id', 'side')
  tmp2 = tmp[, c('subj_id', 
                 'side.y', 
                 paste(vv, '.y', sep=''), 
                 paste(vv, '.x', sep=''))]
  names(tmp2) = gsub('\\.y', '_1', names(tmp2))
  names(tmp2) = gsub('\\.x', '_2', names(tmp2))
  names(tmp2)[1:2] = c('subj_id', 'side')
  tmp = rbind(tmp1, tmp2)
  tmp = cbind(tmp, xxx = pmax(tmp[, paste(vv, '1', sep='_')], 
                              tmp[, paste(vv, '2', sep='_')]))
  names(tmp)[names(tmp) == 'xxx'] = paste(vv, 'max', sep='_')
  tmp = cbind(tmp, xxx = pmin(tmp[, paste(vv, '1', sep='_')], 
                              tmp[, paste(vv, '2', sep='_')]))
  names(tmp)[names(tmp) == 'xxx'] = paste(vv, 'min', sep='_')
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
for (i in 1:nrow(predictor_dat)) {
  dat = read.csv(predictor_dat$test[i], header=F)
  dat$V1 = gsub('^test\\/', '', dat$V1)
  dat = transform(dat,
    subj_id = gsub('|_left|_right|\\.jpeg$', '', V1),
    side = gsub('^[0-9]*_|\\.jpeg$', '', V1),
    pred = V2)
  dat = dat[,c('subj_id', 'side', 'pred')]
  names(dat)[3] = predictor_dat$id[i]
  train = merge(train, dat)
}
head(train)

dat = train[, c('subj_id', 'side')]

for (i in 1:nrow(predictor_dat)) {
  id = predictor_dat$id[i]
  cuts = as.numeric(cutoffs_dat[cutoffs_dat$id == id, -(1:2)])
  train_tmp = train[, c('subj_id', 'side', id)]
  train_tmp = left_right_join(train_tmp, id)
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
  names(new_dat)[3] = paste0('preds_', id)
  dat = merge(dat, new_dat)
}

train_x = dat[, !names(dat) %in% c('subj_id', 'level', 'image', 'side')]
dv = dummyVars(~., data=train_x)
train_x = predict(dv, train_x)
head(train_x)


tmp = data.frame(subj_id = dat$subj_id, 
                 side = as.numeric(dat$side) - 1, 
                 level = 0, 
                 train_x)
write.table(tmp, 'ensemble_test_matrix.tsv', 
            sep='\t', quote=F, na='', row.names=F, col.names=T)
tmp = data.frame(subj_id = dat$subj_id, 
                 side = as.numeric(dat$side) - 1)
write.table(tmp, 'ensemble_test_matrix_ids.tsv', 
            sep='\t', quote=F, na='', row.names=F, col.names=T)









fit = readRDS('../../models/output3/models/ens-tiered3.rds')
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
dat = train[, c('subj_id', 'side')]
length(preds)
head(preds)
head(dat)

dat$image = paste0(dat$subj_id, '_', dat$side)
dat$score = preds
dat$level = 0


cutoffs = c(0.7, 1.3487674, 1.8685806, 2.5627728)
dat$level[dat$score > cutoffs[1]] = 1
dat$level[dat$score > cutoffs[2]] = 2
dat$level[dat$score > cutoffs[3]] = 3
dat$level[dat$score > cutoffs[4]] = 4
head(dat)
table(dat$level)
prop.table(table(dat$level))
write.table(dat[, c('image', 'level')], '../../models/output3/submissions/ens-tiered3-1-submission.csv', 
            sep=',', quote=F, na='', row.names=F)


d1 = read.csv('../../models/output3/submissions/ens-tiered3-1-submission.csv')
d2 = read.csv('../../models/output3/submissions/best.csv')
head(d1)

dat = merge(d1, d2, by='image')
table(dat$level.x, dat$level.y)
