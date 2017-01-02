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


left_right_join = function(dat) {
  tmp = dat
  tmp = merge(tmp[tmp$side == 'left', ], 
              tmp[tmp$side == 'right', ], 
              by = 'subj_id')
  vv = predictor_dat$id
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
  for (i in 1:nrow(predictor_dat)) {
    current_id = predictor_dat$id[i]
    tmp = cbind(tmp, xxx = pmax(tmp[, paste(current_id, '1', sep='_')], 
                                tmp[, paste(current_id, '2', sep='_')]))
    names(tmp)[names(tmp) == 'xxx'] = paste(current_id, 'max', sep='_')
    tmp = cbind(tmp, xxx = pmin(tmp[, paste(current_id, '1', sep='_')], 
                                tmp[, paste(current_id, '2', sep='_')]))
    names(tmp)[names(tmp) == 'xxx'] = paste(current_id, 'min', sep='_')
  }
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


predictor_file_name = 'ens-105-predictors.csv'
predictor_dat = read.csv(predictor_file_name, stringsAsFactors=F)
print(nrow(predictor_dat))

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
  print(i)
  print(predictor_dat$test[i])
  print(head(dat))
  names(dat)[3] = predictor_dat$id[i]
  train = merge(train, dat)
  print(dim(train))
}
head(train)


train$m46 = 0.5 * (train$m46 + train$m46_2)
train$m46_2 = NULL
predictor_dat = predictor_dat[predictor_dat$id != 'm46_2', ]

train = left_right_join(train)
train = max_min(train)


train_x = train[, !names(train) %in% c('subj_id', 'level', 'sz')]
train_x = model.matrix(~(0+.)^2, data=train_x)
dim(train_x)

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
dat = train[, c('subj_id', 'side')]
length(preds)
head(preds)
head(dat)

dat$side = factor(dat$side, 
                  levels=c(0, 1), 
                  labels=c('left', 'right'))
dat$image = paste0(dat$subj_id, '_', dat$side)
dat$score = preds
dat$level = 0
cutoffs = c(1.5064173, 2.200046, 2.9302843, 4.0473447)
dat$level[dat$score > cutoffs[1]] = 1
dat$level[dat$score > cutoffs[2]] = 2
dat$level[dat$score > cutoffs[3]] = 3
dat$level[dat$score > cutoffs[4]] = 4
head(dat)
table(dat$level)
prop.table(table(dat$level))

write.table(dat[, c('image', 'level')], '../../models/output3/submissions/ens105-submission.csv', 
            sep=',', quote=F, na='', row.names=F)
write.table(dat[, c('image', 'score')], '../../models/output3/submissions/ens105-test-scores.csv', 
            sep=',', quote=F, na='', row.names=F)
