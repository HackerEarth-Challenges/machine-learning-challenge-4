path <- "challenge_four/"
setwd(path)


## load data
library(data.table)
train <- fread("train_data.csv")
test <- fread("test_data.csv")

## check data
head(train)
head(test)

## check missing values - no missing values
colSums(is.na(train))
colSums(is.na(test))

## check target class
train[,.N/nrow(train), target]


# Model 1 -----------------------------------------------------------------

## lets predict the majority class
sampsub <- fread("sample_submission.csv")
samplesub[, target := 0]
fwrite(samplesub, "sub0.csv")


# Model 2 + 3 -----------------------------------------------------------------
## Random Forest
## Gradient Boosting

library(xgboost)
library(caTools)

getXGBData <- function(train, test)
{
  split <- sample.split(Y = train$target, SplitRatio = 0.5)
  
  dtrain <- train[split]
  dvalid <- train[!split]
  
  dtrain <- xgb.DMatrix(data = as.matrix(dtrain[,-c('connection_id','target'), with=F]), label = dtrain$target)
  dvalid <- xgb.DMatrix(data = as.matrix(dvalid[, -c('connection_id', 'target'), with = F]), label = dvalid$target)
  dtest <- xgb.DMatrix(data = as.matrix(test[,-c('connection_id'), with=F]))
  
  return (list(train = dtrain, test = dtest, eval = dvalid))
  
}

getMulAcc <- function(pred, dtrain)
{
  label <- getinfo(dtrain, "label")
  acc <- mean(label == pred)
  return(list(metric = "maccuracy", value = acc))
}


runModels <- function(dtrain, dtest, dvalid, XGB = 0) ## set 1 for XGB: Default run is RF
{
  
  if(XGB == 1)
  {
    cat('Running Gradient Boosting Model...\n\n')
    # default parameters
    params <- list(objective = 'multi:softmax',
                   num_class = 3)
    
    watchlist <- list('train' = dtrain, 'valid' = dvalid)
    clf <- xgb.train(params
                     ,dtrain
                     ,1000
                     ,watchlist
                     ,feval = getMulAcc
                     ,print_every_n = 20
                     ,early_stopping_rounds = 30
                     ,maximize = T
    )
    
    
    pred <- predict(clf, dtest)
    
  } else if (XGB == 0) {
    
    cat('Running Random Forest Model...\n\n')
    params <- list(booster = 'dart'
                   ,objective = 'multi:softmax'
                   ,num_class = 3
                   ,normalize_type = 'tree'
                   ,rate_drop = 0.1)
    
    watchlist <- list('train' = dtrain, 'valid' = dvalid)
    clf <- xgb.train(params
                     ,dtrain
                     ,1000
                     ,watchlist
                     ,feval = getMulAcc
                     ,print_every_n = 20
                     ,early_stopping_rounds = 30
                     ,maximize = T
    )
    
    
    pred <- predict(clf, dtest)
    
  } 
  
  return(pred)

}

xgbdata <- getXGBData(train, test)

predsRF <- runModels(dtrain = xgbdata$train, dtest = xgbdata$test, dvalid = xgbdata$eval)
predsXGB <- runModels(dtrain = xgbdata$train, dtest = xgbdata$test, dvalid = xgbdata$eval,XGB = 1)


## make submissions
sampsub1 <- fread("sample_submission.csv")
sampsub1[, target := predsRF]
fwrite(sampsub1, "sub1.csv")

sampsub2 <- fread("sample_submission.csv")
sampsub2[, target := predsXGB]
fwrite(sampsub2, "sub2.csv")


## What's next ? 
## Since the data set is pretty clean, you can now start with feature engineering, ensembling models etc. 
