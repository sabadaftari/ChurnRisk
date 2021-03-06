knitr::opts_chunk$set(echo = TRUE)
library(tidyverse,verbose=FALSE)    
library(randomForest,verbose=FALSE)
library(dplyr)
library(Amelia)
library(mice)
library(e1071)
set.seed(7879) #big prime number
# ---------------------------------------------------

#tmp=md.pattern(data)
data = read.csv("/Users/sabadaftari/Downloads/train_student.csv")
test = read.csv("/Users/sabadaftari/Downloads/score_student_withID.csv")
# -------------------only promo false----------------------

data = data %>% filter(promo == "False")
test = test %>% filter(promo == "False")
str(test)
str(data)
# -------------------discarding variables----------------------
data = subset(data, select = -c(time_since_voice_overage)) #too much missing values
test = subset(test, select = -c(time_since_voice_overage))
# ---------------------cleaning factors -----------------------

test$active =as.factor(test$active)

data$plan_type = as.factor(data$plan_type)
data$gender = as.integer(as.factor(data$gender))

test$plan_type =as.factor(test$plan_type)
test$gender = as.integer(as.factor(test$gender))

data["plan_type_bring"] = 0 #one-hot encoding
data["plan_type_buy"] = 0
data["plan_type_rent"] = 0
data["time_since_data_overage_missing"] = 0
data["time_since_overage_missing"] = 0
data["time_since_complaints_missing"] = 0
data["time_since_technical_problems_missing"] = 0

test["plan_type_bring"] = 0 #one-hot encoding
test["plan_type_buy"] = 0
test["plan_type_rent"] = 0
test["time_since_data_overage_missing"] = 0
test["time_since_overage_missing"] = 0
test["time_since_complaints_missing"] = 0
test["time_since_technical_problems_missing"] = 0

data$plan_type_bring[data$plan_type == 'bring'] <- 1  
data$plan_type_buy[data$plan_type == 'buy'] <- 1 
data$plan_type_rent[data$plan_type == 'rent'] <- 1 
data$time_since_data_overage_missing[data$time_since_data_overage == TRUE] <- 1  
data$time_since_overage_missing[data$time_since_overage == TRUE] <- 1  
data$time_since_complaints_missing[data$time_since_complaints == TRUE] <- 1  
data$time_since_technical_problems_missing[data$time_since_technical_problems == TRUE] <- 1 

test$plan_type_bring[test$plan_type == 'bring'] <- 1  
test$plan_type_buy[test$plan_type == 'buy'] <- 1 
test$plan_type_rent[test$plan_type == 'rent'] <- 1 
test$time_since_data_overage_missing[test$time_since_data_overage == TRUE] <- 1  
test$time_since_overage_missing[test$time_since_overage == TRUE] <- 1  
test$time_since_complaints_missing[test$time_since_complaints == TRUE] <- 1  
test$time_since_technical_problems_missing[test$time_since_technical_problems == TRUE] <- 1 
#===== missing values
data = data %>%
  mutate(time_since_data_overage
         = replace(time_since_data_overage,
                   is.na(time_since_data_overage),
                   120)) #120 a value outside of the range
data = data %>%
  mutate(time_since_overage
         = replace(time_since_overage,
                   is.na(time_since_overage),
                   120))
data = data %>%
  mutate(time_since_complaints
         = replace(time_since_complaints,
                   is.na(time_since_complaints),
                   80)) #80 a value outside of the range
data = data %>%
  mutate(time_since_technical_problems
         = replace(time_since_technical_problems,
                   is.na(time_since_technical_problems),
                   80))
#test set
test = test %>%
  mutate(time_since_data_overage
         = replace(time_since_data_overage,
                   is.na(time_since_data_overage),
                   120))
test = test %>%
  mutate(time_since_overage
         = replace(time_since_overage,
                   is.na(time_since_overage),
                   120))
test = test %>%
  mutate(time_since_complaints
         = replace(time_since_complaints,
                   is.na(time_since_complaints),
                   80))
test = test %>%
  mutate(time_since_technical_problems
         = replace(time_since_technical_problems,
                   is.na(time_since_technical_problems),
                   80))
data=data %>% 
  mutate(MISS_phone_price=is.na(phone_price),MISS_voice_minutes=is.na(voice_minutes))
test=test %>% 
  mutate(MISS_phone_price=is.na(phone_price),MISS_voice_minutes=is.na(voice_minutes))

data$MISS_phone_price = as.integer(data$MISS_phone_price)
test$MISS_phone_price = as.integer(test$MISS_phone_price)

data$MISS_voice_minutes = as.integer(data$MISS_voice_minutes)
test$MISS_voice_minutes = as.integer(test$MISS_voice_minutes)

data = data %>%
  mutate(phone_price
         = replace(phone_price,
                   is.na(phone_price),
                   0))

data = data %>%
  mutate(voice_minutes
         = replace(voice_minutes,
                   is.na(voice_minutes),
                   0))
test = test %>%
  mutate(phone_price
         = replace(phone_price,
                   is.na(phone_price),
                   0))
test = test %>%
  mutate(voice_minutes
         = replace(voice_minutes,
                   is.na(voice_minutes),
                   0))

#===== train-validation sets====
trainID=sample(1:860000,588000)         
train=data[trainID,]
valid=data[-trainID,]

#============== multiple imputation with miss Forest=====
library(missForest)
imputed <- missForest(train)
imputed_imp  <- imputed$ximp
imputed_imp %>% is.na() %>% colSums()
imp_res$OOBerror

#============== multiple imputation with rpart =====
models=lapply(paste(names(train),"~.-id-family_id-churn_in_12-unique_id-promo-voice_minutes"),as.formula)
imputation_tree=lapply(models,rpart,data=train)

for(i in c(6,33)){
  train[[i]][is.na(train[[i]])]=predict(imputation_tree[[i]],newdata=train[is.na(train[[i]]),],type="vector")
  valid[[i]][is.na(valid[[i]])]=predict(imputation_tree[[i]],newdata=valid[is.na(valid[[i]]),],type="vector")
  #test[[i]][is.na(test[[i]])]=predict(imputation_tree[[i]],newdata=test[is.na(test[[i]]),],type="vector")
}

write.csv(train,"/Users/sabad/Downloads/CreditGameData/Train1.csv", row.names = FALSE)
write.csv(valid,"/Users/sabad/Downloads/CreditGameData/Valid1.csv", row.names = FALSE)
write.csv(test,"/Users/sabad/Downloads/CreditGameData/Test1.csv", row.names = FALSE)
#---------------missForest--------------

library(missForest)
registerDoParallel(cores=4)
imp_res <- missForest(train,parallelize = "forests")
nhanes_imp  <- imp_res$ximp
nhanes_imp %>% is.na() %>% colSums()
imp_res$OOBerror

imp_res <- missForest(train, variablewise = TRUE, parallelize = "variables")
imp_res$OOBerror

write.csv(imp_res$ximp,"/Users/sabad/Downloads/CreditGameData/TrainrfnoNA14000.csv", row.names = FALSE)

library(missForest)
registerDoParallel(cores=4)
imputed = read.csv("/Users/sabad/Downloads/CreditGameData/TrainrfnoNA14000.csv")
#split the data
training.sample <- imputed$churn_in_12 %>% createDataPartition(p=0.8, list=FALSE)
valid <- imputed[-training.sample,]
train <- imputed[training.sample,]


dataforimp = subset(train, select = c(id,family_id,unique_id,total_technical_problems,phone_balance,
                                      plan_type,base_monthly_rate_phone,total_complaints,total_voice_consumption,
                                      total_data_consumption,complaints,text_consumption,
                                      base_monthly_rate_plan,data_consumption,data,total_text_consumption))

str(dataforimp)
# -------------------imputation mice----------------------
#install.packages(mice')
library(mice)
require(parallel)
detectCores()
cl <- parallel::makeCluster(detectCores())
doParallel::registerDoParallel(cl)

tempData <- mice(data = dataforimp,m=1,maxit=40,method='cart',seed=435)
parmice(seed=,)
densityplot(tempData)
#tempData <- parlmice(data = dataforimp,m=2,maxit=50,meth='pmm',cluster.seed=245435,n.core =4, n.imp.core = 2)

completedData <- complete(tempData)
dataforimp$phone_price = completedData$phone_price


write.csv(dataforimp,"/Users/sabadaftari/Downloads/completedTrain.csv", row.names = FALSE)

completedData = read.csv("/Users/sabadaftari/Downloads/completedTrain.csv")
summary(completedData)
train$phone_price = completedData$phone_price
write.csv(train,"/Users/sabadaftari/Downloads/completedTrainn.csv", row.names = FALSE)

# -------------------Train Imputation----------------------
completedData$churn_in_12 = as.factor(completedData$churn_in_12)
valid$churn_in_12 = as.factor(valid$churn_in_12)
summary(trainforimp)

test = read.csv("/Users/sabadaftari/Downloads/completedTest.csv")

trControl <- trainControl(method = "cv",
                          number = 5,
                          search = "grid")
tuneGrid <- expand.grid(.mtry = 2)
bow <- train(churn_in_12~total_technical_problems+
               plan_type+base_monthly_rate_phone+total_complaints+
               total_voice_consumption+total_data_consumption+
               complaints+text_consumption+base_monthly_rate_plan+
               data_consumption+data+total_text_consumption
             ,data=dataforimp,
             method = "rf",
             metric = "Accuracy",
             tuneGrid = tuneGrid,
             trControl = trControl,
             importance = TRUE,
             nodesize = 2,
             ntree = 250,
             maxnodes = 12)
#gbm

fitControl <- trainControl(method = "repeatedcv", number = 2, repeats = 2)
tic()
set.seed(42)
gbm_model_voters <- train(churn_in_12 ~ .-id-unique_id-family_id,data = dataforimp,
                          method = "gbm",
                          trControl = fitControl,
                          verbose = FALSE)
dataforimp$unique_id = is.factor(dataforimp$unique_id)
pbow=predict(gbm_model_voters ,test,type='prob')[,2]
pbow
#--------------------- Valid Imputation---------------------
valid= subset(valid,select = c(id,family_id,unique_id,total_technical_problems,phone_balance,churn_in_12,
                               plan_type,base_monthly_rate_phone,total_complaints,total_voice_consumption,
                               phone_price,total_data_consumption,complaints,text_consumption,
                               base_monthly_rate_plan,data_consumption,data,total_text_consumption))
tempValid <- mice(data = valid,
                  pred=quickpred(valid,
                                 include= c('total_technical_problems',
                                            'total_complaints','complaints','text_consumption',
                                            'total_voice_consumption','phone_price','total_data_consumption',
                                            'data_consumption','data','total_text_consumption'),
                                 exclude= c('id', 'family_id','unique_id','plan_type','base_monthly_rate_plan'
                                            ,'churn_in_12','base_monthly_rate_phone')),m=2,maxit=100,meth='pmm',seed=435)
completedValid <- complete(tempValid,2)
densityplot(tempValid)
write.csv(completedValid,"/Users/sabadaftari/Downloads/completedValid.csv", row.names = FALSE)

#--------------------- Test Imputation---------------------

testt= subset(test,select = c(id,family_id,unique_id,unique_family,total_technical_problems,phone_balance,
                              plan_type,base_monthly_rate_phone,total_complaints,total_voice_consumption,
                              phone_price,total_data_consumption,complaints,text_consumption,
                              base_monthly_rate_plan,data_consumption,data,total_text_consumption))
tempTest <- mice(data = testt,m=1,maxit=10,meth='pmm',seed=435)
completedTest <- complete(tempTest)
test$phone_price = completedTest$phone_price
densityplot(tempTest)
write.csv(test,"/Users/sabadaftari/Downloads/completedTest.csv", row.names = FALSE)
str(completedTest)


export3 = cbind(p,completedTest)[sort.list(p,decreasing=TRUE),]
export3 = slice_head(export3,n = 3/10 * 1000000)
write.csv(export3[c("unique_family")],"/Users/sabad/Downloads/CreditGameData/CompletedTest.csv", row.names = FALSE)
