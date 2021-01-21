library(caret)
library(xgboost)

#---------------models------------------

data = read.csv("/Users/sabadaftari/Downloads/TrainImputed.csv")
valid = read.csv("/Users/sabad/Downloads/CreditGameData/Valid1.csv")

#====================CrossValidation Random Forest ====================

trControl <- trainControl(method = "repeatedcv",
                          number = 3,
                          search = "grid")
set.seed(1234)

train$churn_in_12 = as.factor(train$churn_in_12)
valid$churn_in_12 = as.factor(valid$churn_in_12)


#search best mtry = 1 selected
tuneGrid <- expand.grid(.mtry = c(1: 6))
rf_mtry <- train(churn_in_12~data_consumption+text_consumption+total_data_consumption+
                   total_text_consumption+total_voice_consumption+total_technical_problems+
                   phone_balance+base_monthly_rate_phone+phone_price+age
                 ,data=train,
                 method = "rf",
                 metric = "Accuracy",
                 trControl = trControl,
                 tuneGrid=tuneGrid,
                 importance = TRUE,
                 ntree = 50,
                 nodesize=14)
print(rf_mtry)
rf_mtry$bestTune$mtry
max(rf_mtry$results$Accuracy)
best_mtry <- rf_mtry$bestTune$mtry 
#search the best maxnodes
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = 1)
for (maxnodes in c(20: 30)) {
  set.seed(1234)
  rf_maxnode <- train(churn_in_12~data_consumption+text_consumption+total_data_consumption+
                        total_text_consumption+total_voice_consumption+total_technical_problems+
                        phone_balance+base_monthly_rate_phone+phone_price+age
                      ,data=train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 150)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
maxnodes=12
winner=27
#search the best ntrees
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {#250 selected
  set.seed(5678)
  rf_maxtrees <- train(churn_in_12~data_consumption+text_consumption+total_data_consumption+
                         total_text_consumption+total_voice_consumption+total_technical_problems+
                         phone_balance+base_monthly_rate_phone+phone_price+age
                       ,data=train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 2,
                       maxnodes = 27,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)


#final tuned model-------------------------------
train$churn_in_12 = as.factor(train$churn_in_12)
valid$churn_in_12 = as.factor(valid$churn_in_12)
str(valid)

trControl <- trainControl(method = "cv",
                          number = 5,
                          search = "grid")
tuneGrid <- expand.grid(.mtry = 2)
bow <- train(churn_in_12~data_consumption+text_consumption+total_data_consumption+
               total_text_consumption+total_voice_consumption+total_technical_problems+
               phone_balance+base_monthly_rate_phone+phone_price+age
             ,data=train,
             method = "rf",
             metric = "Accuracy",
             tuneGrid = tuneGrid,
             trControl = trControl,
             importance = TRUE,
             nodesize = 2,
             ntree = 250,
             maxnodes = 12)

pboww=predict(bow,valid,type = 'prob')
str(valid)
valid$churn_in_12 = as.integer(valid$churn_in_12)
train$churn_in_12 = as.integer(train$churn_in_12)
valid = subset(valid, select = -c(voice_minutes))
str(valid)
pboww
roc(valid$churn_in_12,pboww,col="red")$AUC
as.matrix(bow$importance/max(bow$importance))
confusion(valid$churn_in_12,predict(bow,valid))
