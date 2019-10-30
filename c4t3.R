# Required
library(doParallel)
library(plotly)
library(corrplot)#course2task3
library(caret)#nearZeroVar() ,rfe() function (recursive feature elimination), PREPROCESS()
library(dplyr)#to select only specific columns, recode()
library(C50)
library(e1071) #svm
library(kknn)
library(randomForest)
library(xgbLinear) #gradient boosting


# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(2)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 



set.seed(100)
 
iphoneDF <- read.csv("~/Data Analytics Certification/Course 4/Task3/iphone_smallmatrix_labeled_8d.csv",
                         header=TRUE,stringsAsFactors=FALSE) 
View(iphoneDF)
str(iphoneDF)
summary(iphoneDF)
summary(iphoneDF$ios)
summary(iphoneDF$iphonecampos)
summary(iphoneDF$galaxtdis)
summary(iphoneDF$htcperunc )

str(iphoneDF$ios)

is.na(iphoneDF)
  
plot_ly(iphoneDF, x= ~iphoneDF$iphonesentiment, type='histogram')
#Tip: While printing your cor() output you might see a message about reaching "max print" 
#and a truncated matrix. You can solve this by increasing "max print" with options(). 
#The options() function lets you control a wide variety of global options.

options(max.print=1000000)
corrData <- cor(iphoneDF)
corrData
## 1. Checking to see to remove features with high correlation with Y(dependent)
corrplot(corrData)
corrplot(corrData,method = "pie",number.cex = .5)
corrplot(corrData,method = "number",number.cex = .4)

##2.Examine Feature Variance 
#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: 
##frequency ratio, percentage unique, zero variance and near zero variance 
iphoneDf_nzv<-iphoneDF
nzvMetrics <- nearZeroVar(iphoneDF, saveMetrics = TRUE)
nzvMetrics

#Review your table. Are there features that have zero variance? 
#Near zero variance? Let's use nearZeroVar() again to create an index of near zero variance features.
#The index will allow us to quickly remove features. 
# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(iphoneDF, saveMetrics = FALSE) 
nzv
iphoneDf_nzv
# create a new data set and remove near zero variance features
iphoneDf_nzv <- iphoneDF[,-nzv]
str(iphoneDf_nzv)

#iphoneDf_nzv_trial<-select(iphoneDf_nzv,iphone,2)
#summary(iphoneDf_nzv_trial)
#iphoneDf_nzv<-select(iphoneDf_nzv,iphone,2,5,8,18,23,28,33,38,43,48) #can mention the columnname or no
#str(iphoneDf_nzv)

#3. Recursive Feature Elimination 
#RFE is a form of automated feature selection. Caret's rfe() function with random forest 
#will try every combination of feature subsets and return a final list of recommended features. 
#RFE does not use the outcome so it must be removed from the data set before implementation and 
#then added back in before modeling. 
# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- iphoneDF[sample(1:nrow(iphoneDF), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)
# Get results
rfeResults
predictors(rfeResults)  #The predictors function can be used to get a text string of variable names 
                         #that were picked in the final model

# create new data set with rfe recommended features
iphoneDF_rfe1 <- iphoneDF[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneDF_rfe1$iphonesentiment <- iphoneDF$iphonesentiment
# review outcome
str(iphoneDF_rfe1)

View(iphoneDF_rfe1)
# Plot results
plot(rfeResults, type=c("g", "o"))


#----------------------------List of DFS---------------------------------------------------------
str(iphoneDF)
str(iphoneDf_nzv)
str(iphoneDF_rfe1)

#--------------after preprocessing , convert Y variable to Factor------------------------------------

iphoneDF$iphonesentiment<-as.factor(iphoneDF$iphonesentiment)


#--------define an 75%/25% train/test split of the dataset--------------------------------------------
inTraining <- createDataPartition(iphoneDF$iphonesentiment, p = .70, list = FALSE)
training <- iphoneDF[inTraining,]
testing <- iphoneDF[-inTraining,]

#-------------------10 fold cross validation-----------------------------------------------------
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#--------C5model with10-fold cross validation Automatic Tuning Grid with a tuneLength of 2------------------
c5_t2 <- train(iphonesentiment~., data = training, method ="C5.0", trControl=fitControl, tuneLength = 2) #not much time to run
svmFit <- train(iphonesentiment~., data = training, method = "svmLinear2", trControl=fitControl, tuneLength = 2) #2.38 pm #done before 2:57
rfFit <- train(iphonesentiment~., data = training, method = "rf", trControl=fitControl, tuneLength = 2)#2.57
gbFit <- train(iphonesentiment~., data = training, method = "xgbLinear", trControl=fitControl, tuneLength = 2)
kknnFit <- train(iphonesentiment~., data = training, method = "kknn", trControl=fitControl, tuneLength = 2)


#--------------------Predict --------------------------------------------------------------------
Pred_svm<-predict(svmFit,testing)
Pred_rf<-predict(rfFit,testing)
Predict_gb<-predict(gbFit,testing)
Predict_c5<-predict(c5_t2,testing)
Predict_kknn<-predict(kknnFit,testing)


#--------------------Output ------------------------------------------------------------------
Output_c5<-confusionMatrix(Predict_c5, testing$iphonesentiment)
Output_rf<-confusionMatrix(Pred_rf, testing$iphonesentiment)
Output_gb<-confusionMatrix(Predict_gb, testing$iphonesentiment)
Output_svm<-confusionMatrix(Pred_svm, testing$iphonesentiment)
Output_kknn<-confusionMatrix(Predict_kknn, testing$iphonesentiment)

Output_c5 #  A: 0.7676 K : 0.5465  II
Output_rf #  A: 0.7566 K : 0.5337 III
Output_gb #  A: 0.7676 K : 0.5485  I
Output_svm #  A: 0.708 K : 0.406   IV
Output_kknn #  A: 0.3165 K : 0.1531  V

#-------------------accuracy and kappa ONLY & other metrices-------------------------------------------------------------------
Ak_c5<-as.matrix(Output_c5,what="overall") 
Ak_c5
SSP_c5<-as.matrix(Output_c5, what = "classes")  # S,S,p and other metrices
SSP_c5

Ak_rf<-as.matrix(Output_rf,what="overall") 
Ak_rf
SSP_rf<-as.matrix(Output_rf, what = "classes")  # S,S,p and other metrices
SSP_rf

Ak_gb<-as.matrix(Output_gb,what="overall") 
Ak_gb
SSP_gb<-as.matrix(Output_gb, what = "classes")  # S,S,p and other metrices
SSP_gb

Ak_svm<-as.matrix(Output_svm,what="overall") 
Ak_svm
SSP_svm<-as.matrix(Output_svm, what = "classes")  # S,S,p and other metrices
SSP_svm

Ak_kknn<-as.matrix(Output_kknn,what="overall") 
Ak_kknn
SSP_kknn<-as.matrix(Output_kknn, what = "classes")  # S,S,p and other metrices
SSP_kknn
#----------- Resampling--------------------------------------------------------------------

ModelData <- resamples(list(C50 = c5_t2,kknn = kknnFit, RF = rfFit,xgb = gbFit,svm=svmFit ))
summary(ModelData)

?postResample()
postResample(Predict_gb,testing$iphonesentiment)




#-------------- apply top 3 perfomering models in the other datasets -----------------------


str(iphoneDf_nzv)
str(iphoneDF_rfe1)

inTraining_nzv <- createDataPartition(iphoneDf_nzv$iphonesentiment, p = .70, list = FALSE)
training_nzv <- iphoneDF[inTraining_nzv,]
testing_nzv<- iphoneDF[-inTraining_nzv,]

inTraining_rfe1 <- createDataPartition(iphoneDF_rfe1$iphonesentiment, p = .70, list = FALSE)
training_rfe1<- iphoneDF[inTraining_rfe1,]
testing_rfe1<- iphoneDF[-inTraining_rfe1,]

gbFit_nzv<- train(iphonesentiment~., data = training_nzv, method = "xgbLinear", trControl=fitControl, tuneLength = 2)
c5_t2_nzv<- train(iphonesentiment~., data = training_nzv, method ="C5.0", trControl=fitControl, tuneLength = 2)
rfFit_nzv<- train(iphonesentiment~., data = training_nzv, method = "rf", trControl=fitControl, tuneLength = 2)

gbFit_rfe1<- train(iphonesentiment~., data = training_rfe1, method = "xgbLinear", trControl=fitControl, tuneLength = 2)
c5_t2_rfe1<- train(iphonesentiment~., data = training_rfe1, method ="C5.0", trControl=fitControl, tuneLength = 2)
rfFit_rfe1<- train(iphonesentiment~., data = training_rfe1, method = "rf", trControl=fitControl, tuneLength = 2)


Pred_gbFit_nzv<-predict(gbFit_nzv,testing_nzv)
Pred_c5_t2_nzv<-predict(c5_t2_nzv,testing_nzv)
Pred_rfFit_nzv<-predict(rfFit_nzv,testing_nzv)

Pred_gbFit_rfe1<-predict(gbFit_rfe1,testing_rfe1)
Pred_c5_t2_rfe1<-predict(c5_t2_rfe1,testing_rfe1)
Pred_rfFit_rfe1<-predict(rfFit_rfe1,testing_rfe1)


Output_gbFit_nzv<-confusionMatrix(Pred_gbFit_nzv, testing_nzv$iphonesentiment)
Output_c5_nzv<-confusionMatrix(Pred_c5_t2_nzv, testing_nzv$iphonesentiment)
Output_rfFit_nzv<-confusionMatrix(Pred_rfFit_nzv, testing_nzv$iphonesentiment)

Output_gbFit_rfe1<-confusionMatrix(Pred_gbFit_rfe1, testing_rfe1$iphonesentiment)
Output_c5_rfe1<-confusionMatrix(Pred_c5_t2_rfe1, testing_rfe1$iphonesentiment)
Output_rfFit_rfe1<-confusionMatrix(Pred_rfFit_rfe1, testing_rfe1$iphonesentiment)

Output_gbFit_nzv #Acc-0.7723 ,Kappa -0.5611
Output_c5_nzv #Acc-0.7746 ,kappa - 0.5637
Output_rfFit_nzv #Acc -0.76067, Kappa-0.5451
Output_gbfit_rfe1
Output_c5_rfe1 #Acc- 0.7738, Kappa - 0.5654
Output_rfFit_rfe1 #Acc- 0.7661 ,Kappa - 0.5616

# ---------  Feature Engineering again -----------------------------------------------------------
 # -----------------  1. Altering Dependent Variable-----------------------------------------------------
 #------------------ 2. Principal Component Analysis--------------------------------------------------


# create a new dataset that will be used for recoding sentiment
iphoneRC <- iphoneDF

# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(iphoneRC)
str(iphoneRC)
summary(iphoneRC$iphonesentiment)

# make iphonesentiment a factor
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)

# model using the best learner
inTraining_RC <- createDataPartition(iphoneRC$iphonesentiment, p = .70, list = FALSE)
training_RC<- iphoneRC[inTraining_RC,]
testing_RC<- iphoneRC[-inTraining_RC,]

 # ---------- train, test and predict ---------------

gbFit_RC<- train(iphonesentiment~., data = training_RC, method = "xgbLinear", trControl=fitControl, tuneLength = 2)
c5_t2_RC<- train(iphonesentiment~., data = training_RC, method ="C5.0", trControl=fitControl, tuneLength = 2)

Pred_gbFit_RC<-predict(gbFit_RC,testing_RC)
Pred_c5_t2_RC<-predict(c5_t2_RC,testing_RC)

Output_gbFit_RC<-confusionMatrix(Pred_gbFit_RC, testing_RC$iphonesentiment)
Output_c5_RC<-confusionMatrix(Pred_c5_t2_RC, testing_RC$iphonesentiment)
#----------------------------------pca-------------------------------------------------------------------

# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(training[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)
preprocessParams_V75<- preProcess(training[,-59], method=c("center", "scale", "pca"), thresh = 0.75)
print(preprocessParams_V75)

#We now need to apply the PCA model, create training/testing and add the dependant variable.
# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, training[,-59])
train.pca75 <- predict(preprocessParams_V75, training[,-59])


# add the dependent to training
train.pca$iphonesentiment <- training$iphonesentiment
train.pca75$iphonesentiment <- training$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, testing[,-59])
test.pca75 <- predict(preprocessParams_V75, testing[,-59])

# add the dependent to training
test.pca$iphonesentiment <- testing$iphonesentiment
test.pca75$iphonesentiment <- testing$iphonesentiment

# inspect results
str(train.pca)
str(test.pca)
gbFit_PCS<- train(iphonesentiment~., data = train.pca, method = "xgbLinear", trControl=fitControl, tuneLength = 2)
gbFit_PCA<-gbFit_PCS
c5_PCA<- train(iphonesentiment~., data = train.pca, method ="C5.0", trControl=fitControl, tuneLength = 2)

Pred_gbFit_PCA<-predict(gbFit_PCA,test.pca)
Pred_C5_PCA<-predict(c5_PCA,test.pca)

Output_gbFit_PCA<-confusionMatrix(Pred_gbFit_PCA, test.pca$iphonesentiment)
Output_C5_PCA<-confusionMatrix(Pred_C5_PCA, test.pca$iphonesentiment)

#-------- TESTING FOR 0.75-----------------------
gbFit_PCA75<- train(iphonesentiment~., data = train.pca75, method = "xgbLinear", trControl=fitControl, tuneLength = 2)
c5_PCA75<- train(iphonesentiment~., data = train.pca75, method ="C5.0", trControl=fitControl, tuneLength = 2)

Pred_gbFit_PCA75<-predict(gbFit_PCA75,test.pca75)
Pred_C5_PCA75<-predict(c5_PCA75,test.pca75)

Output_gbFit_PCA75<-confusionMatrix(Pred_gbFit_PCA75, test.pca75$iphonesentiment)
Output_C5_PCA75<-confusionMatrix(Pred_C5_PCA75, test.pca75$iphonesentiment)

#----------- Model and PostResample----------------------------------------------------------
ModelData <- resamples(list(C50 = c5_PCA75,xgb = gbFit_PCA75 ))

summary(ModelData)
postResample(Pred_C5_PCA75,test.pca75$iphonesentiment)


#---------- writeoutput---------------------------------------------------------------------------------

output <- cbind(iphoneDF, Pred_gbFit_RC)
write.csv(output, file="predicted_new_productattributes.csv", row.names = FALSE)



#------------------------------------------------------------------------------------
# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)