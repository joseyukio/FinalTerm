
library(plyr)
library(data.table)
library(dplyr)
library(bit64)
library(caret)
library(nnet)
library(randomForest)
library(e1071)
library(ROCR)
library(DMwR) ## SMOTE
library(gbm)
library(adabag)
library(mboost)
library(ada)
library(Metrics)

#####################################################################################
## Prediction
#############################
imp.df<-read.csv("data/yukiodb6.txt",header=TRUE,sep=",")
half_df <- nrow(imp.df)/2 
a.df <- imp.df[1:half_df, ] 
b.df <- imp.df[(half_df + 1) : nrow(imp.df), ]
#a.df <- imp.df[1:100000, ] 
#b.df <- imp.df[100001:200000, ] 
rm(imp.df)

## using GBM
system.time(gb.model <- gbm(clicked ~ ., data = a.df, n.trees = 4000, cv.folds = 2, distribution = "gaussian"))
#system.time(gb.model <- gbm(clicked ~ ., data = a.df, n.trees = 2000, cv.folds = 2, distribution = "bernoulli"))
#saveRDS(gb.model, file = "gb.model.tree.4000.gaussian.cv.2.RDS")
#gb.model <- gbm(clicked ~ InvId + cityc + gender + hMod + others, data = a.df, n.trees = 2000, cv.folds = 2, distribution = "gaussian")
#gb.model <- readRDS(file = "gb.model.RDS")
best.iter <- gbm.perf(gb.model,method="cv")
print(pretty.gbm.tree(gb.model,1))
gb.predicted <- predict(gb.model, b.df, type = "response", n.trees = best.iter)
gb.predicted.labels <- ifelse(gb.predicted > 0.2, 1, 0)
gb.confm <- confusionMatrix(gb.predicted.labels,b.df$clicked)
gb.confm
## ROCR
pred <- prediction(gb.predicted.labels, b.df$clicked)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(5))
abline(0,1)

#


# names(mat)
# x<-mat[1:9]
# y<-mat[11:20]
# x<-cbind(x,y)
# mat2<-data.frame(x)
# rm(mat)
# rm(x)
# rm(y)
# names(mat2)
# 
# imp.df<-mat2
# rm(mat2)

# half_df <- nrow(imp.df)/2 
# a.df <- imp.df[1:half_df, ] 
# b.df <- imp.df[(half_df + 1) : nrow(imp.df), ]
a.df <- imp.df[1:100000, ] 
b.df <- imp.df[100001:200000, ] 
#rm(imp.df)

# #### 
# set.seed(999)
# #fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)
# system.time(glmboost.model <- train(as.factor(clicked) ~ agecat, data = a.df, method = "glmboost"))
# glmboost.pred <- predict(glmboost.model, newdata =  b.df, type = "prob")[,2]
# 
# glmboost.pred <- predict(glmboost.model, probability = TRUE, type="response", newdata = b.df)
# glmboost.predicted.labels <- ifelse(glmboost.pred > 0.5, 1, 0)
# logit.confm <- confusionMatrix(glmboost.predicted.labels,b.df$clicked)
# logit.confm








## Using ada boosting
adabag.model <- boosting(clicked ~ ., data = a.df, control=rpart.control(type = "response"))
saveRDS(adabag.model, file = "adabag.model.RDS")
adabag.pred <- predict.boosting(adabag.model, newdata= b.df)
adabag.pred$confusion
adabag.pred$error

## Using package ada
set.seed(1)
system.time(ada.model <- ada(clicked ~ ., data = a.df, max.iter = 100, nu = 0.6, bag.frac = 0.1))
ada.pred <- predict(ada.model, newdata= b.df)
ada.confm <- confusionMatrix(ada.pred, b.df$clicked)
ada.confm
varplot(ada.model)
plot(ada.model)
saveRDS(ada.model, file = "ada.model.RDS")


## Using rpart
rpart.model <- rpart(clicked ~., data = a.df)
rpart.pred <- predict(rpart.model, newdata = b.df)
rpart.predicted.labels <- ifelse(rpart.pred > 0.2, 1, 0)
rpart.confm <- confusionMatrix(rpart.predicted.labels,b.df$clicked)
rpart.confm

# using glmboost
system.time(glmboost.model <- glmboost(as.factor(clicked) ~ ., data = a.df, family = binomial("logit")))
glmboost.pred <- predict(glmboost.model, probability = TRUE, type="response", newdata = b.df)
glmboost.predicted.labels <- ifelse(glmboost.pred > 0.5, 1, 0)
logit.confm <- confusionMatrix(glmboost.predicted.labels,b.df$clicked)
logit.confm

## Using Logistic Regression
#imp.df <- readRDS(file = "final.imp.df.RDS")
# imp.df <- readRDS(file = "output/final.imp.df.1percent.RDS")
# imp.df <- read.csv(file = "data/yukiodb4.txt")
# half_df <- nrow(imp.df)/2 
# a.df <- imp.df[1:half_df, ] 
# b.df <- imp.df[(half_df + 1) : nrow(imp.df), ]


# a.df <- imp.df[1:100000, ] 
# b.df <- imp.df[100001:200000, ] 
## remove outliers
# a.df <- filter(a.df, age < 100)
# b.df <- filter(b.df, age < 100)

rm(imp.df)

# system.time(mylogit <- glm(as.factor(clicked) ~ InvId + adType + hBrand + hMod + siteId + adZone + gender +  agecat + os + entertainment + games_app + music + books + fashion + lifestyle + business + sports + news, data = a.df, family = binomial("logit")))

system.time(mylogit <- glm(as.factor(clicked) ~ ., data = a.df, family = binomial("logit")))
mylogit.pred <- predict.glm(mylogit, probability = TRUE, type="response", newdata = b.df)
mylogit.predicted.labels<-ifelse(mylogit.pred>0.5,1,0)
logit.confm <- confusionMatrix(mylogit.predicted.labels,b.df$clicked)
logit.confm
## Check the actual distribution
table(b.df$clicked)
## ROC
pred <- prediction(mylogit.predicted.labels, b.df$clicked)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))


#############
# Naive Bayes #
#############
# remove city code
# mat<-read.csv("data/yukiodb6.txt",header=TRUE,sep=",")
# names(mat)
# x<-mat[1:9]
# y<-mat[11:20]
# x<-cbind(x,y)
# mat2<-data.frame(x)
# rm(mat)
# rm(x)
# rm(y)
# names(mat2)
# 
# imp.df<-mat2
# rm(mat2)
# half_df <- nrow(imp.df)/2 
# a.df <- imp.df[1:half_df, ] 
# b.df <- imp.df[(half_df + 1) : nrow(imp.df), ]
# 
# a.df <- imp.df[1:100000, ] 
# b.df <- imp.df[100001:200000, ] 
set.seed(1)
mybayes<-naiveBayes(as.factor(clicked) ~., data=a.df)
mybayes.pred<-predict(mybayes, b.df, type="raw")
#mybayes.predicted.labels<-ifelse(mybayes.pred>0.5,1,0)
mybayes.predicted.labels <- ifelse(mybayes.pred[, 1] > mybayes.pred[, 2], 0, 1)
mybayes.confm <- confusionMatrix(mybayes.predicted.labels, b.df$clicked)
mybayes.confm

## Check the actual distribution
table(b.df$clicked)
## ROCR
pred <- prediction(mybayes.predicted.labels, b.df$clicked)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(5))




#### using NN
system.time(nn.model <- multinom(as.factor(clicked) ~ .,  data = a.df))
nn.pred <- predict(nn.model,newdata = b.df)
nn.confm <- confusionMatrix(nn.pred, b.df$clicked)
nn.confm
## Check the actual distribution
table(b.df$clicked)

## ROC
pred <- prediction(nn.pred, b.df$clicked)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))

#### using Random Forest
system.time(rf.model <- randomForest(as.factor(clicked) ~ InvId + adType + siteId + adZone + gender +  agecat + os + entertainment + games_app + music + books + fashion + lifestyle + business + sports + news, data = a.df, importance = TRUE))
rf.predicted <- predict(rf.model, b.df)
rf.confm <- confusionMatrix(rf.predicted, b.df$clicked)
rf.confm
saveRDS(rf.model, file = "rf.model.RDS")


## usinf SVM
system.time(svm.a.model <- svm(as.factor(clicked) ~ gender,  data = a.df ))
svm.b.predicted <- predict(svm.a.model, b.df)
svm.b.confm <- confusionMatrix(svm.b.predicted, b.df$clicked)
svm.b.confm

## PCA
pca.a <- prcomp(a.df[, 2:19], scale = TRUE)

## SMOTE
## Run SMOTE
p.o <- 100
p.u <- 10
a <- SMOTE(clicked ~ ., data = a.df, perc.over = p.o, perc.under = p.u, k=5)




################################################################################

## Function to read each click file. Keep only the selected columns
read_click <- function(filename){
        print(filename)
        ret <- fread(input = filename, sep = ";", colClasses = "character", select = c(4))
        ret
}

## Collect all click logs into a single file
path_click = "data/logs/"
filenames <- list.files(path_click, pattern = "*click*", all.files = FALSE, full.names = TRUE, recursive = TRUE, ignore.case = FALSE)
click.df <- ldply(filenames, read_click)
colnames(click.df) <- c("corrId")
head(click.df)

## Function to read each impression file. Keep only the selected columns
read_imp <- function(filename){
        print(filename)
        ret <- fread(input = filename, sep = ";", colClasses = "character", select = c(1, 5, 10, 11, 15, 16, 18, 19, 36, 44, 45, 48, 69, 70) )
        ret
}

## Function to read data frame from file and bind it.
read_bind <- function(filename){
        print(filename)
        tmp <- readRDS(filename)
        imp.df <- rbind(imp.df, tmp)
}

## Indicate the folders to be processed.
folder = list(132, 135, 136, 137, 138, 139, 140, 141,142,146,147,148,149,150,151,152)

## Indicate the fracton of the impression logs that will be sampled.
## Only impression log si sampled. Clicks are retained.
sample_fraction = 0.01

for (i in folder) {
        
        ## Set the path where the impression logs are stored
        path_imp = paste("data/logs/", i, sep = "") ## In this example only the logs from folder 132 is read
        ## List only the impression log file names
        filenames <- list.files(path_imp, pattern = "*impression*", all.files = FALSE, full.names = TRUE, recursive = TRUE, ignore.case = FALSE)
        ## Read the impression files into imp.df data frame
        imp.df <- ldply(filenames, read_imp)
        ## Set the column names
        colnames(imp.df) <- c("time","corrId","InvId","adType","hBrand","hMod", "siteId", "adZone", "keyword", "gender", "age", "city", "msisdn", "os" )
        
        ## Get only the impression with clicks based on the correlation ID
        imp.with.click <- filter(imp.df, corrId %in% click.df$corrId)
        
        ## Sample the impression data frame. Size is the percentage/fraction
        imp.df <- sample_frac(imp.df, size = sample_fraction )
        ## Add the impressions with clicks. To make sure impressions with clicks that
        ## were discarded due to sample are in the file.
        imp.df <- rbind(imp.df, imp.with.click)
        ## Remove possible duplicated rows due to re-insertion of impressions with clicks
        imp.df <- distinct(imp.df)
        
        ## Add the column "clicked". This is our label.
        imp.df$clicked <- ifelse (imp.df$corrId %in% click.df$corrId, 1, 0)
        ## Column corrId is not necessary anymore. Drop it.
        imp.df <- imp.df[-2]
        ## Save the result into a file
        saveRDS(imp.df, file = paste("output/imp.df.", i, ".RDS", sep = ""))
}

## Read all data files 
imp.df <- data.frame()
path_output = "output"
filenames <- list.files(path = path_output, pattern = "imp.df*", all.files = FALSE, full.names = TRUE, recursive = TRUE, ignore.case = FALSE)
imp.df <- ldply(filenames, read_bind)
## Save the final result
saveRDS(imp.df, file = paste("output/final.imp.df.1percent.RDS", sep = ""))

## Since we sampled the impressions we have to consider this when calculating the
## CTR, if needed.








