
set.seed(3606)
library(caret); library(rpart); library(rpart.plot); library(RColorBrewer)
library(rattle); library(randomForest); library(knitr); library(gbm)
Url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(Url_train), na.strings=c("NA", "#DIV/0!", ""))
testing <- read.csv(url(Url_test), na.strings=c("NA", "#DIV/0!", ""))
# dim(training); dim(testing)

nzv <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, nzv$nzv==FALSE]

trainingTEMP <- training
for(i in 1:length(training)) {
  if( sum( is.na( training[, i] ) ) /nrow(training) >= .55) {
    for(j in 1:length(trainingTEMP)) {
      if( length( grep(names(training[i]), names(trainingTEMP)[j]) ) == 1)  {
        trainingTEMP <- trainingTEMP[ , -j]
      }   
    } 
  }
}
training <- trainingTEMP; rm(trainingTEMP)
# dim(training); dim(testing)

training = training[,-c(1:6)]

x1 <- training[, -53]; correlation.matrix <- cor(x1)
highly.correlated <- findCorrelation(correlation.matrix, cutoff=0.75)
# print(names(training)[highly.correlated])
training <- training[, -highly.correlated]
# dim(training); str(training)

inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]
# dim(myTraining); dim(myTesting); dim(testing)

myTesting <- myTesting[colnames(myTraining)]

myTesting <- myTesting[colnames(myTraining)]
testing <- testing[colnames(myTraining[, -32])] # no classe variable in testing
dim(myTraining); dim(myTesting); dim(testing) 

t5 <- Sys.time(); set.seed(3606)
modFitDT <- rpart::rpart(classe ~ ., data=myTraining, method="class")
# rattle::fancyRpartPlot(modFitDT)
predictionsDT <- predict(modFitDT, myTesting, type = "class")
confusionMatrixDT <- confusionMatrix(predictionsDT, myTesting$classe)
# confusionMatrixDT
plot(confusionMatrixDT$table, col = confusionMatrixDT$byClass, 
     main = paste("Decision Tree Confusion Matrix: Accuracy =", 
                  round(confusionMatrixDT$overall['Accuracy'], 4)))
t6 <- Sys.time(); t6-t5

t7 <- Sys.time(); set.seed(3606)
modFitRF <- randomForest(classe ~ ., data=myTraining)
predictionRF <- predict(modFitRF, myTesting, type = "class")
ConfusionMatrixRF <- confusionMatrix(predictionRF, myTesting$classe)
# ConfusionMatrixRF; 
plot(modFitRF, main="Final Model with Random Forests")
plot(ConfusionMatrixRF$table, col = ConfusionMatrixRF$byClass, 
     main = paste("Random Forest Confusion Matrix: Accuracy =", 
                  round(ConfusionMatrixRF$overall['Accuracy'], 4)))
t8 <- Sys.time(); t8 - t7

t9 <- Sys.time(); set.seed(3606)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
fitControlGBM <- caret::trainControl(method = "repeatedcv", number = 5, repeats = 4, allowParallel = TRUE)
modFitGBM <- caret::train(classe ~ ., data = myTraining, method = "gbm", trControl = fitControlGBM, verbose = FALSE)
prodictionGBM <- predict(modFitGBM, newdata=myTesting)
stopCluster(cluster)
ConfusionMatrixGBM <- confusionMatrix(prodictionGBM, myTesting$classe)
# ConfusionMatrixGBM;
plot(modFitGBM, ylim=c(0.7, 1))
t10 <- Sys.time(); t10 - t9

tx <- Sys.time(); set.seed(3606)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
# define the control using a random forest selection function
ctrl <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(training[, -32], training[, 32], sizes=c(1:31), rfeControl=ctrl)
# print(results) # list the chosen features # predictors(results)
# dim(training)
myTraining2 <- myTraining[, predictors(results)[1:10]]
myTraining2$classe <- as.factor(myTraining[, 32])
myTesting2 <- myTesting[, predictors(results)[1:10]]
myTesting2$classe <- as.factor(myTesting[, 32])
# dim(myTraining2); dim(myTesting2)
fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
modFitRF2 <- train(classe ~ ., data=myTraining2, method = "rf",
                   trControl = fitControl)
predictionRF2 <- predict(modFitRF2, myTesting2, type = "raw")
ConfusionMatrixRF2 <- confusionMatrix(predictionRF2, myTesting2$classe)
ConfusionMatrixRF2; 
plot(modFitRF2, main="Final Model with Random Forests")
plot(ConfusionMatrixRF2$table, col = ConfusionMatrixRF2$byClass, 
     main = paste("Random Forest Confusion Matrix: Accuracy =", 
                  round(ConfusionMatrixRF2$overall['Accuracy'], 4)))
stopCluster(cluster)
ty <- Sys.time(); ty - tx

set.seed(3606)
predictionTEST <- predict(modFitRF, testing, type = "class")
predictionTEST
testing2 <- testing[, predictors(results)[1:10]]
predictionTEST2 <- predict(modFitRF2, testing2, type = "raw")
predictionTEST2