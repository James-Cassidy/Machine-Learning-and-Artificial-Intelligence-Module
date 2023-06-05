#Section 3 (40 Marks)

library(randomForest)
library(mlbench)
library(caret)

#Sources used to help
#https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/

setwd("C:/Users/james/Desktop/All Uni Stuff/assignment3_40267110/Section3")

############(For Markers)###########

#Set working directory, go to Session, Set Working Directory, To Source File Location

####################################

#Working Directory on my laptop that has been used to complete Assignemt 3

data<- read.delim("doodle_data\\40267110_2100_items_features.csv")

#Load Dataset
#data<- read.delim("C:\\Users\\james\\Desktop\\All Uni Stuff\\assignment3_40267110\\csc2062_a3data\\0IP2qZ\\all_features.csv")
str(data)
summary(data)
head(data)

colnames(data) <- c("LABEL", 
                    "Index", 
                    "nr_pix", 
                    "rows_with_2", 
                    "cols_with_2", 
                    "rows_with_3p",
                    "cols_with_3p", 
                    "height","width",
                    "left2tile",
                    "right2tile",
                    "verticalness",
                    "top2tile",
                    "bottom2tile",
                    "horizontalness",
                    "zero_bottom_right"
)


data$LABEL <- as.factor(data$LABEL)

###Section 3.1

set.seed(42)

#Manual search by create 5 folds and repeat 3 times
control <- trainControl(method = 'repeatedcv',
                        number = 5,
                        repeats = 3,
                        search = 'grid')
#create tunegrid

#Set mtry to be 2 and increment by 2 all the way to 8
tunegrid <- expand.grid(.mtry = seq(from=2, to=8, by=2))
rf_gridsearch <- list()


#train with different ntree parameters, 25 increments up to 400
for (ntree in c(25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400)){
  set.seed(42)
  modelfit <- train(LABEL~.-Index -zero_bottom_right,
               data = data,
               method = 'rf',
               metric = 'Accuracy',
               tuneGrid = tunegrid,
               trControl = control,
               ntree = ntree)
  key <- toString(ntree)
  rf_gridsearch[[key]] <- modelfit
}

#Compare results
results <- resamples(rf_gridsearch)
summary(results)

plot(results)

#Plot Results
dotplot(results)

modelfit$results

###Section 3.2

control <- trainControl(method='repeatedcv', 
                        number=5,
                        repeats = 3,
                        search = 'random')
set.seed(42)

#Generates 15 mtry values from 1 to 15, this will generate 15 different indepenant runs of the model

tunegrid <- expand.grid(.mtry=c(1:15))

rf_independent <- train(LABEL ~ .,
                   data = data,
                   method = 'rf',
                   metric = 'Accuracy',
                   tuneGrid=tunegrid, 
                   trControl = control,
                   ntree = 350)

print(rf_independent)

plot(rf_independent)

rf_independent$results

#Section 3.3 Best Model Random Forest KNN

controlKNN <- trainControl(method='repeatedcv', 
                        number=5,
                        repeats = 3,
                        savePredictions = TRUE)

#KNN
set.seed(42)
k_values <- seq(from=1, to=15, by=1)


knn_section3 <- train(LABEL ~ height + width + rows_with_3p,
                   method     = "knn",
                   tuneGrid   = expand.grid(k = k_values),
                   trControl  = controlKNN,
                   metric     = "Accuracy",
                   data       = data)

knn_section3

plot(knn_section3)







##############################################
