#Section 2 (30 Marks)

#Sources used
#https://www.edureka.co/blog/knn-algorithm-in-r/

setwd("C:/Users/james/Desktop/All Uni Stuff/assignment3_40267110/Section2")


############(For Markers)###########

#Set working directory, go to Session, Set Working Directory, To Source File Location

####################################

#Working Directory on my laptop that has been used to complete Assignemt 3


features_df<- read.csv("40267110_features.csv")
str(features_df)
summary(features_df)

#Select the first 6 featurers from the data frame
df <- features_df[, c("LABEL","nr_pix", "rows_with_2", "cols_with_2", "rows_with_3p","cols_with_3p", "height")]

df$type <- 0

df$type
df$type[df$LABEL == 'approxequal'] <- "Digit"
df$type[df$LABEL == 'less'] <- "Digit"
df$type[df$LABEL == 'greater'] <- "Digit"
df$type[df$LABEL == 'equal'] <- "Digit"
df$type[df$LABEL == 'lessequal'] <- "Digit"
df$type[df$LABEL == 'greaterequal'] <- "Digit"
df$type[df$LABEL == 'notequal'] <- "Digit"

df$type[df$LABEL == 'one'] <- "Number"
df$type[df$LABEL == 'two'] <- "Number"
df$type[df$LABEL == 'three'] <- "Number"
df$type[df$LABEL == 'four'] <- "Number"
df$type[df$LABEL == 'five'] <- "Number"
df$type[df$LABEL == 'six'] <- "Number"
df$type[df$LABEL == 'seven'] <- "Number"

df$type[df$LABEL == 'a'] <- "Letter"
df$type[df$LABEL == 'b'] <- "Letter"
df$type[df$LABEL == 'c'] <- "Letter"
df$type[df$LABEL == 'd'] <- "Letter"
df$type[df$LABEL == 'e'] <- "Letter"
df$type[df$LABEL == 'f'] <- "Letter"
df$type[df$LABEL == 'g'] <- "Letter"
df$type



#New dataframe

new_df <- df[, c("type","nr_pix", "rows_with_2", "cols_with_2", "rows_with_3p","cols_with_3p", "height")]

#Checking to make sure df is created
str(new_df)

table(new_df$LABEL)
head(new_df)

#Set type as a factor, this is to allow
new_df$type <- as.factor(new_df$type)
new_df$nr_pix <- as.numeric(new_df$nr_pix)
new_df$rows_with_2 <- as.numeric(new_df$rows_with_2)
new_df$cols_with_2 <- as.numeric(new_df$cols_with_2)
new_df$rows_with_3p <- as.numeric(new_df$rows_with_3p)
new_df$cols_with_3p <- as.numeric(new_df$cols_with_3p)
new_df$height <- as.numeric(new_df$height)
str(new_df)

#Normalize function used so KNN can be performed on df
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

#Apply Normalize function to each column apart from LABEL
df.norm <- as.data.frame(lapply(new_df[,2:7], normalize))
head(df.norm)



#Create train_df and test_df

# randomly shuffle rows:
set.seed(42)
df_shuffled <- df.norm[sample(nrow(df.norm)),]
head(df_shuffled)
str(df_shuffled)

#Check to make sure there are no NULL Values
sum(is.na(df_shuffled))


set.seed(42)

# Training dataframe will use all 168 features

train_df = df_shuffled[1:168,]
test_df = df_shuffled[1:168,]


df_train_target <- df[1:168, 1]
df_test_target <- df[1:168,1]


#Checking to make sure there are no null values in dataframe, otherwise KNN will not work
sum(is.na(df_train_target))

require(class)
library(class)
library(caret)


set.seed(42)
i=1
k.trainingsets=1
#function that will only pick odd numbers
oddnumbers <- function(x) x[ x %% 2 == 1 ]

for (i in oddnumbers(1:25)){
  knn.mod <- knn(train=train_df, test=test_df, cl=df_train_target, k=i)
  k.trainingsets[i] <- 100 * sum(df_test_target == knn.mod)/NROW(df_test_target)
  k=i
  cat(k,'=',k.trainingsets[i],'')
}


#Remove NA values from k.trainingsets has it will interfere with graph
k.trainingsets <- na.omit(k.trainingsets)
k.trainingsets

#k1 = 97.02381
#k3 = 35.11905 
#k5 = 29.16667 
#k7 = 20.23810 
#k9 = 23.80952 
#k11 = 16.07143 
#k13 = 19.04762
#k15 = 17.85714 
#k17 = 15.47619 
#k19 = 16.07143 
#k21 = 17.26190 
#k23 = 15.47619 
#k25 = 15.47619

############################################################################

#Section 2.2 - KNN and 5-Cross Validation

trControl <- trainControl(method  = "cv",
                          number  = 5,
                          savePredictions = TRUE)

# K values 1 - 25 (odd numbers inclusive)
set.seed(42)
k_values <- seq(from=1, to=25, by=2)


fit_all_k <- train(type ~ .,
                   method     = "knn",
                   tuneGrid   = expand.grid(k = k_values),
                   trControl  = trControl,
                   metric     = "Accuracy",
                   data       = new_df)
fit_all_k


#Plot accuracy graph for KNN 5 Fold Validation
plot(fit_all_k, type="b", xlab="K- Value",ylab="Accuracy level")


###Graph for Training Set Error Rate and Cross Validated Error Rate

#One Divided by the number of K used for the X axis
x <- c(1/25, 1/23, 1/21, 1/19, 1/17, 1/15, 1/13, 1/11, 1/9, 1/7, 1/5, 1/3, 1)

#Values of the KNN model without Cross Validation (Training Set)
y1 <- k.trainingsets [1:13]/100

#Values for the Cross Validated Accuracies of Section 2
y2 <- c(0.8343112, 0.7863713, 0.7626636, 0.7261319, 0.7441355, 0.6972142, 0.6679399, 
        0.6383499  , 0.6565215  , 0.6445786  , 0.6082047  ,0.5903794, 0.6139292 )

#Plot Graph
plot(x, y1, type="o", col="blue", pch="o", 
     main="Error Rate for Training and Cross Validated Sets",
     ylab="Error Rate", xlab = "1/K", 
     lty = 1, ylim = c(0,1))

points(x, y2, col="red", pch="o")
lines(x, y2, col="red", lty=2)
legend("topright", c("Training Set", "Cross Validated Set"),
       col=c("blue", "red"), lty=1:2, cex=0.8)


