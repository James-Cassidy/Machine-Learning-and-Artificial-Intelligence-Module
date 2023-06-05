
#Section 1 (30 Marks)

#Sources used:
#241_evalution.r
#161_logistic_regression_iris.r

setwd("C:/Users/james/Desktop/All Uni Stuff/assignment3_40267110/Section1")

############(For Markers)###########

#Set working directory, go to Session, Set Working Directory, To Source File Location

####################################

#Working Directory on my laptop that has been used to complete Assignemt 3


features_df<- read.csv("40267110_features.csv")

str(features_df)
summary(features_df)

df <- features_df[, c("LABEL", "nr_pix")]


library(tidyverse)
library(caret)

#Make dummy variable that discriminates maths symbols from other variables
#Set this dummy variable to 1 for symbols

df$maths.symbol <- 0
df$maths.symbol
df$maths.symbol[df$LABEL == 'approxequal'] <- 1
df$maths.symbol[df$LABEL == 'less'] <- 1
df$maths.symbol[df$LABEL == 'greater'] <- 1
df$maths.symbol[df$LABEL == 'equal'] <- 1
df$maths.symbol[df$LABEL == 'lessequal'] <- 1
df$maths.symbol[df$LABEL == 'greaterequal'] <- 1
df$maths.symbol[df$LABEL == 'notequal'] <- 1
df$maths.symbol

summary(df)

set.seed(42)

# Section 1.1

library(ggplot2)

plt <- ggplot(df, aes(x=nr_pix, fill=as.factor(maths.symbol))) +
  geom_histogram(binwidth=1, alpha=.5, position='identity')
plt 
ggsave('histogram_nr_pix.png',scale=0.7,dpi=400)



# Logistic regression predicting If it is a maths symbol using nr_pix
# Generalized Linear Model - glm

glmfit<-glm(maths.symbol ~ nr_pix, data = df, family = 'binomial') 
summary(glmfit)$coef

exp(0.02451)
(1.024813-1) * 100

newdata = as.data.frame(c(20,32,44,56)) 
colnames(newdata) = 'nr_pix'

head(newdata)
predicted = predict(glmfit, newdata, type="response")
predicted

x.range = range(df[["nr_pix"]])

x.values = seq(x.range[1],x.range[2],length.out=1000)

fitted.curve <- data.frame(nr_pix = x.values)
fitted.curve[["maths.symbol"]] = predict(glmfit, fitted.curve, type="response")

# Plot the training data and the fitted curve:
plt <-ggplot(df, aes(x=nr_pix, y=maths.symbol)) + 
  geom_point(aes(colour = factor(maths.symbol)), 
             show.legend = T, position = "dodge")+
  geom_line(data=fitted.curve, colour="orange", size=1)

plt

ggsave('glm_fitted_curve.png',scale=0.7,dpi=400)

remove(x.values)

exp(0.02451)



#Relabel values (1 = yes, 0 = no)
#1 = Maths Symbol, 0 = Not Maths Symbol

df$maths.symbol[df$maths.symbol==1] <- "yes"
df$maths.symbol[df$maths.symbol==0] <- "no"


#Converting to factors to work with the confusion matrix
df$maths.symbol <- as.factor(df$maths.symbol)

class(df$maths.symbol)

library(caret)

#Specify the training method used and number of folds (5 fold Cross Validation)
ctrlspecs <- trainControl(method="cv", number=5, 
                          savePredictions=TRUE,
                          classProbs = TRUE) 

set.seed(42)

model <- train(maths.symbol~nr_pix, data=df, 
               trControl=ctrlspecs, method="glm", family="binomial")

mean(model$pred$pred==model$pred$obs)



print(model)

summary(model)




# Threshold of 0.5 threw an error when trying to run so 0.4 was used

cm = confusionMatrix(table(model$pred$yes >= 0.5, model$pred$obs == "yes"),
                     mode = "prec_recall")
cm


# Threshold of 0.4 used as alternative to 0.5

cm = confusionMatrix(table(model$pred$yes >= 0.4, model$pred$obs == "yes"),
                     mode = "prec_recall")
cm
#True Positive Rate
cm$byClass["Sensitivity"]

#True Negative Rate
cm$byClass["Specificity"]


#Used to see ROC Curve
library(MLeval)

res <- evalm(model)
res$roc
res

################################################################################################
