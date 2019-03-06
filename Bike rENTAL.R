rm(list=ls())
setwd("F:/DATASCIENCE/EDWISOR/Project/Bike RENTAL")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

# reading the data files
data=read.csv("day.csv")

# getting some information about the  data
str(data)
summary(data)
dim(data)

#checking the unique values of the variables
col = names(data)
for (i in col){
  print(i)
        print(length(unique(data[,i])))
        }

#We can conclude that some of the numerical variables are discrete and should be treated as factors.
#Storing names of numerical and categorical variables in different objects
names_of_num = c("instant","temp","atemp","hum","windspeed","casual","registered","cnt")
names_of_cat = c("season","yr","mnth","holiday","weekday","workingday","weathersit")

#Converting to factors 
for (i in names_of_cat)
  {
    data[,i] = as.factor(data[,i])
    print(class(data[,i]))  
    }

#******************************************MISSING VALUE ANALYSIS******************************************************
sum(is.na(data))
#No missing values found.

#******************************************OUTLIERS ANALYSIS******************************************************
for (i in 1:length(names_of_num)) {
  assign(paste0("gn",i), ggplot(aes_string(y = (names_of_num[i]), x = "cnt"),data = subset(data))+
           stat_boxplot(geom = "errorbar" , width = 0.5) + 
           geom_boxplot(outlier.color="red", fill = "grey" , outlier.shape=18, outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=names_of_num[i],x="cnt")+
           ggtitle(paste("Box Plot of Count and",names_of_num[i])))
  print(i)   
  print(names_of_num[i])
}
#Drawing the Boxplot

gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,ncol=2)
gridExtra::grid.arrange(gn6,gn7,ncol = 2)

#Getting the outliers data element from each variable

for (i in names_of_num) {
  print(i)
  val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
  print(length(val))
  print(val)
}
#Humidity has 2 outliers - having values 0 and 0.18
#Windspeed has 13 outliers - having a minimum value of 0.38 
#Casual has 44 outliers - having a minimum value of 2248

#Plotting the Outlier containing variables with the Target Variable to check if Outliers are relevant or noise.
plot(data$hum,data$cnt)
plot(data$windspeed,data$cnt)
plot(data$casual,data$cnt)

#Since the number of outliers is very less and the Outliers are random as seen in the plots,
    #we can  consider them as noise and can decide to delete them 
for (i in names_of_num) 
    {
      value = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
      data = data[which(!data[,i] %in% value),]
    }

#*******************EXPLORING THE DATA WITH VISUALISATIONS*********************
dim(data)
#CATEGORICAL VARIABLES
# creating boxplots of different categorical variables with the target variables to see the variation

target_v = c('casual','registered','cnt')
for (i in names_of_cat)
  {
    for (j in target_v)
    {
      boxplot(data[,j]~data[,i] , xlab = i, ylab=j)
    }
  
  }

#From the boxplots we can conclude that:-
  #1.The target variables show similar pattern wrt varibles- year,weather,month and season
  #2.As expected, casual users are more on holidays and weekends.
  #3.Registered users are mostly on workingdays/weekdays.


#NUMERICAL VARIABLES
#scatter plot of numerical variables with target variables
num_v = c('temp','atemp','windspeed','hum')
for (i in num_v)
  {
    for(j in target_v)
    {
      plot(data[,i],data[,j],xlab=i,ylab=j)
    }
  }

#********************PLOTTING CORRELATION MATRIX FOR FEATURE SELECTION*********
library(corrgram)

corrgram(data[,names_of_num],order = F, upper.panel = panel.pie, text.panel = panel.txt, main = "correlation plot" )

#Temperature and Feeling temperature are understandably highly correlated.
  #It will be better one of them.
  #We can drop from numerical variables- 'temp' and 'instant'.

#********************FEATURE SELECTION&ENGINEERING ON CATEGORICAL VARIABLES****************************

#Since casual and registered have different trends on weekends and weekdays,
  #We decide to create a new variable having the type of day

data$day_type=0
data$day_type[data$holiday==0 & data$workingday==0]="weekend"
data$day_type[data$holiday==1]="holiday"
data$day_type[data$holiday==0 & data$workingday==1]="working day"

data$day_type=as.factor(data$day_type)

#Reducing the variables not needed for analysis

data$instant = NULL
data$temp = NULL
data$casual = NULL
data$registered = NULL

set.seed(1234)
train.index = createDataPartition(data$cnt, p = .80, list = FALSE)
train = data[ train.index,]
test  = data[-train.index,]

#*******************MODEL BUILDING*********************************************
#LINEAR REGRESSION MODEL

#training a model on the data
library(stats)
lr_model = lm(cnt ~ day_type+hum+atemp+windspeed+season+weathersit+holiday+workingday+weekday+yr , data = train)
lr_model
lr_pre = predict(lr_model,test[,-12])
#evaluating model performance
summary(lr_model)
RMSE(lr_pre,test$cnt) # = 799
RMSLE(lr_pre,test$cnt) # = 0.25

#SUMMARY OF MODEL:-
  #1. R-squared = 0.8323
  #2. Adjusted R-sqaured = 0.8272
  #3. Residuals has a maximum value of 2840 and min of -3574, which is a bad sign for atleast two observations


#Random Forest is Chosen for this Regression Problem
#After experimenting with number of trees and variables at each split we train the model

fit = randomForest(cnt ~ day_type+hum+atemp+windspeed+season+weathersit+holiday+workingday+weekday+yr, data=train,importance=TRUE, ntree=300,mtry=4)
fit
#% variance explained(r_2) = 87.74

pred = predict(fit,test)

importance(fit)
plot(fit)
summary(fit)

#Error Evaluation
install.packages("MLmetrics")
library(MLmetrics)
RMSE(pred,test$cnt) #= 662.8

RMSLE(pred,test$cnt) #= 0.18

plot(test$cnt,pred,xlabel="Actual values",ylabel= "Predicted Value")
