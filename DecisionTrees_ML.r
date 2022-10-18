# Import Libraries 
library(rpart)
library(readr)
library(caTools)
library(dplyr)
library(party)
library(partykit)
library(rpart.plot)
library(datasets)
library(caTools)
library(party)
library(dplyr)
library(magrittr)

# Read CSV file
mydata <- read.csv("C:/Users/...", stringsAsFactors = TRUE) #Input location of 2019_APD_Arrests_DATA.csv

df = mydata %>%
  select('APD_RACE_DESC','RACE_KNOWN', 'Person.Search.YN','Search.Based.On','Search.Found','Reason.for.Stop')
df = na.omit(df)

sample_data = sample.split(df, SplitRatio = 0.7)
train_data <- subset(df, sample_data == TRUE)
test_data <- subset(df, sample_data == FALSE)

myFormula <- df$APD_RACE_DESC ~ RACE_KNOWN + Person.Search.YN + Search.Based.On + Search.Found + Reason.for.Stop

model<- ctree(APD_RACE_DESC ~ ., train_data)

ctree_ = ctree(myFormula, df)

predict_model<-predict(ctree_, test_data)

m_at <- table(test_data$APD_RACE_DESC, predict_model)
m_at


ac_Test =sum(diag(m_at)) / sum(m_at)
print(paste('Accuracy for test is found to be', ac_Test))
plot(model)