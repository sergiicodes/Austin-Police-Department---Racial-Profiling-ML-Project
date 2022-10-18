I created a neural network that makes predictions based on the 2019 dataset from the Austin Police Department. 
To run this, install the necessary libraries in a virtual env. Many graduate classes and clubs recommend Spyder. I opt for Visual Studio Code, but I also like checking across different IDEs.

I created the Adaptive Boosting algorithm using Python. AdaBoost, or Adaptive Boosting, was used for my ensembles as it adapts to the errors of the previous weak hypotheses. 
Again, I imported the necessary libraries using the sklearn library. I used pandas to read the csv and replaced my selected Categories as numbers: White 1, Black 2, Latinx 3, Asian 4, etc.

For my decision trees, I did this in R. First, import the libraries needed and as seen. Then import the csv file using read.csv in R. I made sure to read the columns (categories) that needed to be assessed. These categories are APD_RACE_DESC, RACE_KNOWN, Person Search YN, Search Based On, Search Found, and Reason for Stop.
