---
title: "Human Activity Recognition and Dumbbell Exercise Classe"
author: "Elizaveta Karaseva"
date: "December 22, 2018"
source: http://groupware.les.inf.puc-rio.br/har
output:
      html_document:
        keep_md: yes
---

## Overview

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. It consists of different metrics measured while doing exercises with a dumbbell. It also includes a manner in whih every exercise was performed ("classe" variable). The goal of the project is to predict the manner in which they did the exercise. 

This report describes the model building process including different steps such as feature selection and cross-validation.

5 classification algorithms have been reviewed, but only the best performing are demonstrated in this report - Random Forest and Boosting. For expected Accuracy Rates, Out of Sample Errors and Final Predictions, refer to "Models Review and Prediction" section in the bottom of this document.



## Read Data

There is a set of statistical metrics with missing values that summarises results of measurements within each window (avg, var etc). Because the final goal is to classiffy each problem_id from the testing set, it doesn't seem approproate to rely on the metrics missing from the training set. Therefore, as the initial feature selection step, we recommend removing missing values from the training dataset.



The training set was further split into training (70%) and validation (30%) sets. And here is the breakdown:


```r
dim(training); dim(validation); dim(testing)
```

```
## [1] 13737    53
```

```
## [1] 5885   53
```

```
## [1] 20 53
```

Training set will be used for exploration, feature selection and modeling purposes. Validation set will be used to asses accuracy of trained models. And finally, testing data set will be used to predict classe levels.

##Exploratory and Feature Selection

First, let's see how many levels of each classe represented in the training data set and if there are relationships between potential predictors in a dataset.

![](HAR_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

According to the correlation matrix, there are quite a few variables that are highly correlated. Some correlations are obvious, like for example, correlation between **total_accel_belt** and **accel_belt_z** or **magnet_belt_x** and **accel_belt_x**. Some are more interesting, like **gyros_forearm_z** and **gyros_dumbbell_z**. 

One of the standard procedures in feature selection process is to remove highly correlated variables. However, after multiple modeling trials, it was cocluded that removing highly correlated variables from our training data reduces prediction accuracy on validation data set. Therefore, we refrain from removing correlated variables. In addition, there is an explanation from mechanical engineering - 3 measuring devices (accelerometer, gyroscope and magnet) are used to calibrate measuring errors for pitch, roll and yaw calculations. Thus, by removing some measurments, we will stay with more bias in our data. 

Multiple variables have been studied using various plots, but it wasn't easy to identify if there were any specific patterns within sets of 2-4 variables. See one of the examples below:


```r
featurePlot(x=training[,16:18], y=training[,1], plot="ellipse")
```

![](HAR_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

That implies, that linear classification will be harder to implement. Therefore, we will use other machine learining algorithms that tuned well for classification problems like that one.

##Modeling Overview

Multiple Machine Learning Algorithms have been applied to training data set:

- rpart (Decision Trees)

- svm (Support Vector Machines)

- rf (Random Forest)

- gbm (Boosted Tree)

- nb (Naive Bayes)

###Preprocess

Multiple adjustments have been reviewed:

- PCA

- Normalization (center and scale)

After adjusting for different preprocession options, it was discovered, that PCA tended to recude accuracy for any selected model. At the same time, centering and scaling variables helped improve prediction accuracy on validation data sets. Therefore, we recommned refrain from applying PCA for this data set and use "center" and "scale" preprocessing options.

###Cross-Validation

To reduce bias of the prediction algorithms, the following cross-validation parameters have been set:


```r
TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=3)
```

##Modeling Results

In this report, we will only demonstrate algorithms that performed the best - Random Forest and Boosted Trees. Support Vector Machines, Naive Bayes and Decision Tree algorithms have been reviewed but they didn't show satisfactory accuracy rates. To avoid computatinal complications, only beest performing algorithms will be demonstrated in this report.

###Random Forest

```r
randomForest <- train(classe ~ .
                      ,method="rf"
                      , preProcess = c("center","scale")
                      , trControl = TrainingParameters
                      , data=training)
importance <- varImp(randomForest, scale=FALSE)
plot(importance)
```

![](HAR_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

Some of the variables like **roll_belt**, **pitch_forearm** have very high importance. These results can be used to further work on feature selection. 

###Boosted Tree

```r
boostedTree <- train(classe ~ .
                             ,method="gbm"
                             , preProcess = c("center","scale")
                             , trControl = TrainingParameters
                             , data=training
        )
```

##Models Review and Prediction

###Confusion Matrices and Out of Sample Errors

```r
confusionMatrix(validation$classe,predict(randomForest,validation))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    6 1131    2    0    0
##          C    0    9 1014    3    0
##          D    0    0   20  943    1
##          E    0    0    2    1 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9924          
##                  95% CI : (0.9898, 0.9944)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9903          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9912   0.9769   0.9958   0.9991
## Specificity            0.9998   0.9983   0.9975   0.9957   0.9994
## Pos Pred Value         0.9994   0.9930   0.9883   0.9782   0.9972
## Neg Pred Value         0.9986   0.9979   0.9951   0.9992   0.9998
## Prevalence             0.2853   0.1939   0.1764   0.1609   0.1835
## Detection Rate         0.2843   0.1922   0.1723   0.1602   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9981   0.9948   0.9872   0.9958   0.9992
```

```r
###Out of Sample Error
1 - sum(predict(randomForest,validation) == validation$classe)/length(predict(randomForest,validation))
```

```
## [1] 0.007646559
```

```r
confusionMatrix(validation$classe,predict(boostedTree,validation))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1641   20   10    2    1
##          B   49 1056   28    6    0
##          C    0   36  974   14    2
##          D    1    0   43  915    5
##          E    5    9    4   12 1052
## 
## Overall Statistics
##                                          
##                Accuracy : 0.958          
##                  95% CI : (0.9526, 0.963)
##     No Information Rate : 0.2882         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9469         
##  Mcnemar's Test P-Value : 4.522e-09      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9676   0.9420   0.9197   0.9642   0.9925
## Specificity            0.9921   0.9826   0.9892   0.9901   0.9938
## Pos Pred Value         0.9803   0.9271   0.9493   0.9492   0.9723
## Neg Pred Value         0.9869   0.9863   0.9825   0.9931   0.9983
## Prevalence             0.2882   0.1905   0.1799   0.1613   0.1801
## Detection Rate         0.2788   0.1794   0.1655   0.1555   0.1788
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9798   0.9623   0.9545   0.9771   0.9931
```

```r
###Out of Sample Error
1 - sum(predict(boostedTree,validation) == validation$classe)/length(predict(boostedTree,validation))
```

```
## [1] 0.04197111
```

###Final Prediction

```r
predict(randomForest, testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
predict(boostedTree, testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
