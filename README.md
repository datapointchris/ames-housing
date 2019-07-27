# Ames Housing Data and Kaggle Challenge
---
## *Exploring the Data Science Process*


### Table of Contents
[Problem Statement](#problem-statement)  
[Intro](#intro)  
[Acknowledgments](#acknowledgements)  
[Looking at the Data](#looking-at-the-data)  
[Summary Stats](#summary-stats)  
[Nulls](#nulls)  
[Splitting Up Columns](#splitting-up-columns)  
[Categorical Columns](#categorical-columns)  
[Ordinal Columns](#ordinal-columns)  
[Weighted Average Fill Function](#weighted-average-fill-function)  
[Numerical Columns](#numerical-columns)  
[Feature Engineering](#feature-engineering)  
[Bivariate Analysis](#bivariate-analysis)  
[Sale Price and Log](#sale-price-and-log)  
[One Hot Encoding / Dummy Variables](#one-hot-encoding-/-dummy-variables)  
[Export / Import](#export--import)  
[Highly Correlated Columns](#highly-correlated-columns)  
[Split the Data for Training and Test](#split-the-data-for-training-and-test)  
[Model](#model)  
[Scores](#scores)  
[Submit the Predictions](#submit-the-predictions)  
[Inferential Visualizations](#inferential-visualizations)  
[Business Recommendations](#business-recommendations)  
[What I Learned](#what-i-learned)  
[Future of the Project](#future-of-the-project)  
 
***




### Problem Statement
> We were tasked with creating a model to predict the sale prices of houses in the Ames, Iowa database.  We were able to choose from our basic linear regression, lasso, and ridge models.  
In order to keep the competition fair, we were not allowed to use any models that we had not covered in class yet, or any advanced modeling techniques beyond the 4th week of instruction.  




### Intro  

This is lengthy dataset for housing.  The data collection seems extensive, although there are many columns that seems to be highly subjective in nature and have an arbitrary ordinal rating.  
This is a good base to start with, and the data analysis portion will give us good insight into whether all of these variables are necessary to build a working model, or they may not add any useful data to the model and it will be unnecessary to include them in future data collections to save time and expense.




### Acknowledgements  

*I want to make it very clear:*
I made **extensive** use of the following kaggle submissions as help and templates for some of my own code and also ideas on how to impute values or how to handle null values.  Most of the similarities will be seen between my plots and visualizations.

These are very good resources for this competition and EDA of the data:  
https://www.kaggle.com/niteshx2/top-50-beginners-stacking-lgb-xgb

https://www.kaggle.com/leeclemmer/exploratory-data-analysis-of-housing-in-ames-iowa

https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

https://www.kaggle.com/rohitbokade94/ames-iowa-housing-price-prediction



### Looking at the Data

**Housing contains both datasets**

The purpose of this is so that any changes made to the training set are also made to the testing set that has been held out.  Note these are not the same as the train/test split sets.



### Summary Stats

- 81 columns, 2930 total rows
- Lots of object columns that will need to be converted properly
- Some of the means and standard deviations are way out of line
- Min and max on () columns indicates most likely mistyped data



### Nulls

- There are a couple of columns that have a lot of missing data.  These columns don't seem to add much value, and the Misc Feature column already has another column labeled 'Misc Value' that automatically inputs the added value, so this column is partially redundant anyway.
- Columns with more than 50% missing data will be deleted.



### Splitting Up Columns

It will be much easier to work with the data for both visualizing and analysis, along with imputing values and filling/dropping nulls if we split the data into:  
- Categorical Columns
- Ordinal Columns
- Numerical Columns



### Categorical Columns

- Filling all misc values with the "No xx" formula



### Ordinal Columns

 These columns have a rating usually from poor to average to excellent.  We can assign the categories a numerical order from 1-5 but this implies that a rating 5, excellent, is 5x better than poor, and 5/3 better than the average column.  Since I'm an aspiring statistician / data scientist this won't cut it.  
 Instead I created a function that takes each of the categories (poor - excellent) and their relationship with *SalePrice* and uses that as the weight that each category will be valued.  
 In my model this substantially improved my scores on the testing data, and the competition ranking.

### Weighted Average Fill Function

#### I'm particularly proud of this function and its ingenuity:

    def get_ratios(column):
        for name, group in housing.groupby(column)['SalePrice']:
            mean_max = housing.groupby(column)['SalePrice'].mean().max()
            ratio = group.mean() / mean_max
            housing[column].replace({name:ratio}, inplace=True)
    # This takes the column and converts the ordinal catgories into ratio of average sale price across categories



### Numerical Columns

>From looking at the frontage data, and seeing that much of it is missing, the closest approximation I came up with is to fill each row with the median number for that particular neighborhood.  
>Another way to approach this would be to include some more variables such as plot size to get a more accurate prediction on the *Lot Frontage*.  
>For my analysis, I don't believe *Lot Frontage* will have that much of an effect on the overall sale price, so this is accurate enough.



### Feature Engineering

> I believe that square footage is a big factor in the *SalePrice* of a home so I created two extra columns to help promote that as a determining factor.  
> *TotalGoodSQ* is the sum of all the good or finished square footage in the home.  
> *TotalSQ* is the sum of all the square footage in the home, regardless of condition.  



### Bivariate Analysis

- I believe *Gr Liv Area*, which is above ground living area, has a strong correlation to *SalesPrice*. 
- Homes over 5000 sq foot are strange outliers, some seem to have tons of sq ft and low sale prices. These do not seem to be outliers, but rather incomplete sales or data.


**Low Correlation**  
    ['Utilities',
     'BsmtFin SF 2',
     'Low Qual Fin SF',
     'Bsmt Half Bath',
     '3Ssn Porch',
     'Pool Area',
     'Misc Val',
     'Mo Sold',
     'Yr Sold']

+ Even though these columns have low correlation with *SalePrice*, they may be useful in other circumstances, or have cofactors that make them important.  My beautiful LASSO model will eliminate the unimportant ones so they will stay.
+ Worth noting is that possibly the year and month sold have interesting characteristics because of the economy and yearly cycles.  This data could be useful in other predictive ways, such as time spent to sell a home.


**High Correlation**

- *Overall Qual* has a high correlation -> HIGHER than my guess of *Gr Liv Area*
- *Gr Liv Area* has a high correlation
- *Garage Cars* and *Garage Area* have a high correlation
- *Full bath* is an indicator of *SalePrice*



### Sale Price and Log

- *SalePrice* appears to be right skewed.  Since we are using a linear model, we are assuming that all variables are normally distributed.  Taking the log of *SalePrice* should fix this problem.
- *Note:* Most likely there are predictor variables that also need to be normalized with the log function, time permitting.


### Trends in Data

- Most of this is discussed in the [Inferential Visualizations](#inferential-visualizations) and [Business Recommendations](#business-recommendations) sections.
- This was just the natural place to create these visualizations in order to keep the modeling notebook clean.



### One Hot Encoding / Dummy Variables

- Create dummy variables for categorical columns.  Straightforward.



### Export / Import

- Write out the CSV file, and back in.  I chose to do this so that if I messed up my data while modeling, I did not have to run a bunch of cells again.
- I am starting the modeling phase with a cleaned dataset that is ready for fitting.



### Highly Correlated Columns

- I chose not to delete the highly correlated columns
- - This was in part because both of my engineered columns would be removed.
- Also I am using the lasso model for fitting, which will zero out the column if it's unnecessary.



### Split the Data for Training and Test

- Before the train/test split.  So I have the original data that I will use to test my trained model, before submitting the results to Kaggle.



### Model

- I chose to use the lasso model and keep all of my variables in the model and let lasso zero out the columns it finds unnecessary or not condusive to the accuracy of the model.
- All variables must be scaled first since they are in differing units and would cause problems.



### Scores

> 91% on the training data
> 91% on the testing data
> 91% on the cross validation score
> - I believe this model will perform well and generalize well over new data.  It haven't made it so specific that it will only predict this data, it is above 90% accuracy, and it performs as well on the testing data as it did on the training data.
> - This model is around 5th in the overall rankings, with some others only being slightly better.
> - They may suffer from overfitting when our models are exposed to the other 70% of the Kaggle testing data.
> - Finally, the submissions above this model are mostly 3x as many submits and probably hours and hours of work.  Point of diminishing returns has most likely been reached at this point in time.

- The residuals plot looks mostly symmetric since I took the log of *SalePrice*

![residuals](../images/residuals.png)



### Submit the predictions

- I pulled in the original testing dataset to get the Id column back.
- - This turned out to be easier than trying to keep it attached to the training dataset and exclude it from all cleaning and modeling phases.
- Make sure to exponentiate the predictions before adding them to the submission.



### Inferential Visualizations

**Neighborhoods**

> Based on the visualizations produced in this section, it is clear that more homes are sold in the summmer months.  However, this has no correlation to the average sale price for the respective months.
> Stonebrook, Northridge Heights, Northridge, Greenhill, and Veenker are the neighborhoods with the highest average selling price.
> Meadow Village, Iowa Dot and Railroad, Briardale, Old Town, Brookside are the neighborhoods with the lowest average selling price.

**Features Increasing Value**

- Overall Quality
- *Total Square Feet*
- *Total Good Square Feet*
- External Quality
- Above Ground Living Area Square Footage
- Basement Quality
- Kitchen Quality
- Total Basement Square Feet
- Garage Area


**Features Decreasing Value**

- Year Sold
- Low Quality Square Footage
- Basement Half Bathrooms
- Kitchen(s) Above Ground
- Enclosed Porch


**Application of model:**

- Whereas I feel this model does a good job of predicting the housing prices based on the data provided in this set, I believe ANY model that is forced to evaluate homes in any area where location is an important factor will perform poorly.
- Square footage, bathrooms/bedrooms, and overall quality will of course determine the *SalePrice* in relation to other homes, but there would need to be a correction factor for zip code, county, or neighborhood in order to apply the model to other locations.


### What I Learned

- Functions.  Functions are the key to making tedious tasks fast.
    - It's not always necessary to delete null values.
    - Sometimes a good guess is better than deletion of the row or column
    - Some filled null values do not influence the model, but the deleted rows or columns would
- Most of the time spent to make a model is spent in making the data formatted properly
- Models should not be one and done
- There are many methods of accomplishing the same task, no particular one is right always
- Most likely there is a python package to do just what you need, if you can find it.


### Future of the project. 

[ ] Clean up the code  
[ ] Add comments for clarity  
[ ] Update readme  
[ ] Add some more visualizations for a better presentation  
[ ] Format to be better in Github / Portfolio  
[ ] Create functions to cycle through models  
[ ] Create a template for README  
[ ] Create a template for Data Science Workflow  
[ ] Create a functions module for data cleaning and processing  
[ ] Evaluate different models / split notebook  
[ ] Model tuning and hyperparameters  
[ ] Production model  
[ ] Apply GridSearch to model and evalute  