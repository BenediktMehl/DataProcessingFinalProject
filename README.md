# DataProcessingFinalProject

## The problem
We have a recipe dataset with columns like: title, ingredients, directions, etc. We want to predict the rating. Even though the rating is a categorical variable, we will treat it as a regression problem and use only regression models.
It is also interesting to see that the distribution of rating is very skewed and the baseline is already quite good with an mse around 1.8. 
![RatingDistribution](results/rating.png "RatingDistribution")


## Methodology
I splitted the tasks into three parts: 
- preprocessing: Task 3.1 till 3.3
- processing: Task 3.4 till 3.5
- extension: Task 4

### Preprocessing and Analysis
The preprocessing is done in the [preprocessing.ipynb](preprocessing.ipynb) file. 

#### Analysis of the categories
There are very many different categories in the dataset. So it is very hard to see by eye which ones could have influence on the rating.
To get an overview of the categories I looked at that diagram:
![Categories](results/count_corr_ana.png "Categories")
Every dot is a category. The x axis shows the difference between the mean distribution of the ratings in general and the mean distribution of the ratings in that category. If the difference is high, I expect the correlation between the category and the rating to be high. If the difference is low that means the category is not very informative. The y axis shows the count of the category. Categories with a high correlation, but a low count are not very significant.
I want to look at the categories with a high count and a high correlation:
![CategoriesSignificant](results/ana_signific.png "CategoriesSignificant")

One can see that for example the category "Alcoholic" has a high number of 0 ratings and a low number of 4.375 ratings. This means that the category "Alcoholic" is very informative for the rating and if we see a recipe with the category "Alcoholic" we can expect a lower rating.
The category "Champagne" seems to be pretty polarizing. There are a lot of 0 ratings and a lot of 5 ratings. This means that the category "Champagne" is very informative for the rating and if we see a recipe with the category "Champagne" we can expect either a very high or a very low rating.

I also did a PCA on the numerical columns to if one could reduce the dimensionality of the dataset. 
![PCA](results/pca.png "PCA")
The PCA showed that the first two components already explain 99% of the variance. This is why I reduced the dimensionality of the dataset to two dimensions and still have a good representation of the data.

#### tfidf
Tfidf is done on the combination of the following columns: title, description and directions. I decided to add the column title because I think the title of a recipe is very important for the rating.
 

#### 

## Results
Results analysis: include graphs, metrics, and a discussion on model
performance.

# Extension
## The problem
Cooking with Leftover Food - If there is leftover ingredients a user wants to enter the ingredients and get a meal and recipe for that meal.

## Methodology
Transformerbased local models vs groq

Groq provides easy and free to use API keys for their LLMs. 

## Results









