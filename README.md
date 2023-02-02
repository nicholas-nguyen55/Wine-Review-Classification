# Wine-Review-Classification

Introduction: 

This project goes over a wine reveiw dataset that we will use for differenet machine learning classification techniques.  In this report, we will discuss the data acquiring and data wrangling process, then we will discuss the nature of the data. Lastly, we will report the accuracy of the classification models we ran. The goal of this project was to understand how to use descriptive data in wine reviews to predict the outcomes of the following three variables: price, points, and varieties. 

Step 1: Data acquisition 

In this section, we will describe the data we have used in this project. 

For the final project, we used a wine review dataset that was scrapped in 2017 from Wine Enthusiast 
(available on Kaggle: https://www.kaggle.com/datasets/zynicide/wine-reviews). 

Step 2: Data loading and preprocessing

Data loading: In this step, we started with loading the csv files that were saved in the local directory. More specifically:

1.	Using pd.read_csv(), we loaded the dataset into the Jupyter notebook and named the dataframe as reviewdf. 
2.	Renaming the dataset columns to the following: "Index", "Country", "Review", "Designation", "Points", "Price", "Province", "Region_1", "Region_2", "Variety", "Winery"

Data cleaning and preprocessing: Clean up null values, remove unwanted columns, and preprocess the text column “Review”. 

Cleaning up missing data: we first used :is.null.sum() to determine the proportion of missing values from the merged 

(1)	For “country” and “province” attributes, there were 5 total values missing from the dataset, which is very little. Therefore, we just dropped the missing values.
(2)	For “variety” attribute, only 1 value was missing, therefore, the same treatment was done, using the following code: 
winereviews.dropna(how="any", subset = ["country", "province"])
(3)	For “price” attribute, there were 13643 values missing. That was a significantly larger number of missing values, therefore, it would be problematic if we tried to drop them. Instead, after inspecting the distribution of the dataset using boxplot in seaborn, see below: 

![image](https://user-images.githubusercontent.com/113401627/216464495-ea12c841-314f-4f7b-ada7-aa4904f41a77.png)

After the imputation, we did descriptive statistical analysis with price and points, and the output were as followed:

![image](https://user-images.githubusercontent.com/113401627/216464527-e6977fb5-bab3-44c3-a298-a258c8c4f0cc.png)

Text data “Review” preprocess:

In this dataset, the one column that needed most attention was “Review”. There were 150k reviews in this dataset. Obviously, there would be a lot of noises with the punctuation, numbers, and stop words if we do not clean up the dataset, which can ultimately impact the results of the text analysis. Therefore, we have taken two types of processing to clean up the data, (1) POS tagging and extract only nouns and adjectives as these are probably the best language features of the wine reviews as each wine review is a description of the reviewer’s tastes and feelings about the wine. (2) Tokenized words with all punctuations, numbers, and stop words (including a customized list of words) removed. 

(1) POS tagging and extract nouns, adjectives, adjective phrases, and noun phrases: First, we were interested in only the nouns and adjectives used in each review. Therefore, to extract the nouns and adjectives for each review, we had to first use NLTK pos_tagged method to tag all words in the sentences. We did not move any punctuation in this step because it is an important indicator for POS tagging, nor did we eliminate any stop words. The code we used in this step are:

- Lower case all the text:

![image](https://user-images.githubusercontent.com/113401627/216464598-eabd0003-3169-4bc6-90e1-def3c628d048.png)

- Tokenize sentences into words within the nested list and apply POS tagging to the tokens.

![image](https://user-images.githubusercontent.com/113401627/216464615-c2715cb3-9299-46f3-b70f-b3f645daba29.png)

It is tricky to work with pandas dataframe since the column has 150k entries; therefore, we had to use apply(nltk.word_tokenize) to all the reviews and then pos_tag_sents tag the sentences within each review. 

-	Extract nouns and adjectives from each review and saved the results into the new columns “nouns” and “adjectives” 

Then we used regex expressions to retrieve two types of POS. To retrieve nouns, we used the POS tags “NN”, “NNS”, “NNP”, AND “NNPS” because they stand for singular nouns, plural nouns, singular proper nouns, and plural proper nouns. Then we used a lambda function to apply this expression to each row in the column “TaggedReview”. The results were saved as a new column “nouns”.

![image](https://user-images.githubusercontent.com/113401627/216464699-7bf07e79-7485-4b97-97c0-8b5100e86f95.png)

Then we applied the same procedure to extract all adjectives from the Tagged Review column with regex expression 'JJ', 'JJR', 'JJS'. All adjectives that were retrieved were saved in the new column “adjectives”. 

![image](https://user-images.githubusercontent.com/113401627/216464725-0383f31a-a539-40a3-a219-bfefecb8382f.png)

- Extract noun phrases and adjective phrases using regex and saved them into new columns “nounphrases” and “adjphrases”. 

For the adjective phrases, we used the regex expression: grammar_adjph = "ADJPH: {<RB.?>+<JJ.?>}". Then we used the chunking technique to parse the adjective phrases from the subtrees when the label matches the regex parser we set up. Then all the adjective phrases were extracted and appended back to the panda dataframe reviewdf as “adjphrases”. 

For the noun phrases, we used the regex expression grammar_nounph = "NP:{<DT>?<JJ>*<NN>}" and repeated the same process and saved the results to the “nounphrases” column. 

(2) Tokenize text attribute: Since we are also interested in looking at some word frequency distributions of the wine description data, we used the NLTK package to perform the following tasks to tokenize each review (which has been lower cased in the first step of POS tagging) to create a bag of words. 

-	Tokenize using nltk.word_tokenize() 

![image](https://user-images.githubusercontent.com/113401627/216464772-3d9a054b-aedd-4589-911f-6b953e9d5e42.png)

-	Using Regex expressions to capture non-alpha characters (punctuations and numbers)

![image](https://user-images.githubusercontent.com/113401627/216464795-c89783a5-efc0-4dd8-a032-006ed10996a5.png)

-	Remove stopwords using stopwords from the nltk package
-	Then we customized a list once stopwords came with the nltk package were removed. We did a brief inspection of the dataset and noted the following words not meaningful in the analysis: “'drink' , 'now', 'wine' ,'flavour','flavor', "'s",'still', 'flavours', 'flavors', "'ll"” 

![image](https://user-images.githubusercontent.com/113401627/216464824-93e29145-2d4b-46a9-bf99-94d30da36f9c.png)

Once the cleaning was done to all the text data, we saved the cleaned tokens into a new column called “word_token_clean”, which was heavily used in the classification task. 

Here is a preview of the winereview dataset I worked with in most of the business questions. The data was also saved as “winereviewupdated.csv” for the ease of retrieval for later analysis:  

![image](https://user-images.githubusercontent.com/113401627/216464845-74b95b08-d96a-4d6e-8d18-5ce90eebc793.png)

Step 3: Basic descriptive data analysis and further data preprocessing

After data cleaning, the last step is to understand the basic descriptive data of the dataset. In this example, I have used df[‘col1’].describe() to calculate the max, mean or count, etc. for the attributes, depending on the data types of the attributes, they have different results. I have also used df.std() to obtain the standard deviation for both price and points attributes. 

For example, for price and points, which are continuous variables, using describe(),  we were able to get the descriptive stats. On the left side, we can see the length, mean, std, min, max and the quantiles of points. We can see that the points wines in this dataset received ranged from 80 – 100, which was anticipated because Wine Enthusiast magazine only included wines that received 80 points or above in their reviews. We could see that the data has a relatively small standard deviation of 3.03, with the median around 89 points and mean at 88.8, which we could assume the data points are normally distributed. 

The table on the right side is the descriptive stats for price and we can see this attribute varied in a significant range from 4 – 3400 dollars. The most expensive bottle of wine is 3400 dollars. 

![image](https://user-images.githubusercontent.com/113401627/216464897-551fd80a-849f-4408-aee4-74f12e835419.png)
![image](https://user-images.githubusercontent.com/113401627/216464903-e948c61b-adc7-457e-b2b9-5de406cb927f.png)

We also plotted the histogram for the distribution of points and price in order to understand the data more. In the following section, 
Treatment for the points attribute: For points the wines in this dataset received, as one can see, the points that wines in the dataset received are normally distributed, with its peak around 88 points. 

![image](https://user-images.githubusercontent.com/113401627/216464932-8065b614-16fa-40a8-a7e4-8e0965d9971b.png)

It is clear that the wine review points in this dataset is somewhat normally distributed. It has its peak between around 87-89. The mean of the points given to the varieties was around 87.9. The maximum was 100 which was very few. The minimum was 80, which was again very few. And the Standard deviation of 3.22 suggests that the points given to each wine was 88 (approx.) +/- 3 which means anything ranging from 85 to 91 which is a very good score.

Based on the results, we could divide the variable of Points (continuous) into a categorical variable when conducting classification tasks: 

•	80 to 85 points = "average"
•	85 to 90 points = "good"
•	90 to 95 points = "great"
•	95 to 100 points = "excellent

Using the following code, we created a new column called Quality to house the four categorical labels of all wines based on the points they received. 

![image](https://user-images.githubusercontent.com/113401627/216464973-102f9e23-ad61-4538-bc00-503345683a62.png)

We also graphed a pie chart to visualize the number of data in each category. We can see that the majority of the data has fallen into good, great, and excellent, and only a few were categorized in average. 

![image](https://user-images.githubusercontent.com/113401627/216464992-33ee4ed7-b17a-40d2-84f2-d7b9962732a5.png)

Treatment for the price attribute: Whereas for the price distribution of wines in the data, we can see that the data is highly skewed to the right. The number of high-priced wines is barely invisible in the histogram. When zooming in to only visualize the distribution of wine price under 100 dollars, we can clearly see the data is positive skewed with most of the wine at 15-25 range and the number gradually declined from 40 – 100 dollars. 

![image](https://user-images.githubusercontent.com/113401627/216465039-0d9a71d9-d3c3-4e66-8196-2c77989add0e.png)

The most common price point for the wines in the dataset is around 20 dollars. So based on the descriptive analysis of the Price, the mean is around 30 which is much below 50. The data also has a very high standard deviation i.e. 34 so the prices vary from 4 to 60 dollars. 

Based on this information, we divided up the price into a categorical variable as follows:

•	0-10
•	10-20
•	20-30
•	30-50
•	50-100
•	Above 100

Using the following codes, we were able to create a new column called Price_val: 

![image](https://user-images.githubusercontent.com/113401627/216465114-4c723906-82fd-44eb-a70b-15449eb840c1.png)

After dividing up the price column, we did a pie chart to visualize the distribution of the price category. We can see that 1/3 of the data fell into the category above 100 dollars and the rest were equally distributed into the 10-100 range. Only a small proportion of the data has $0-10 price tags. 

![image](https://user-images.githubusercontent.com/113401627/216465212-9bd4d1ee-f1c4-4842-9732-0b88bbf082cf.png)

Treatment of variable attribute: The last variable that we did analysis on was the variety of wine. In total, there were 643 varieties of wine, and the below is a bar plot of the counts of varieties of wines that have over 1000 reviews only because it was impossible to plot all the wines that have one or two entries of reviews only. 

In addition, the first four that had over 10k reviews each were: Chardonnay, Pinot Noir, Cabernet Sauvignon, and Red Blend. We subset these four top-reviewed wines later in our classification models. 

![image](https://user-images.githubusercontent.com/113401627/216465315-06122e29-ae1b-467c-ab9e-f11e436fa023.png)

Word frequency distribution 

In this section, we discuss the distribution of unigrams frequencies, 20 most frequent nouns, 20 most frequent adjectives, 20 most frequent noun phrases and 20 most frequent adjective phrases. Overall, the data contained 3,441,539 tokens, just over 3.4 million tokens after we removed all punctuations, stop words, and numbers, which is quite impressive for its size. There were 1,980,620 tokens of nouns in the reviews, almost 2 million tokens. Adjectives are fewer in the corpus, which is just under 1 million (981,885) tokens. 

Then we did a frequency distribution for all the categories of tokens we extracted from the wine reviews. The following is a brief discussion of the features: 
For the 20 most frequent unigrams, we can see that fruit and finish are the two most frequent words, followed by aromas, acidity, and tannings. These are common descriptors for wines, which is not surprising. In addition, the majority of the most frequently-used words are either nouns or adjectives. Then we started to see more descriptors specific to certain wines, e.g. cherry, ripe, black, dry, spice, etc. There are two assumptions that could help us with the classification tasks: 

(1)	Nouns and adjectives are more important than the other POS in the wine review classification.

(2)	The results might indicate that reviews could work well with classifications of wine varieties, because the descriptors get more specific to describe certain types of wines going down the tokenized word list. 

We also plotted a word cloud to visualize the most common tokens in this corpus and we can see the most frequent words in the biggest 

![image](https://user-images.githubusercontent.com/113401627/216465433-3d37d95f-45d5-4566-9207-812732c72e2b.png)
![image](https://user-images.githubusercontent.com/113401627/216465439-4bf95c29-db56-4549-93cd-7a866a861b94.png)

For the 20 most common nouns, we can see a lot of overlapping in this list compared to the tokenized words. Overall, we can see that the frequent nouns have a huge overlap with the most 20 common unigram tokens, in which fruit and finish ranked 3rd and 4th in this frequent noun list, tannins, cherry, palate, spice, notes, and oak as well are overlapping with the word tokens. We suspect that this would cause the performance of using nouns as input for the classification to yield comparable results as using the tokenized words as input. 

Since we did not remove any stop words when POS tagging the sentences, there are a few words that could be considered stop words that could be removed from the results, e.g. wine, and flavors. Also, interestingly, the sign “%” was tagged as nouns. We think the NLTK POS tagging performed fine, but there were some inconsistency, as we could see the mismatches between the numbers of nouns and the number of tokens of the same words, e.g. (finish as a noun has 33,140 tokens and finish has a total of 33,7724 tokens and some of them might be verbs); it would be nice if we could do the backward tagging to remove some of these words, which might yield better results extracting the nouns. Here is the top 20 frequent noun list and the word cloud side by side. The size of the fonts in the word cloud indicates the frequency of the nouns in the corpus. The larger the fonts, the more frequent the nouns. 

![image](https://user-images.githubusercontent.com/113401627/216465499-955280d2-2aa0-4815-ab24-16916e68792c.png)
![image](https://user-images.githubusercontent.com/113401627/216465506-982236b3-2148-4e67-a7ab-f2240ad8a1b1.png)

For the 20 most frequent adjectives, we can see that “black” and “dry” are the most frequent adjectives, which overlapped with the tokenized words list. The adjective list contained mostly descriptors of the color and taste of wines. However, “ripe” was not at the top of the list with only 17069 tokens, as compared to 26720 tokens in the unigram, which was quite surprising, because ripe can only be used as adj. In the word cloud, we can see the black, fresh, dry, and red and rich are the most frequent adjectives based on the size of their fonts. 

![image](https://user-images.githubusercontent.com/113401627/216465534-39b04bd5-dbfd-4a7c-b825-990c85ce9c2c.png)
![image](https://user-images.githubusercontent.com/113401627/216465541-2de2ecb9-34fb-46c3-82ee-bb115fcf29ca.png)

For the 20 most frequent bigrams, we see that the most frequent bigram two types (1) wine descriptors regarding their flavors: “black cherry”, “black fruit”, “ripe fruit”, “berry fruit”, “tropical fruit”, “red pepper”, “green apple” etc, and (2) wine varieties: cabernet sauvignon, pinot noir, sauvignon blank.

![image](https://user-images.githubusercontent.com/113401627/216465565-fd89fca6-b26e-40bd-ad6c-7ab919fa8703.png)

Step 3: Generate Word Clouds

We also wanted to have some visualization of the word frequencies, so we generated some word clouds with noun and adjective phrases that we extracted from all wine reviews. By using the matplotlib, numpy, pandas and wordcloud packages, we created basic word cloud images. As the default background is set to black, we changed it to white in order to make the cloud easier to read.

![image](https://user-images.githubusercontent.com/113401627/216465642-650902eb-c879-4085-bd2f-2961d54b8c42.png)
![image](https://user-images.githubusercontent.com/113401627/216465653-5a019ce0-9167-4b39-ae7f-51da978b0bea.png)

From there, we masked the word clouds into a shape by converting jpeg images into a numpy arrays, changed the text color themes and the amount of text within each image. 

For the adjective phrases word cloud, we used a wine bottle and glass jpeg, set the maximum word count to 250 words, changed the text color pattern using the ‘twilight’ colormap theme from the matplotlib library, and used a light grey contour color to mimic the appearance of white wine.

![image](https://user-images.githubusercontent.com/113401627/216465713-d9bcd816-3fa0-4554-8128-d3fc37c9145d.png)

The same method was used to configure the word cloud for noun phrases. We used a different image to create the shape, set the maximum word count at 300 words, used the ‘RdGy’ colormap, and made the contour a dark red hue to imitate bottles of red wine.

![image](https://user-images.githubusercontent.com/113401627/216465736-c7ee5a06-673b-43d8-b0da-e95d80ab2fa5.png)

Findings

By using word clouds, we were able to visualize the sentiment, commonalities and themes of the wine reviews in our dataset. The different sizes and colors of text represent the prevalence and importance of commonly used words and phrases. The most used adjectives revolved around taste followed by sentiment rather than other attributes such as appearance and smell. Sweet and dry tend to be used when pairing wine with food, while terms like rich and soft refer to the level of complexity and fullness. Other terms such as good and almost could insinuate the reviewer's sentiment about their experience purchasing wine or how they felt about the wine itself. As for the most frequently used nouns, there appeared to be more variety among the types of words used. For obvious reasons, wine would be one of the most commonly used nouns, as well as the terms finish and palate since they are the most significant characteristic in assessing wine quality. Overall, we found the word clouds to be highly effective in visualizing text data to get a general sense of all that was written in the reviews. 

Step 4: Classification Tasks 

In this step, we discuss our results with the classification models we did. 

As we mentioned in step 3 of data descriptive analysis, it would be infeasible to classify the review data using individual points and price, since there was a huge variation among the numeric variables. Also, since the points and price ranges can be vast, it would be more helpful to give the range specific values to each one. Therefore, we made the decision to turn the continuous variables of “price” and “points” into categorical variables. These new columns are defined as wine_quality and price_val.  make running our classification tools run without any errors. 

We used 2 separate dataframes to see different classification results. The first dataframe used is “reviewdf2”, and the second dataframe used is “classifydf”. The reason for using different dataframes is because we wanted to test the classification models on both the pretokenized dataframe (reviewdf2) and the tokenized dataframe (classifydf).  Most of the results were most successful using the tokenized dataframe, which is what we included in our final Jupyter notebook for all the codes and in this report. 

We used the Naïve Bayes model for our classification tasks. To begin we created a make_pipeline that has the TF-IDF vectorizer and Multinomial Naïve Bayes functions inside. This was the main model that we used for our classification of various variables and our code for modeling: 

![image](https://user-images.githubusercontent.com/113401627/216465824-91b26c39-a43a-41de-ae0d-b7e663fc61e5.png)

We chose to use TF-IDF vectorizer as we thought it was a better option than Count Vectorizer. TF-IDF means Term Frequency - Inverse Document Frequency, which is statistic based on the frequency of a word in the corpus. In addition, it provides a numerical representation of the importance of a word for statistical analysis. TF-IDF is better than Count Vectorizers because it not only focuses on the frequency of words present in the corpus but also provides the importance of the words. Words that were less important for analysis were removed, hence making the model building less complex by reducing the input dimensions. 

For each model training, we first built a new dataframe to only include the two columns that we’d like to model. The example below is our first model we ran with  “word_token_cleaned” and “Quality”. Then we used the following codes to split the train test set by 70% and 30% and randomized the dataset. 

![image](https://user-images.githubusercontent.com/113401627/216465852-371b33d2-6b48-4197-9f6b-562227e74ef4.png)

Next, we trained the model with x_train and y_train and made predictions with x_test set. Lastly, we calculated the accuracy score for the classification model and the model accuracy score which was presented in percentage. 

![image](https://user-images.githubusercontent.com/113401627/216465891-e3afdacc-f985-48ef-9be5-4ce3b559d527.png)

We also ran several codes to obtain the classification results: 

(1)	Cross-validation scores with 10 kfolds: 

![image](https://user-images.githubusercontent.com/113401627/216465930-f53de970-9f41-4d0f-b5f9-a6fd1e4b739a.png)

(2)	Cross-validation scores with 5 kfolds:

![image](https://user-images.githubusercontent.com/113401627/216465945-5566d727-83fc-4b06-bb9e-51e3e1936f74.png)

(3) Classification report that contained precision, recall, f1-score, and support scores for the said model.

![image](https://user-images.githubusercontent.com/113401627/216465976-dd7e1a2f-9d5c-4815-a719-0c66f3356cf4.png)

After building the model, we first tested the model by classifying wine_quality with the variety of wines, just to test if the model would run smoothly without error codes. Of course, that is not any of the models we intended to run, but still, we can see the results are less than favorable. 

![image](https://user-images.githubusercontent.com/113401627/216466003-5cf59e19-8ab7-4d85-a71c-6e5896ba0424.png)
![image](https://user-images.githubusercontent.com/113401627/216466006-0f8a37e3-183c-4b4b-9f70-6b730237dfc3.png)

With a k-fold of 5 the average cross-validation score was only 0.5559. As one can tell, the prediction was not able to predict any of the “Excellent” scores given for the classification. This is a problem that we did not want, so we tried a multitude of different classification predictions using various variables to try and find the best accuracy we can get. 

In the following sections, we will report the best scores of three groups of modeling we did using tokenized wine reviews as input and wine quality (categorized points), price_val (categorized price), and wine variety. We will report our results from the worst performed model with reviews to predict wine price to the best performed model with reviews to predict wine varieties. 

Classification using reviews to predict wine price_val

In this section, we will report the classification results of wine reviews to predict the labeling of wine price, using the model Nicholas has built using TFIDF vectorizer. As we mentioned before, we categorized the price that each wine received into 6 categories “0-10”, “10-20”, “20-30”, “30-50”, “50-100” and “Above 100”.  We also preprocessed the wine reviews to only contain tokenized words (with all punctuations, numbers, and stop words removed). Then we carried out three models, all using Naïve Bayes algorithm, to see which texts helped best predict the price of wine. 

(1)	Modeling with tokenized words and wine price using Naïve Bayes
(2)	Modeling with nouns and wine price oing Naïve Bayes
(3)	Modeling with adjectives and wine price using Naïve Bayes

For this modeling, we had the best score with tokenized words and price_val. With a wide range of prices and the number of words used in the reviews we knew the accuracy for this specific classification would be rather low. Also, due to the imbalance in the dataset (very small number of above 100 dollars in the dataset), we have 0 in precision score and f1-score in the category of “Above 100 dollars”. 

![image](https://user-images.githubusercontent.com/113401627/216466058-67663a5e-c76e-46c4-85ce-651ca4b6ed82.png)

Similarly, none of the parts of speech (nouns and adjectives) performed well either. In fact, they had lower accuracy scores compared to just using tokenized words, as we can see from the snapshot of the results from the adjectives predicting wine prices. Interestingly, this time we were able to capture the precision scores for all categories and in fact, adjectives have the highest precision score in predicting price over 100 dollars. 

![image](https://user-images.githubusercontent.com/113401627/216466086-6e895311-08f4-4567-ae31-a1e9702e4a43.png)

Classification using reviews to predict wine quality 

In this section, we will report the classification results of wine reviews to predict the labeling of wine quality, using the model Nicholas has built using TFIDF vectorizer. As we mentioned before, we categorized the points that each wine received into 4 categories “average”, “good”, “great”, and “excellent”. We also preprocessed the wine reviews to only contain tokenized words (with all punctuations, numbers, and stop words removed). Then we carried out three models, all using Naïve Bayes algorithm, to see which texts helped best predict the quality of wine. 

(4)	Modeling with tokenized words and wine quality using Naïve Bayes
(5)	Modeling with nouns and wine quality using Naïve Bayes
(6)	Modeling with adjectives and wine quality using Naïve Bayes 

Overall, using wine reviews, we got an accuracy score ranging from 62.85% (nouns to predict wine quality) to 67.75% (tokenized words to predict wine quality), with tokenized words performed the best in predicting the wine quality. The precision scores for each wine quality label can be found in the classification report below. We did run into the issue with having 0 in the Excellent category, which means there was not enough data in this category to support the classification. 

![image](https://user-images.githubusercontent.com/113401627/216466130-2c6995e6-7e67-491f-90dd-acf207026656.png)

However, this is only the average performing model in our experiment. 

Classification using tokenized words to predict wine variety

Lastly, we ran three models to use wine reviews to predict the top 4 varieties of wines. As we mentioned in the descriptive data report, there were over 600 varieties of wines, and it would be impossible to model the data with so many vectors. Therefore, to make the classification tasks easier, we only used the top four varieties of wine, namely “Cabernet Sauvignon”, “Chardonnay”, “Pinot Noir”, and “Red Blend”, because these 4 had the most reviews at over 10,000. 

(7)	Modeling with tokenized words and top 4 wine varieties using Naïve Bayes
(8)	Modeling with nouns and top 4 wine varieties using Naïve Bayes
(9)	Modeling with adjectives and top 4 wine varieties using Naïve Bayes

Our highest score that we were able to achieve was an accuracy score of 88.75%. This came when we decided to use the word_token_cleaned variable with the Variety of wines. The classification report is also shown below. All categories of wine varieties received over 0.85 point in precision scores, with weighted average of 0.89, which is excellent outcome for us. 

![image](https://user-images.githubusercontent.com/113401627/216466170-c825469a-41e0-4537-806f-1c316b647e47.png)

This was exciting to see as the variety of wine was able to help coincide with the words they were being reviewed as at a high rate of prediction. Using the top 4 wine varieties also assisted in creating a high accuracy score for our model. Instead of using all the wine varieties, the top 4 varieties created a relatively balanced dataset and were able to predict their reviewed words.

Next, we wanted to compare classification results of nouns and adjectives to the top 4 wine varieties. We can see that the accuracy score for nouns predicting the varieties performed better in this model, compared to the adjectives. We suggested that it is possible that the tokenized words may have already contained some descriptors for wines that were easy to categorize, based on our observations in the word frequency distribution. When it comes to the parts of speech accuracy score, nouns had about a 17% higher score. This is surprising because there were more nouns than adjectives within our dataset, yet the prediction of the correct nouns to variety was higher than adjectives.

(1)	Accuracy score and classification report for nouns predicting wine varieties.

![image](https://user-images.githubusercontent.com/113401627/216466220-f6cc0784-25aa-4994-a04a-cd0de472fc5d.png)

(2)	 Accuracy score and classification report for adjective predicting wine varieties. 

![image](https://user-images.githubusercontent.com/113401627/216466241-0de83fa3-c75c-473e-9f44-8d3f749bb76a.png)

We suggested that the high accuracy could be explained by that fact the tokenized words contained certain descriptors that are used to describe certain types of wines, which helped when the algorithm tried to classify them into different wine varieties. 

SVM models with wine reviews and wine varieties (did not work out as expected)

As per the professor’s feedback, we tried again with svm algothrism with a smaller sample set. We took the best performing model with wine reviews predicting wine varieties and subset the data to 10% of its original size using the following codes. We shuffled the data so that it was randomized and include a balanced dataset of all four varieties of wines. 

![image](https://user-images.githubusercontent.com/113401627/216466263-d4c3cbb2-7c53-4f52-83db-8dd1a08d1d74.png)

The data included around 1000 reviews for each of the categories. 

After building the model with svm with “rbf” kernal, the results came out only 29% accurate and the precision scores were 0 in predicting 3 out of 4 wine varieties. Therefore, we felt the codes could be modified to get better output, but due to our limitations, we left this part out for the professor’s input (see the Jupyter notebook submission). 






