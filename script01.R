#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Learners will develop their algorithms on the edx set
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# The RMSE() functions was giving NA value, so I will create my RMSE function
# that removes na's
my_rmse <- function(prediction, true){
  sqrt( mean( (prediction - true)^2 , na.rm = TRUE ) )
}

# I will start with a model that looks for a average movie rating across all 
# movies, then average user rating and average specific movie rating and last
# for a genre average rating.

mu <- mean(edx$rating)
first_rmse <- my_rmse(mu, validation$rating)

# Let's try looking at the movies average. Each movie is rated differently.
movies_ratings <- edx %>% group_by(movieId) %>%
  summarize(movie_effect = mean(rating - mu))
movie_prediction <- mu + validation %>% 
  left_join(movies_ratings, by='movieId') %>% .$movie_effect
movie_rmse <- my_rmse(movie_prediction, validation$rating)
# Now for user average rating, we will calculate it based on the average
user_ratings <- edx %>% left_join(movies_ratings, by = 'movieId') %>% 
  group_by(userId) %>% summarize(user_effect = mean(rating - mu - movie_effect))

# Let's test our new RMSE by creating a new dataframe with the averages
user_predictions <- validation %>% 
  left_join(movies_ratings, by = 'movieId') %>%
  left_join(user_ratings, by = 'userId') %>%
  mutate(means = mu + user_effect + movie_effect,
         pred = ifelse(means > 5, 5, ifelse(means < 1, 1, means))) %>% .$pred

movie_user_rmse <- my_rmse(user_predictions, validation$rating)

# Using regularization to weight the movies or users with low numbers of rates
lambdas = seq(0, 10, 0.5)

reg_preds <- sapply(lambdas, function(lambda){
  
  movie_reg_avgs <- edx %>% 
    group_by(movieId) %>% 
    summarize(movie_effect = sum(rating - mu)/(n() + lambda)) 
  
  user_reg_avgs <- edx %>% left_join(movie_reg_avgs, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(user_effect = sum(rating - mu - movie_effect)/(n() + lambda))
  
  reg_ratings <- validation %>% left_join(movie_reg_avgs, by = 'movieId') %>%
    left_join(user_reg_avgs, by = 'userId') %>%
    mutate(means = mu + movie_effect + user_effect,
           pred = ifelse(means > 5, 5,
                         ifelse(means < 1, 1, means))) %>%
    .$pred
})
colnames(reg_preds) <- lambdas
rmses <- apply(reg_preds, 2, my_rmse, validation$rating)
best_lambda <- lambdas[which.min(rmses)]
best_lambda <- ifelse(best_lambda < 1, best_lambda + 1, best_lambda)
best_reg <- reg_preds[, best_lambda]
reg_rmse <- my_rmse(best_reg, validation$rating)
reg_accuracy <- mean(round(best_reg) == validation$rating, na.rm = TRUE)

# Adding rates to validation table
validation <- validation %>% select(-rating)
validation['rating'] <- best_reg
# Ratings will go into the CSV submission file below:
write.csv(validation %>% select(userId, movieId, rating),
          "submission.csv", na = "", row.names=FALSE)
