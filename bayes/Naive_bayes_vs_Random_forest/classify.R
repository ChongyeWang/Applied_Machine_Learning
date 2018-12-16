#################### Part 1 ####################


get_middle <- function(orig_matrix){
  #get the middle 20 * 20 square
  middle_vec = c()
  for(i in 5:24){
    for(j in 5:24){
      index = (i - 1) * 28 + j
      middle_vec <- c(middle_vec, orig_matrix[index])
    }
  }
  return (middle_vec)
}



get_mean_and_sd <- function(orig_matrix, label){
  #return the mean list and sd list 
  
  ##### append to matrix #####
  matrix0 = c()
  matrix1 = c()
  matrix2 = c()
  matrix3 = c()
  matrix4 = c()
  matrix5 = c()
  matrix6 = c()
  matrix7 = c()
  matrix8 = c()
  matrix9 = c()
  
  digit = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  
  for(row in 1:length(label)){
    
    if(label[row] == 0){
      matrix0 <- rbind(matrix0, as.matrix(orig_matrix[row, 1:400]))
      digit[1] <- digit[1] + 1
    }
    else if(label[row] == 1){
      matrix1 <- rbind(matrix1, as.matrix(orig_matrix[row, 1:400]))
      digit[2] <- digit[2] + 1
    }
    else if(label[row] == 2){
      matrix2 <- rbind(matrix2, as.matrix(orig_matrix[row, 1:400]))
      digit[3] <- digit[3] + 1
    }
    else if(label[row] == 3){
      matrix3 <- rbind(matrix3, as.matrix(orig_matrix[row, 1:400]))
      digit[4] <- digit[4] + 1
    }
    else if(label[row] == 4){
      matrix4 <- rbind(matrix4, as.matrix(orig_matrix[row, 1:400]))
      digit[5] <- digit[5] + 1
    }
    else if(label[row] == 5){
      matrix5 <- rbind(matrix5, as.matrix(orig_matrix[row, 1:400]))
      digit[6] <- digit[6] + 1
    }
    else if(label[row] == 6){
      matrix6 <- rbind(matrix6, as.matrix(orig_matrix[row, 1:400]))
      digit[7] <- digit[7] + 1
    }
    else if(label[row] == 7){
      matrix7 <- rbind(matrix7, as.matrix(orig_matrix[row, 1:400]))
      digit[8] <- digit[8] + 1
    }
    else if(label[row] == 8){
      matrix8 <- rbind(matrix8, as.matrix(orig_matrix[row, 1:400]))
      digit[9] <- digit[9] + 1
    }
    else if(label[row] == 9){
      matrix9 <- rbind(matrix9, as.matrix(orig_matrix[row, 1:400]))
      digit[10] <- digit[10] + 1
    }
  }
  
  for(i in 1:10){
    digit[i] <- digit[i] / length(label)
  }
  
  matrix0 <- matrix(matrix0, ncol = 400, nrow = length(matrix0) / 400)
  matrix1 <- matrix(matrix1, ncol = 400, nrow = length(matrix1) / 400)
  matrix2 <- matrix(matrix2, ncol = 400, nrow = length(matrix2) / 400)
  matrix3 <- matrix(matrix3, ncol = 400, nrow = length(matrix3) / 400)
  matrix4 <- matrix(matrix4, ncol = 400, nrow = length(matrix4) / 400)
  matrix5 <- matrix(matrix5, ncol = 400, nrow = length(matrix5) / 400)
  matrix6 <- matrix(matrix6, ncol = 400, nrow = length(matrix6) / 400)
  matrix7 <- matrix(matrix7, ncol = 400, nrow = length(matrix7) / 400)
  matrix8 <- matrix(matrix8, ncol = 400, nrow = length(matrix8) / 400)
  matrix9 <- matrix(matrix9, ncol = 400, nrow = length(matrix9) / 400)
  
  
  ##### append mean and standard deviation #####
  curr_feature_mean0 = c()
  curr_feature_mean1 = c()
  curr_feature_mean2 = c()
  curr_feature_mean3 = c()
  curr_feature_mean4 = c()
  curr_feature_mean5 = c()
  curr_feature_mean6 = c()
  curr_feature_mean7 = c()
  curr_feature_mean8 = c()
  curr_feature_mean9 = c()
  
  curr_feature_sd0 = c()
  curr_feature_sd1 = c()
  curr_feature_sd2 = c()
  curr_feature_sd3 = c()
  curr_feature_sd4 = c()
  curr_feature_sd5 = c()
  curr_feature_sd6 = c()
  curr_feature_sd7 = c()
  curr_feature_sd8 = c()
  curr_feature_sd9 = c()
  
  mean_vec0 = c()
  sd_vec0 = c()
  mean_vec1 = c()
  sd_vec1 = c()
  mean_vec2 = c()
  sd_vec2 = c()
  mean_vec3 = c()
  sd_vec3 = c()
  mean_vec4 = c()
  sd_vec4 = c()
  mean_vec5 = c()
  sd_vec5 = c()
  mean_vec6 = c()
  sd_vec6 = c()
  mean_vec7 = c()
  sd_vec7 = c()
  mean_vec8 = c()
  sd_vec8 = c()
  mean_vec9 = c()
  sd_vec9 = c()
  
  
  for(i in 1:400){
   
    curr_feature0 <- matrix0[, i]

    curr_feature_mean0 <- mean(curr_feature0)
    curr_feature_sd0 <- sd(curr_feature0)
    mean_vec0 <- c(mean_vec0, curr_feature_mean0)
    sd_vec0 <- c(sd_vec0, curr_feature_sd0)
    
    curr_feature1 <- matrix1[,i]
    curr_feature_mean1 <- mean(curr_feature1)
    curr_feature_sd1 <- sd(curr_feature1)
    mean_vec1 <- c(mean_vec1, curr_feature_mean1)
    sd_vec1 <- c(sd_vec1, curr_feature_sd1)
    
    curr_feature2 <- matrix2[,i]
    curr_feature_mean2 <- mean(curr_feature2)
    curr_feature_sd2 <- sd(curr_feature2)
    mean_vec2 <- c(mean_vec2, curr_feature_mean2)
    sd_vec2 <- c(sd_vec2, curr_feature_sd2)
    
    curr_feature3 <- matrix3[,i]
    curr_feature_mean3 <- mean(curr_feature3)
    curr_feature_sd3 <- sd(curr_feature3)
    mean_vec3 <- c(mean_vec3, curr_feature_mean3)
    sd_vec3 <- c(sd_vec3, curr_feature_sd3)
    
    curr_feature4 <- matrix4[,i]
    curr_feature_mean4 <- mean(curr_feature4)
    curr_feature_sd4 <- sd(curr_feature4)
    mean_vec4 <- c(mean_vec4, curr_feature_mean4)
    sd_vec4 <- c(sd_vec4, curr_feature_sd4)
    
    curr_feature5 <- matrix5[,i]
    curr_feature_mean5 <- mean(curr_feature5)
    curr_feature_sd5 <- sd(curr_feature5)
    mean_vec5 <- c(mean_vec5, curr_feature_mean5)
    sd_vec5 <- c(sd_vec5, curr_feature_sd5)
    
    curr_feature6 <- matrix6[,i]
    curr_feature_mean6 <- mean(curr_feature6)
    curr_feature_sd6 <- sd(curr_feature6)
    mean_vec6 <- c(mean_vec6, curr_feature_mean6)
    sd_vec6 <- c(sd_vec6, curr_feature_sd6)
    
    curr_feature7 <- matrix7[,i]
    curr_feature_mean7 <- mean(curr_feature7)
    curr_feature_sd7 <- sd(curr_feature7)
    mean_vec7 <- c(mean_vec7, curr_feature_mean7)
    sd_vec7 <- c(sd_vec7, curr_feature_sd7)
    
    curr_feature8 <- matrix8[,i]
    curr_feature_mean8 <- mean(curr_feature8)
    curr_feature_sd8 <- sd(curr_feature8)
    mean_vec8 <- c(mean_vec8, curr_feature_mean8)
    sd_vec8 <- c(sd_vec8, curr_feature_sd8)
    
    curr_feature9 <- matrix9[,i]
    curr_feature_mean9 <- mean(curr_feature9)
    curr_feature_sd9 <- sd(curr_feature9)
    mean_vec9 <- c(mean_vec9, curr_feature_mean9)
    sd_vec9 <- c(sd_vec9, curr_feature_sd9)
    
  }

  feature_mean_list <- c()
  feature_sd_list <- c()
  
  feature_mean_list <- rbind(feature_mean_list, mean_vec0)
  feature_mean_list <- rbind(feature_mean_list, mean_vec1)
  feature_mean_list <- rbind(feature_mean_list, mean_vec2)
  feature_mean_list <- rbind(feature_mean_list, mean_vec3)
  feature_mean_list <- rbind(feature_mean_list, mean_vec4)
  feature_mean_list <- rbind(feature_mean_list, mean_vec5)
  feature_mean_list <- rbind(feature_mean_list, mean_vec6)
  feature_mean_list <- rbind(feature_mean_list, mean_vec7)
  feature_mean_list <- rbind(feature_mean_list, mean_vec8)
  feature_mean_list <- rbind(feature_mean_list, mean_vec9)
  
  feature_sd_list <- rbind(feature_sd_list, sd_vec0)
  feature_sd_list <- rbind(feature_sd_list, sd_vec1)
  feature_sd_list <- rbind(feature_sd_list, sd_vec2)
  feature_sd_list <- rbind(feature_sd_list, sd_vec3)
  feature_sd_list <- rbind(feature_sd_list, sd_vec4)
  feature_sd_list <- rbind(feature_sd_list, sd_vec5)
  feature_sd_list <- rbind(feature_sd_list, sd_vec6)
  feature_sd_list <- rbind(feature_sd_list, sd_vec7)
  feature_sd_list <- rbind(feature_sd_list, sd_vec8)
  feature_sd_list <- rbind(feature_sd_list, sd_vec9)

  result_mean_list <- c()
  result_sd_list <- c()

  result_mean_list <- c(result_mean_list, feature_mean_list)
  result_sd_list <- c(result_sd_list, feature_sd_list)
  
  result_mean_sd_list = c()
  result_mean_sd_list <- c(result_mean_sd_list, result_mean_list)
  result_mean_sd_list <- c(result_mean_sd_list, result_sd_list)
  result_mean_sd_list <- c(result_mean_sd_list, digit)
  
  return (result_mean_sd_list)
  
}

normal_distribution_v_original_matrix <- function(test, test_label, mean_list, sd_list, digit_list){
  #get the accuracy using normal distribution and original matrix
  num_of_correct <- 0
  
  for(i in 1:length(test_label)){
    
    score0 <- log(digit_list[1])
    score1 <- log(digit_list[2])
    score2 <- log(digit_list[3])
    score3 <- log(digit_list[4])
    score4 <- log(digit_list[5])
    score5 <- log(digit_list[6])
    score6 <- log(digit_list[7])
    score7 <- log(digit_list[8])
    score8 <- log(digit_list[9])
    score9 <- log(digit_list[10])
    
    test1 <- c()
    
    for(row in 1:length(test_label)){
      test1 <- rbind(test1, as.matrix(test[row, 1:400]))
    }

    
    test1 <- matrix(test1, ncol = 400, nrow = length(test_label))

   
    print(length(test1[1, ]))
    print(length(test1[, 1]))
    
    
    start0 <- 0
    start1 <- 400
    start2 <- 800
    start3 <- 1200
    start4 <- 1600
    start5 <- 2000
    start6 <- 2400
    start7 <- 2800
    start8 <- 3200
    start9 <- 3600
    
    mean0 <- mean_list[1:400]
    
    mean1 <- mean_list[401:800]
    mean2 <- mean_list[801:1200]
    mean3 <- mean_list[1201:1600]
    mean4 <- mean_list[1601:2000]
    mean5 <- mean_list[2001:2400]
    mean6 <- mean_list[2401:2800]
    mean7 <- mean_list[2801:3200]
    mean8 <- mean_list[3201:3600]
    mean9 <- mean_list[3601:4000]
    
    sd0 <- sd_list[1:400]
    
    sd1 <- sd_list[401:800]
    sd2 <- sd_list[801:1200]
    sd3 <- sd_list[1201:1600]
    sd4 <- sd_list[1601:2000]
    sd5 <- sd_list[2001:2400]
    sd6 <- sd_list[2401:2800]
    sd7 <- sd_list[2801:3200]
    sd8 <- sd_list[3201:3600]
    sd9 <- sd_list[3601:4000]
    
    
    for(j in 1:400){
      score0 <- score0 + log(pnorm(test1[i, j], mean = mean0[j], sd = sd0[j]))
      
      score1 <- score1 + log(pnorm(test1[i, j], mean = mean1[j], sd = sd1[j]))
      score2 <- score2 + log(pnorm(test1[i, j], mean = mean2[j], sd = sd2[j]))
      score3 <- score3 + log(pnorm(test1[i, j], mean = mean3[j], sd = sd3[j]))
      score4 <- score4 + log(pnorm(test1[i, j], mean = mean4[j], sd = sd4[j]))
      score5 <- score5 + log(pnorm(test1[i, j], mean = mean5[j], sd = sd5[j]))
      score6 <- score6 + log(pnorm(test1[i, j], mean = mean6[j], sd = sd6[j]))
      score7 <- score7 + log(pnorm(test1[i, j], mean = mean7[j], sd = sd7[j]))
      score8 <- score8 + log(pnorm(test1[i, j], mean = mean8[j], sd = sd8[j]))
      score9 <- score9 + log(pnorm(test1[i, j], mean = mean9[j], sd = sd9[j]))
    }
    

    
    score_list <- c(score0, score1, score2, score3, score4, score5, score6, score7, score8, score9)
  
    estimated <- 0
    
    curr_max <- score0
    count <- 0
    
    
  
    
    for(val in score_list){
      if(val > curr_max){
        estimated <- count
        curr_max <- val
      }
      count <- count + 1
    }
    
    real <- test_label[i]
    if(estimated == real){
      num_of_correct <- num_of_correct + 1
    }
  }
  overall_accuracy <- num_of_correct / length(test_label)
  print(overall_accuracy)
  return (overall_accuracy)
}



#read data from csv file
training_matrix_data <- read.csv(file="/Users/chongyewang/Desktop/hw1/Part2/train.csv", header=FALSE, sep=",")
testing_matrix_data <- read.csv(file="/Users/chongyewang/Desktop/hw1/Part2/test.csv", header=FALSE, sep=",")


#remove the first row 
training_matrix_data <- training_matrix_data[-1,]
testing_matrix_data <- testing_matrix_data[-1,]

#get the label for the training data and testing data
training_label = training_matrix_data[, 2]
testing_label = testing_matrix_data[, 2]


#remove the first and second column
training_matrix_data <- training_matrix_data[,-1]
training_matrix_data <- training_matrix_data[,-1]

testing_matrix_data <- testing_matrix_data[,-1]
testing_matrix_data <- testing_matrix_data[,-1]

training_matrix_data <- data.matrix(training_matrix_data)
testing_matrix_data <- data.matrix(testing_matrix_data,)

#the middle 20 * 20 matrix
training_20_20_matrix = c()
testing_20_20_matrix = c()


for(i in 1:length(training_matrix_data[,1])){  
  
  new_vec <- get_middle(training_matrix_data[i,])

  training_20_20_matrix <- rbind(training_20_20_matrix, new_vec)

}


mean_and_sd <- get_mean_and_sd(training_20_20_matrix, training_label)       


mean_list <- mean_and_sd[1:4000]
sd_list <- mean_and_sd[4001:8000]
digit_prob_list <- mean_and_sd[8001:8010]





for(i in 1:length(testing_matrix_data[,1])){         
  new_vec <- get_middle(testing_matrix_data[i,])
  testing_20_20_matrix <- rbind(testing_20_20_matrix, new_vec)
}

#get the normal distribution v original matrix overall accuracy       
normal_v_original_overall_accuracy <- normal_distribution_v_original_matrix(testing_20_20_matrix, testing_label, mean_list, sd_list, digit_prob_list)









