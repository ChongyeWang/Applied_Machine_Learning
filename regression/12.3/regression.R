library(glmnet)
setwd('/Users/chongyewang/Desktop/CS498/hw7/12.3/blogFeedback')

train <- read.csv("blogData_train.csv", header=TRUE, sep=",")
train_x <- as.matrix(train[, 1 : ncol(train) - 1])
train_y <- as.matrix(train[,ncol(train)])
print(dim(train_x))
print(dim(train_y))
fit12.3 <- cv.glmnet(train_x, train_y, family = "poisson", alpha = 1)
plot(fit12.3)
pred_y <- predict(fit12.3, train_x, s='lambda.1se', type = "response")
plot(train_y,pred_y)



test_name=Sys.glob(file.path("/Users/chongyewang/Desktop/CS498/hw7/12.3/blogFeedback","blogData_test*.csv"))
merge.data = read.csv(file = test_name[1],header=F,sep=",") 
n=length(test_name)
for (i in 2:n){
  new.data = read.csv(file = test_name[i], header=F, sep=",")
  merge.data = rbind(merge.data,new.data)
}
test_x <- as.matrix(merge.data[,1:ncol(merge.data) - 1])
test_y <- as.matrix(merge.data[, ncol(merge.data)])

pred_test_y <- predict(fit12.3, test_x, s = fit12.3$lambda.1se, type = "response")
plot(test_y, pred_test_y)