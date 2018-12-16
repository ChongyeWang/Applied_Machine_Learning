library(glmnet)
setwd('/Users/chongyewang/Desktop/CS498/hw7/12.4')


data <- read.csv("I2000.csv", header=FALSE, sep=",")
print(dim(data))
tissue <- read.csv("Tissues.csv", header=FALSE, sep=",")
data=as.matrix(t(data))
print(dim(data))
tissue=as.matrix(tissue/abs(tissue))

fit12.4 = cv.glmnet(data, tissue, family = "binomial", type.measure="class", alpha = 1)
plot(fit12.4)
pred_t=predict(fit12.4, data, type='class', s='lambda.1se')
print(pred_t)
accuracy= sum (pred_t== tissue )/dim(tissue)[1]
print(accuracy)