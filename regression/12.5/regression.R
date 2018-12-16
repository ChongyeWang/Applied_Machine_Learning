library(glmnet)
setwd('/Users/chongyewang/Desktop/CS498/hw7/12.5')

crusio  <- read.csv("Crusio1.csv", header=TRUE, sep=",")
c_data=cbind(crusio$sex, crusio[,4:41])
dim(c_data)
f_data=c_data[complete.cases(c_data),] 
dim(f_data)
data_x=as.matrix(f_data[,-1])
data_y=as.matrix(f_data[,1])
fit12.5 = cv.glmnet(data_x, data_y, family = "binomial", type.measure="class", alpha = 1)
plot(fit12.5)
pred_sex=predict(fit12.5, data_x, type='class', s='lambda.1se')
accuracy_sex= sum (pred_sex== data_y )/dim(data_y)[1]
print(accuracy_sex)
f <- 0
m <- 0
for (sex in data_y){
  if (sex == 'f'){
    f <- f + 1
  }
  else{
    m <- m + 1
  }
}
baseline <- 0
if (m > f){
  baseline <- m / (m + f)
}else{
  baseline <- f / (m + f)
}

library(glmnet)
setwd('/Users/chongyewang/Desktop/CS498/hw7/12.5')

crusio  <- read.csv("Crusio1.csv", header=TRUE, sep=",")

strain_data = cbind(crusio$strain,crusio[,4:41])
s_data = strain_data[complete.cases(strain_data),]
strain_table = as.data.frame(table(s_data[,1]))
strain_table = strain_table [which(strain_table$Freq>=10),]
y = as.vector(s_data[,1]) %in% as.vector(strain_table$Var1)
filt_data=s_data[y,]
strain_x=as.matrix(filt_data[,-1])
strain_y=as.matrix(filt_data[,1])
print(strain_y)
fit12.5b = cv.glmnet(strain_x, strain_y, family = "multinomial", type.measure="class", alpha = 1)
plot(fit12.5b)
pred_strain=predict(fit12.5b, strain_x, type='class', s='lambda.1se')
accuracy_strain= sum (pred_strain== strain_y )/dim(strain_y)[1]
print(accuracy_strain)

