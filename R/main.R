library(MASS)

all_data = read.table('data.txt', header=FALSE, sep = "", dec=".")
all_data <- all_data[-c(365,369,372,373),]
y <- all_data[,14]
X <- all_data[,c(1:13)]
y = data.matrix(y)
X = data.matrix(X)
linear_model = lm(y~X)

all_data <- all_data[-c(367,365,368),]
y <- all_data[,14]
X <- all_data[,c(1:13)]
y = data.matrix(y)
X = data.matrix(X)
linear_model = lm(y~X)
#plot(linear_model)

all_data <- all_data[-c(405,373,407),]
y <- all_data[,14]
X <- all_data[,c(1:13)]
y = data.matrix(y)
X = data.matrix(X)
linear_model = lm(y~X)
#plot(linear_model)

bc <- boxcox(y~X)
lambda = with(bc, X[which.max(y)])
print(lambda)
plot(linear_model)
transformed_y = (y^lambda - 1) / lambda
linear_model = lm(transformed_y~X)
plot(fitted(linear_model), transformed_y)
