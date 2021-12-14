library(glmnet)
library(pROC)
library(sparseSVM)
library(caret)
set.seed(844)
#load data
xtrain = as.matrix(read.csv("xtrain.csv")[,2:8035])
xtest = as.matrix(read.csv("xtest.csv")[,2:8035])
ytrain = read.csv("ytrain.csv")[,2]
ytest = read.csv("ytest.csv")[,2]
#cv for lambda for logistic regression with lasso
cvlasso = cv.glmnet(xtrain, ytrain, family = "binomial",alpha = 1, type.measure = "auc", nfolds= 5, lambda = seq(0.01, 0.2, 0.05))
plot(cvlasso)
#lasso model
lasso = glmnet(xtrain, ytrain, family = "binomial",alpha = 1,standardize = F, lambda = cvlasso$lambda.min)
#performance on test
lassopred = predict(lasso, newx = xtest, type = 'link')
lassopredt = predict(lasso, newx = xtrain, type = 'link')
plot(roc(factor(ytest),lassopred))
plot(roc(factor(ytrain),lassopredt), add=T,col='red')
legend("bottomright",legend = c("test=0.934","train=0.942"),lty=1,col=c(1,"red"))
auc(ytrain,lassopredt)
auc(ytest,lassopred)
#variables selected
vlasso = 0
for (i in 1:length(coef(lasso))){
  if ((coef(lasso)[,1][i])!=0){vlasso=vlasso+1}
}

#top 10 genes
coef(lasso)[,1][order(abs(coef(lasso)[,1]))][8025:8035]

#cv for lambda for logistic regression with ridge
cvridge = cv.glmnet(xtrain, ytrain, family = "binomial",alpha = 0, type.measure = "auc", nfolds= 5, lambda = seq(0.1, 0.9, 0.1))
plot(cvridge)
#ridge model
ridge = glmnet(xtrain, ytrain, family = "binomial",alpha = 0,standardize = F, lambda = cvridge$lambda.min)
#performance on test
ridgepred = predict(ridge, newx = xtest, type = 'link')
ridgepredt = predict(ridge, newx = xtrain, type = 'link')
plot(roc(factor(ytest),ridgepred))
plot(roc(factor(ytrain),ridgepredt), add=T,col='red')
legend("bottomright",legend = c("test=0.93","train=0.977"),lty=1,col=c(1,"red"))
auc(ytrain,ridgepredt)
auc(ytest,ridgepred)

#variables selected
vridge = 0
for (i in 1:length(coef(ridge))){
  if ((coef(ridge)[,1][i])!=0){vridge=vridge+1}
}


#top 10 genes
coef(ridge)[,1][order(abs(coef(ridge)[,1]))][8025:8035]


#cv for lambda and alpha for logistic regression with elastic net

alpha = seq(0.01, 0.2, 0.05)
lambdas = numeric(length(alpha))
cvms = numeric(length(alpha))
for (i in 1:length(alpha)) {
  model = cv.glmnet(xtrain, ytrain,type.measure = "auc", nfolds= 5,
                    family = "binomial",alpha = alpha[i])
  lambdas[i] = model$lambda.min
  cvms[i] = max(model$cvm)
}
lambdas[which.max(cvms)]
alpha[which.max(cvms)]
# cv
plot(x = c(seq(0.01, 0.2, 0.05)), y = cvms,ylab = "auc",xlab = "alpha",type = "h")
points(x = c(seq(0.01, 0.2, 0.05)),y=cvms, pch=19)

#elastic net model
elas = glmnet(xtrain, ytrain,standardize = F,
              alpha = alpha[which.max(cvms)], lambda = lambdas[which.max(cvms)], 
              family = "binomial")
#performance on test
elasepred = predict(elas, newx = xtest, type = 'link')
elaspredt = predict(elas, newx = xtrain, type = 'link')
plot(roc(factor(ytest),elasepred))
plot(roc(factor(ytrain),elaspredt), add=T,col='red')
legend("bottomright",legend = c("test=0.936","train=0.965"),lty=1,col=c(1,"red"))
auc(ytrain,elaspredt)
auc(ytest,elasepred)

plot(x=c(seq(0.01, 0.2, 0.05)), y = cvms,ylab = "auc",xlab = "alpha",type="h")


#variables selected
velas = 0
for (i in 1:length(coef(elas))){
  if ((coef(elas)[,1][i])!=0){velas=velas+1}
}

#top 10 genes
coef(elas)[,1][order(abs(coef(elas)[,1]))][8024:8035]




#comparison 
plot(x = c(1,2,3),y=c(as.numeric(auc(ytest,lassopred)), as.numeric(auc(ytest,ridgepred)), as.numeric(auc(ytest,elasepred))),ylim = c(0.93,0.94),ylab="auc", xlab="",xaxt="n",type = "h")
points(x = c(1,2,3),y=c(as.numeric(auc(ytest,lassopred)), as.numeric(auc(ytest,ridgepred)), as.numeric(auc(ytest,elasepred))), pch=19)
text(x = c(1,2,3),y=c(as.numeric(auc(ytest,lassopred)), as.numeric(auc(ytest,ridgepred)), as.numeric(auc(ytest,elasepred))),
     labels = c(round(auc(ytest,lassopred),3), round(auc(ytest,ridgepred),3), round(auc(ytest,elasepred),2)),
     pos=3,cex= 0.9)
axis(1, at=1:3, labels=c("lasso","ridge","elastic net"))







#cv for lambda for svm with lasso
cvlassos = cv.sparseSVM(xtrain, ytrain,alpha=1,lambda = seq(0.1, 0.9, 0.1),nfolds = 5)
plot(cvlassos)
#lasso model
lassos = sparseSVM(xtrain, ytrain,alpha=1,lambda = cvlassos$lambda.min, preprocess = "none")
#performance on test
lassospred = predict(lassos, X=xtest)[,1]
acclassos = sum(lassospred == ytest)/length(ytest)

#variables selected
vlassos = 0
for (i in 1:length(coef(lassos))){
  if ((coef(lassos)[,1][i])!=0){vlassos=vlassos+1}
}
#confusion matrix

confusionMatrix(factor(lassospred), factor(ytest))


#cv for lambda for svm with ridge
cvridges = cv.sparseSVM(xtrain, ytrain,alpha=0,lambda = seq(0.1, 0.9, 0.1),nfolds = 5)
plot(cvridges)
#lasso model
ridges = sparseSVM(xtrain, ytrain,alpha=0,lambda = cvridges$lambda.min, preprocess = "none")
#performance on test
ridgespred = predict(ridges, X=xtest)[,1]
accridges = sum(ridgespred == ytest)/length(ytest)

#variables selected
vridges = 0
for (i in 1:length(coef(ridges))){
  if ((coef(ridges)[,1][i])!=0){vridges=vridges+1}
}
#confusion matrix

confusionMatrix(factor(ridgespred), factor(ytest))

#cv for lambda and alpha for svm with elastic net

alpha = seq(0.1, 0.9, 0.1)
lambdas = numeric(length(alpha))
cve = numeric(length(alpha))
for (i in 1:length(alpha)) {
  model = cv.sparseSVM(xtrain, ytrain,alpha=alpha[i],nfolds = 5)
  lambdas[i] = model$lambda.min
  cve[i] = min(model$cve)
}
lambdas[which.min(cve)]
alpha[which.min(cve)]
# cv
plot(x = alpha, y = cve,ylab = "cv error",xlab = "alpha",type = "h")
points(x = alpha,y=cve, pch=19)


#elastic net model
elass = sparseSVM(xtrain, ytrain,preprocess = "none",
                  alpha = alpha[which.min(cve)], lambda = lambdas[which.min(cve)])
#performance on test
elasspred = predict(elass, X=xtest)[,1]
accelass = sum(elasspred == ytest)/length(ytest)

#variables selected
velass = 0
for (i in 1:length(coef(elass))){
  if ((coef(elass)[,1][i])!=0){velass=velass+1}
}
#confusion matrix

confusionMatrix(factor(elasspred), factor(ytest))

#comparison 
y = c(acclassos,accridges,accelass)
plot(x = c(1,2,3),y=y,ylim = c(0,1),ylab="accuracy", xlab="",xaxt="n",type = "h")
points(x = c(1,2,3),y=y,  pch=19)
text(x = c(1,2,3),y=y, 
     labels = round(y,2),
     pos=3,cex= 0.9)
axis(1, at=1:3, labels=c("lasso","ridge","elastic net"))

