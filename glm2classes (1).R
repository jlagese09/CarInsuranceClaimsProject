#----------------------------------------------------------------------------------------------------------------------------
#Previous Data Gatherings/Functons/Etc

#Adding Data
dat <- read.csv("C:\\Users\\jlage\\Desktop\\GLMdata.csv", header=TRUE)
#Installing relative libraries
library(rpart)
library(repr) # Scale plot size:
library(rpart.plot)
library(tidyverse) # work quietly:
suppressMessages(library(tidyverse))
suppressMessages(library(lme4))
library(tensorflow)
library(keras)
#use_condaenv("r-tensorflow")
library(xgboost)
suppressMessages(library(xgboost))

#install_keras()

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Data Preparations
#There are two lines that I had to change because they weren't previously working. They are labelled below
dat$VehGas <- factor(dat$VehGas)
dat$ClaimNb <- pmin(dat$ClaimNb,1)
dat$ClaimNb1 <- pmin(dat$ClaimNb, 15)
dat$Exposure <- pmin(dat$Exposure,1)
dat$VehBrand2 <- as.factor(ifelse(dat$VehBrand=='B12' & dat$VehAge==0 & 
dat$VehGas =='Regular', 'B12RN', as.character(dat$VehBrand)))
 #Adding the new column that is the strange combination of brand b12 and age = 0 (new)
dat2 <- dat

#Also changed the next two lines to make sure the 'AreaGLM variable is a number
dat2$AreaGLM <- as.factor(dat2$Area)
dat2[,"AreaGLM"] <-as.integer(relevel(dat2[,"AreaGLM"], ref="D"))

dat2$VehPowerGLM <- as.factor(pmin(dat2$VehPower,9))
VehAgeGLM <- cbind(c(0:110), c(1, rep(2, 10), rep(3, 100)))
dat2$VehAgeGLM <- as.factor(VehAgeGLM[dat2$VehAge+1,2])

#Code below I added a bit of my spice to (was not working before data needed releveled)
dat2[,"VehAgeGLM"] <-relevel(dat2[,"VehAgeGLM"], ref="2")

DrivAgeGLM <- cbind(c(18:100), c(rep(1,21-18), rep(2,26-21), rep(3,31-26), rep(4,41-31), rep(5,51-41), rep(6,71-51), rep(7,101-71)))
dat2$DrivAgeGLM <- as.factor(DrivAgeGLM[dat2$DrivAge-17,2])
dat2[,"DrivAgeGLM"] <-relevel(dat2[,"DrivAgeGLM"], ref="5")
dat2$BonusMalusGLM <- as.integer(pmin(dat2$BonusMalus, 150))
dat2$DensityGLM <- as.numeric(log(dat2$Density))

#Below is my modification
dat2[,"Region"] <- relevel(factor(dat2[,"Region"]), ref="R24")



#----------------------------------------------------------------------------------------------------------------------------------
# Evaluation Metrics and Helper functions
# Function PD: Calculate Poisson Deviance
PD <- function(pred, obs) {200*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)}
PD
# Function PD2: Print Poisson Deviance learn/test
PD2 <- function(txt, l.c, l.x, t.c, t.x) {
    sprintf("%s, Learn/Test: %.2f%% / %.2f%%", txt, PD(l.c, l.x), PD(t.c, t.x))
}

# Function CF2: Print claim frequency
CF2 <- function(txt, l.c, l.x, t.c, t.x) {
    sprintf("%s: %.2f%% / %.2f%%", txt, sum(l.c)/sum(l.x)*100, sum(t.c)/sum(t.x)*100)
}

# Function Benchmark.GLM2: Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2 <- function(txt, pred, obs) {
  index <- ((PD(pred, obs) - PD(test$fit.cf, test$ClaimNb)) / (PD(test$fitGLM2, test$ClaimNb) - PD(test$fit.cf, test$ClaimNb))) * 100
  sprintf("GLM2-Improvement-Index (PD test) of %s: %.1f%%", txt, index)
}

# Calculate next year's bonus-malus in range 54-125 (preparation for chapter 8)
# Define Bonus-function (guessed from BonusMalus-bands, see frequency table and ch.8)
Bonus <- function(x) {
  a <- ifelse(x>125,0,ifelse(x==125,118,ifelse(x>100,x-6,ifelse(x>80,x-5,ifelse(x>60,x-4,ifelse(x>=54,x-3,0))))))
  return(a)
}

# BonusMalus 1 year later:
dat.1y <- dat2
dat.1y$BonusMalusGLM <- Bonus(dat2$BonusMalusGLM)



#------------------------------------------------------------------------------------------------------------------
# for later use (Ch.5) we create five 20%-subsamples ("folds") and take the last fold as the holdout data set
k <- 5
set.seed(42)
fold <- sample(1:k, nrow(dat2), replace = TRUE)
dat2$fold <- fold
learn <- dat2[dat2$fold != 5,]    # 80%
test <- dat2[dat2$fold == 5,]    # 20%
CF2("Claim Frequency (Actual) Learn/Test", learn$ClaimNb, learn$Exposure, test$ClaimNb, test$Exposure)
names(test)
table(test$ClaimNb)
# Model INT "predictions"
cf <- sum(learn$ClaimNb)/sum(learn$Exposure) # claim frequency
learn$fit.cf <- cf*learn$Exposure
test$fit.cf <- cf*test$Exposure
# End of pre data processing, function adding, etc everything's set to go for chapter two and wont print everything from the previous chapter.

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#4.1: 'GLM4' featuring neural net residual analysis.
#Using interactions found using neural network residual analysis

d.glm4 <- glm(ClaimNb ~ VehPowerGLM * VehAgeGLM + VehAgeGLM * VehBrand
                + VehGas * VehAgeGLM + DensityGLM
                + BonusMalusGLM 
                + DrivAge,
                data=learn, offset=log(Exposure), family=poisson())

summary(d.glm4)
learn$fitGLM4 <- fitted(d.glm4)
test$fitGLM4 <- predict(d.glm4, newdata=test, type="response")
dat$fitGLM4 <- predict(d.glm4, newdata=dat2, type="response")

#4.2: GLM5 improved GLM with interactions from XGBoost:
d.glm5 <- glm(ClaimNb ~ VehPowerGLM * VehAgeGLM + VehAgeGLM * VehBrand
                + VehGas * VehAgeGLM + DensityGLM
                + BonusMalusGLM 
                + DrivAge,
                data=learn, offset=log(Exposure), family=poisson())

summary(d.glm5)
learn$fitGLM5 <- fitted(d.glm5)
test$fitGLM5 <- predict(d.glm5, newdata=test, type="response")
dat$fitGLM5 <- predict(d.glm5, newdata=dat2, type="response")
# Print Poisson Deviance

summary(test$fitGLM5)


# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
Benchmark.GLM2("GLM5", test$fitGLM5, test$ClaimNb)

#End of Chapter 4

### Adding code to calculate P-values ####
# P-values for GLM4 and GLM5
pchisq(167872,df=542877,lower.tail=FALSE)
pchisq(167549,df=542874,lower.tail=FALSE)


##### Adding Simple Linear Regression ##########
linear<- lm(d.glm4)
summary(linear)


###### next week run sample data that maddy comes up with.
###### 
#------------------------------------------------------------------------------------------------------
#Chapter 5: Cross Validation and Boxplots
# k folds as defined in Ch. 1.6

PD.test.INT  <- vector() # initialize PD-test-folds-vector
PD.test.GLM2 <- vector() #
PD.test.GLM2S <- vector() #
PD.test.GLM4 <- vector()
PD.test.GLM5 <- vector()

for (i in 1:k) {
    learn <- dat2[dat2$fold != i,]
    test <- dat2[dat2$fold == i,]
    (n_l <- nrow(learn))
    (n_t <- nrow(test))
    sum(learn$ClaimNb)/ sum(learn$Exposure)

    ### Model INT (intercept-only).  No model, just average claim frequency
    (cf <- sum(learn$ClaimNb)/sum(learn$Exposure))
    test$fit.cf <- cf*test$Exposure

    # out-of-sample losses (in 10^(-2))
    PD.test.INT[i] <- PD(test$fit.cf, test$ClaimNb)




### Model GLM1: Basic GLM without interactions
    d.glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + BonusMalusGLM
              + VehBrand + VehGas + DensityGLM + Region + AreaGLM
              + DrivAge,
              data=learn, offset=log(Exposure), family=binomial())
summary(d.glm1)
d.glm1$R
learn$fitGLM1 <- fitted(d.glm1)
test$fitGLM1 <- predict(d.glm1, newdata=test, type="response")
dat$fitGLM1 <- predict(d.glm1, newdata=dat2, type="response")

summary(dat$fitGLM1)
RMSE=function(x,y)
{
  rmse=sqrt(mean(x-y)^2)
  return(rmse)
}
RMSE(test$fitGLM1, test$ClaimNb)

summary(test$ClaimNb)
summary(test$fitGLM1)
n=length(test$fitGLM1)
n
Y=rep(0,n)
for (i in 1:n)
    if (test$fitGLM1[i] > 0.2) Y[i]=1
head(Y)
effi.rate=sum(diag(table(Y,test$fitGLM1)))/n
if((test$fitGLM1<0.2),Y=0, Y=1)

install.packages(caret)
library(caret)
data.frame(
R2 = rsquare(test$fitGLM1, test$ClaimNb)
)
plot(test$fitGLM1, test$ClaimNb)

predict = round(test$fitGLM1)
table(predict, test$ClaimNb)

table(Y, test$ClaimNb)

hist(dat$ClaimNb1)


y <- 0
if(test$fitGLM1<0.2){0
else1}






### Model GLM2: Basic GLM without interactions
    d.glm2 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + BonusMalusGLM
              + VehGas + DensityGLM + AreaGLM
              + DrivAge,
              data=learn, offset=log(Exposure), family=binomial())
summary(d.glm2)

learn$fitGLM2 <- fitted(d.glm2)
test$fitGLM2 <- predict(d.glm2, newdata=test, type="response")
dat$fitGLM2 <- predict(d.glm2, newdata=dat2, type="response")



RMSE=function(x,y)
{
  rmse=sqrt(mean(x-y)^2)
  return(rmse)
}
RMSE(test$fitGLM2, test$ClaimNb)

plot(test$fitGLM2, test$ClaimNb)

predict = round(test$fitGLM2)
table(predict, test$ClaimNb)




### Model GLM3: Basic GLM without interactions
    d.glm3 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + BonusMalusGLM
              + VehGas + DensityGLM
              + DrivAge,
              data=learn, offset=log(Exposure), family=binomial())
summary(d.glm3)

learn$fitGLM3 <- fitted(d.glm3)
test$fitGLM3 <- predict(d.glm3, newdata=test, type="response")
dat$fitGLM3 <- predict(d.glm3, newdata=dat2, type="response")


RMSE=function(x,y)
{
  rmse=sqrt(mean(x-y)^2)
  return(rmse)
}
RMSE(test$fitGLM3, test$ClaimNb)

plot(test$fitGLM3, test$ClaimNb)

predict = round(test$fitGLM3)
table(predict, test$ClaimNb)

library(
ggplot





     ### Model GLM2: Basic GLM without interactions
    d.glm2 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + BonusMalusGLM
              + VehBrand + VehGas + DensityGLM + Region + AreaGLM
              + DrivAge + log(DrivAge) +  I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4),
              data=learn, offset=log(Exposure), family=poisson())
    test$fitGLM2 <- predict(d.glm2, newdata=test, type="response")
    PD.test.GLM2[i] <- PD(test$fitGLM2, test$ClaimNb)


    ### Model GLM2S: GLM2 with dummy car brand "B12RN"
    d.glm2s <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + BonusMalusGLM
              + VehBrand2 + VehGas + DensityGLM + Region + AreaGLM
              + DrivAge + log(DrivAge) +  I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4),
              data=learn, offset=log(Exposure), family=poisson())
    test$fitGLM2S <- predict(d.glm2s, newdata=test, type="response")
    PD.test.GLM2S[i] <- PD(test$fitGLM2S, test$ClaimNb)


    ### Model GLM4: Featuring neural net residual analyses
    d.glm4 <- glm(ClaimNb ~ AreaGLM + VehPowerGLM * VehAgeGLM + VehAgeGLM * VehBrand
              + VehGas * VehAgeGLM + DensityGLM + Region
              + BonusMalusGLM * log(DrivAge) + log(BonusMalusGLM) + I(BonusMalusGLM^2)
              + I(BonusMalusGLM^3) + I(BonusMalusGLM^4)
              + DrivAge + I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4),
              data=learn, offset=log(Exposure), family=poisson())
    test$fitGLM4 <- predict(d.glm4, newdata=test, type="response")
    PD.test.GLM4[i] <- PD(test$fitGLM4, test$ClaimNb)


    ### Model GLM5: Improved GLM with Interactions from XGBoost
    d.glm5 <- glm(ClaimNb ~ AreaGLM + VehPowerGLM * VehAgeGLM  + VehAgeGLM * VehBrand
              + VehGas * VehAgeGLM + DensityGLM + Region
              + BonusMalusGLM * log(DrivAge) +  VehAgeGLM *log(BonusMalusGLM) + I(BonusMalusGLM^2)
              + I(BonusMalusGLM^3) + I(BonusMalusGLM^4)
              + DrivAge * DensityGLM + I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4),
              data=learn, offset=log(Exposure), family=poisson())
    test$fitGLM5 <- predict(d.glm5, newdata=test, type="response")
    PD.test.GLM5[i] <- PD(test$fitGLM5, test$ClaimNb) }


# write vectors in a Data Frame
df <- data.frame(1:5,PD.test.INT,PD.test.GLM2,PD.test.GLM2S,PD.test.GLM4,PD.test.GLM5)
names(df) <- c("fold","INT","GLM2","GLM2S","GLM4","GLM5")
# calculate GLM2-Improvement-Index
df.idx <- df
df.idx$GLM2S<- round(((df$GLM2S- df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$GLM4 <- round(((df$GLM4 - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx$GLM5 <- round(((df$GLM5 - df$INT) / (df$GLM2 - df$INT)) * 100,2)
df.idx

#Plotting deviances
ggplot(df, aes(fold)) +
  geom_point(aes(y = INT,  colour = "INT"),  size=6) +
  geom_point(aes(y = GLM2, colour = "GLM2"), size=6) +
  geom_point(aes(y = GLM2S,colour = "GLM2S"),size=6) +
  geom_point(aes(y = GLM4, colour = "GLM4"), size=6) +
  geom_point(aes(y = GLM5, colour = "GLM5"), size=6) +
  ylab("Poisson Deviance") + theme(text = element_text(size=15))

#Transform data from wide to long
df.long <- gather(df[1:6],model,PD,INT,GLM2,GLM2S,GLM4,GLM5)
tail(df.long)

# create boxplot
ggplot(df.long, aes(model, PD)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0) +
  stat_summary(fun = mean, colour="darkred", geom="point",shape=18, size=7) +
  coord_flip() +
  xlab("Model") +
  ylab("Poisson Deviance in 10^(-2)") +
  theme(text = element_text(size=20))

#Plotting GLM2 improvements
df.idx  %>% select(fold,GLM2S,GLM4,GLM5) %>%
  gather(model,PD,GLM2S,GLM4,GLM5) %>%
  ggplot(aes(model, PD)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0) +
    stat_summary(fun = mean, colour="darkred", geom="point",shape=18, size=7) +
    coord_flip() +
    xlab("Model") +
    ylab("GLM2-Improvement-Index") +
    theme(text = element_text(size=20))
'


'



#The massive storm of preprocessing will be included in this box
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#Chapter 6: Deep Learning Approach with Embeddings and the "CANN"
#6.1: Preprocessing
# feature pre-processing

# min-max-scaler:
PreProcess.Continuous <- function(var1, dat2){
  names(dat2)[names(dat2) == var1]  <- "V1"
  dat2$X <- as.numeric(dat2$V1)
  dat2$X <- 2*(dat2$X-min(dat2$X))/(max(dat2$X)-min(dat2$X))-1
  names(dat2)[names(dat2) == "V1"]  <- var1
  names(dat2)[names(dat2) == "X"]  <- paste(var1,"X", sep="")
  dat2
}

# pre-procecessing function:
Features.PreProcess <- function(dat2){
  dat2 <- PreProcess.Continuous("AreaGLM", dat2)
  dat2 <- PreProcess.Continuous("VehPower", dat2)
  dat2$VehAge <- pmin(dat2$VehAge,20)
  dat2 <- PreProcess.Continuous("VehAge", dat2)
  dat2$DrivAge <- pmin(dat2$DrivAge,90)
  dat2 <- PreProcess.Continuous("DrivAge", dat2)
  dat2$BonusMalus <- pmin(dat2$BonusMalus,150)
  dat2 <- PreProcess.Continuous("BonusMalus", dat2)

  #Next two lines contain my own modifications to solve runtime errors
  dat2$VehBrand <- as.factor(dat2$VehBrand)
  dat2[,"VehBrandX"] <- relevel(factor(dat2[,"VehBrand"]), ref="B12")

  dat2$VehBrandX <- as.integer(dat2$VehBrand)-1
  dat2$VehGasX <- as.integer(dat2$VehGas)-1.5
  dat2$Density <- round(log(dat2$Density),2)
  dat2 <- PreProcess.Continuous("Density", dat2)
  dat2$RegionX <- as.integer(dat2$Region)-1  # char R11,,R94 to number 0,,21
  dat2
}

#Keep original variables and fit GLM2 (CANN) model
dat2 <- Features.PreProcess(dat2[,c(1:12,14)])

# choosing learning and test sample (based on folds)
dat2$fold <- fold
learn <- dat2[dat2$fold != 5,]    # 80%
test <- dat2[ dat2$fold == 5,]    # 20%
learn0 <- learn
test0 <- test
learn1 <- learn
test1 <- test
learn2 <- learn
test2 <- test
CF2("Claim Frequency (Actual) Learn/Test", learn$ClaimNb,learn$Exposure, test$ClaimNb,test$Exposure)

# setting up the matrices
features <- c(14:18, 20:21) # definition of feature variables (non-categorical)
q0 <- length(features)
print(features)
sample_n(dat2[4:22], 12)



Xlearn <- as.matrix(learn[, features])  # design matrix learning sample # learning data
Brlearn <- as.matrix(learn$VehBrandX)   #These are the categorical samples
Relearn <- as.matrix(learn$RegionX)
Ylearn <- as.matrix(learn$ClaimNb)

# testing data
Xtest <- as.matrix(test[, features])    # design matrix test sample
Brtest <- as.matrix(test$VehBrandX)
Retest <- as.matrix(test$RegionX)
Ytest <- as.matrix(test$ClaimNb)

#----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------


# choosing the right volumes for EmbNN and CANN
Vlearn <- as.matrix(log(learn$Exposure))
Vtest <- as.matrix(log(test$Exposure))
print(paste("Number of basic features (without VehBrand and Region):",q0))
(lambda.hom <- sum(learn$ClaimNb)/sum(learn$Exposure))


####---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#Creation of Model (First Neural Network)

#6.2: Setting up Common Neural Network Architecture
#THIS IS JUST THE NEURAL NETWORK BY ITSELF
# hyperparameters of the neural network architecture (as specified in "01 CANN approach.r")
q1 <- 20 # Number of neuron in hidden layer 1
q2 <- 15
q3 <- 10
d <- 2   # dimensions embedding layers for categorical features
(BrLabel <- length(unique(learn$VehBrandX)))
(ReLabel <- length(unique(learn$RegionX)))

# define the network architecture
Design   <- layer_input(shape = c(q0),  dtype = "float32", name = "Design")
VehBrand <- layer_input(shape = c(1),   dtype = "int32", name = "VehBrand")
Region   <- layer_input(shape = c(1),   dtype = "int32", name = "Region")
LogVol   <- layer_input(shape = c(1),   dtype = "float32", name = "LogVol")

BrandEmb = VehBrand %>%
  layer_embedding(input_dim = BrLabel, output_dim = d, input_length = 1, name = "BrandEmb") %>%
  layer_flatten(name="Brand_flat")

RegionEmb = Region %>%
  layer_embedding(input_dim = ReLabel, output_dim = d, input_length = 1, name = "RegionEmb") %>%
  layer_flatten(name="Region_flat")

Network = list(Design, BrandEmb, RegionEmb) %>% layer_concatenate(name="concate") %>%
  layer_dense(units=q1, activation="tanh", name="hidden1") %>%
  layer_dense(units=q2, activation="tanh", name="hidden2") %>%
  layer_dense(units=q3, activation="tanh", name="hidden3") %>%
  layer_dense(units=1, activation="linear", name="Network",
              weights=list(array(0, dim=c(q3,1)), array(log(lambda.hom), dim=c(1))))

Response = list(Network, LogVol) %>% layer_add(name="Add") %>%
  layer_dense(units=1, activation=k_exp, name = "Response", trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model <- keras_model(inputs = c(Design, VehBrand, Region, LogVol), outputs = c(Response))
model %>% compile(optimizer = optimizer_nadam(), loss = "poisson")

summary(model)


#---------------------------------------------------------------------------------------------------------
#Fitting of Model

#Set Seed and fit the neural network
set.seed(42) # set seed again
# fitting the neural network (as specified in "01 CANN approach.r")
{t1 <- proc.time()
  fit <- model %>% fit(list(Xlearn, Brlearn, Relearn, Vlearn), Ylearn, epochs=600,
                       batch_size=10000, verbose=0, validation_split=0)
  (proc.time()-t1)}
plot(fit)
#------------------------------------------------------------------------------------------------------
#Calculating predictions after fitting

# calculating the predictions
learn0$fitNNemb <- as.vector(model %>% predict(list(Xlearn, Brlearn, Relearn, Vlearn)))
test0$fitNNemb <- as.vector(model %>% predict(list(Xtest, Brtest, Retest, Vtest)))

# Print claim frequency actual vs predicted
CF2("Claim Frequency NNemb, Test-Sample, Actual/Predicted", test0$ClaimNb,test0$Exposure, test0$fitNNemb,test0$Exposure)

# Print Poisson Deviance
PD2("Poisson Deviance NNemb", learn0$fitNNemb,as.vector(unlist(learn0$ClaimNb)), test0$fitNNemb,as.vector(unlist(test0$ClaimNb)))

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("NNemb", test0$fitNNemb, test0$ClaimNb)

#End of first Neural Network
#-----------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------
#6.4: Combined Actuarial Neural Network with GLM2 (NNGLM)
#This is the CANN (Neural Network) with an incorporated GLM2 model
#First add the GLM2 Model
d.glm2 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + BonusMalusGLM + VehBrand + VehGas + DensityGLM + Region + AreaGLM + DrivAge + log(DrivAge) + I(DrivAge^2) + I(DrivAge^3) + I(DrivAge^4), data=learn, offset=log(Exposure), family=poisson())

#Adding the Learn (Fitting) and Prediction columns for GLM2 model
learn$fitGLM2 <- fitted(d.glm2)
test$fitGLM2 <- predict(d.glm2, newdata=test, type="response")
#---------------------------------------------------------
# Incorporating model GLM2 into a CANN
Vlearn <- as.matrix(log(learn$fitGLM2))
Vtest <- as.matrix(log(test$fitGLM2))
(lambda.hom <- sum(learn$ClaimNb)/sum(learn$fitGLM2))
#----------------------------------------------------------

# repeat model definition
# define the network architecture
Design   <- layer_input(shape = c(q0),  dtype = 'float32', name = 'Design')
VehBrand <- layer_input(shape = c(1),   dtype = 'int32', name = 'VehBrand')
Region   <- layer_input(shape = c(1),   dtype = 'int32', name = 'Region')
LogVol   <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogVol')

BrandEmb = VehBrand %>%
  layer_embedding(input_dim = BrLabel, output_dim = d, input_length = 1, name = 'BrandEmb') %>%
  layer_flatten(name='Brand_flat')

RegionEmb = Region %>%
  layer_embedding(input_dim = ReLabel, output_dim = d, input_length = 1, name = 'RegionEmb') %>%
  layer_flatten(name='Region_flat')

Network = list(Design, BrandEmb, RegionEmb) %>% layer_concatenate(name='concate') %>%
  layer_dense(units=q1, activation='tanh', name='hidden1') %>%
  layer_dense(units=q2, activation='tanh', name='hidden2') %>%
  layer_dense(units=q3, activation='tanh', name='hidden3') %>%
  layer_dense(units=1, activation='linear', name='Network',
              weights=list(array(0, dim=c(q3,1)), array(log(lambda.hom), dim=c(1))))

Response = list(Network, LogVol) %>% layer_add(name='Add') %>%
  layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model <- keras_model(inputs = c(Design, VehBrand, Region, LogVol), outputs = c(Response))
model %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')

summary(model)

#-----------------------------------------------------------------------------------------------------
#Fitting model and calculate predictions
#Re-fit neural network with GLM2-CANN
{t1 <- proc.time()
  fit <- model %>% fit(list(Xlearn, Brlearn, Relearn, Vlearn), Ylearn, epochs=200,
                       batch_size=10000, verbose=0, validation_split=0)
  (proc.time()-t1)}
plot(fit)

#Calculating Predictions
learn1$fitNNGLM <- as.vector(model %>% predict(list(Xlearn, Brlearn, Relearn, Vlearn)))
test1$fitNNGLM <- as.vector(model %>% predict(list(Xtest, Brtest, Retest, Vtest)))

# Print claim frequency actual vs predicted
CF2("Claim Frequency NNGLM, Test-Sample, Actual/Predicted", test1$ClaimNb,test1$Exposure, test1$fitNNGLM,test1$Exposure)

# Print Poisson Deviance
PD2("Poisson Deviance NNGLM", learn1$fitNNGLM,as.vector(unlist(learn1$ClaimNb)), test1$fitNNGLM,as.vector(unlist(test1$ClaimNb)))

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("NNGLM", test1$fitNNGLM, test1$ClaimNb)



#------------------------------------------------------------------------------------------------------------------------------------
#6.5:  The monotonic CANN (NNGLMc)
#To obtain monotonous behavior in BonusMalus, we remove it from feature list of neural network
# changing the matrices (exclude BonusMalus, col18)

#This next block includes:
#1.) Feature selection
#2.) Creation of learning and test data (remember you need one for the non categorical features, and one for the categorical, as well as the data you're trying to predict
#------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
features <- c(14:17, 20:21) # definition of feature variables (non-categorical)
q0 <- length(features)
print(paste("Number of basic features (without VehBrand and Region):",q0))
# learning data
Xlearn <- as.matrix(learn[, features])  # design matrix learning sample
Brlearn <- as.matrix(learn$VehBrandX)
Relearn <- as.matrix(learn$RegionX)
Ylearn <- as.matrix(learn$ClaimNb)
# testing data
Xtest <- as.matrix(test[, features])    # design matrix test sample
Brtest <- as.matrix(test$VehBrandX)
Retest <- as.matrix(test$RegionX)
Ytest <- as.matrix(test$ClaimNb)
# Incorporating model GLM2 into a CANN
Vlearn <- as.matrix(log(learn$fitGLM2))
Vtest <- as.matrix(log(test$fitGLM2))
(lambda.hom <- sum(learn$ClaimNb)/sum(learn$fitGLM2))
#--------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# Creation of 3rd NN
# repeat model definition
# define the network architecture
Design   <- layer_input(shape = c(q0),  dtype = 'float32', name = 'Design')
VehBrand <- layer_input(shape = c(1),   dtype = 'int32', name = 'VehBrand')
Region   <- layer_input(shape = c(1),   dtype = 'int32', name = 'Region')
LogVol   <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogVol')

BrandEmb = VehBrand %>%
  layer_embedding(input_dim = BrLabel, output_dim = d, input_length = 1, name = 'BrandEmb') %>%
  layer_flatten(name='Brand_flat')

RegionEmb = Region %>%
  layer_embedding(input_dim = ReLabel, output_dim = d, input_length = 1, name = 'RegionEmb') %>%
  layer_flatten(name='Region_flat')

Network = list(Design, BrandEmb, RegionEmb) %>% layer_concatenate(name='concate') %>%
  layer_dense(units=q1, activation='tanh', name='hidden1') %>%
  layer_dense(units=q2, activation='tanh', name='hidden2') %>%
  layer_dense(units=q3, activation='tanh', name='hidden3') %>%
  layer_dense(units=1, activation='linear', name='Network',
              weights=list(array(0, dim=c(q3,1)), array(log(lambda.hom), dim=c(1))))

Response = list(Network, LogVol) %>% layer_add(name='Add') %>%
  layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model <- keras_model(inputs = c(Design, VehBrand, Region, LogVol), outputs = c(Response))
model %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')

summary(model)
#-------------------------------------------------------------------------------------------
# re-fitting the neural network with GLM2-CANN
{t1 <- proc.time()
  fit <- model %>% fit(list(Xlearn, Brlearn, Relearn, Vlearn), Ylearn, epochs=200,
                       batch_size=10000, verbose=0, validation_split=0)
  (proc.time()-t1)}
plot(fit)

#-------------------------------------------------------------------------------------------------
# calculating the predictions
learn1$fitNNGLMc <- as.vector(model %>% predict(list(Xlearn, Brlearn, Relearn, Vlearn)))
test1$fitNNGLMc <- as.vector(model %>% predict(list(Xtest, Brtest, Retest, Vtest)))

# Print Poisson Deviance
PD2("Poisson Deviance NNGLMc", learn1$fitNNGLMc,as.vector(unlist(learn1$ClaimNb)), test1$fitNNGLMc,as.vector(unlist(test1$ClaimNb)))

# Improvement in Poisson Deviance on test set compared to GLM2-INT-Improvement
test$fit.cf <- test$Exposure * sum(learn$ClaimNb)/sum(learn$Exposure) # (recalculate INT-Model)
Benchmark.GLM2("NNGLMc", test1$fitNNGLMc, test1$ClaimNb)







PD(test$fitGLM4, test$ClaimNb)
PD(test$fitGLM4, test$ClaimNb)
  PD2("Hello", PD(learn$fitGLM4, learn$ClaimNb), PD(test$fitGLM4, test$ClaimNb))
PD2("Hello", learn$fitGLM4, learn$ClaimNb, test$fitGLM4, test$ClaimNB)
RMSE=function(x,y)
{
  rmse=sqrt(mean(x-y)^2)
  return(rmse)
}
RMSE(test$fitGLM4, test$ClaimNb)

