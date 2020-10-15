#### Create figure 3 ####

### Setup
#Set the Working Directory 
setwd("Data") 
getwd()
# data for prediction 
data <- read.csv(file="SambanisImp.csv")

### Load packages
library(randomForest) 
library(caret) 
library(ROCR) 
library(stepPlr) 
library(doMC)
library(xtable) 
library(separationplot)
library(MLmetrics)
library(LiblineaR)

##Use only the 88 variables specified in Sambanis (2006) Appendix### 
data.full<-data[,c("warstds", "ager", "agexp", "anoc", "army85", "autch98", "auto4", 
                   "autonomy", "avgnabo", "centpol3", "coldwar", "decade1", "decade2",
                   "decade3", "decade4", "dem", "dem4", "demch98", "dlang", "drel",
                   "durable", "ef", "ef2", "ehet", "elfo", "elfo2", "etdo4590",
                   "expgdp", "exrec", "fedpol3", "fuelexp", "gdpgrowth", "geo1", "geo2",
                   "geo34", "geo57", "geo69", "geo8", "illiteracy", "incumb", "infant",
                   "inst", "inst3", "life", "lmtnest", "ln_gdpen", "lpopns", "major", "manuexp", "milper", "mirps0", "mirps1", "mirps2", "mirps3", "nat_war", "ncontig",
                   "nmgdp", "nmdp4_alt", "numlang", "nwstate", "oil", "p4mchg",
                   "parcomp", "parreg", "part", "partfree", "plural", "plurrel",
                   "pol4", "pol4m", "pol4sq", "polch98", "polcomp", "popdense",
                   "presi", "pri", "proxregc", "ptime", "reg", "regd4_alt", "relfrac", "seceduc", "second", "semipol3", "sip2", "sxpnew", "sxpsq", "tnatwar", "trade",
                   "warhist", "xconst")]

###Convert DV into Factor with names for Caret Library### 
data.full$warstds<-factor(data.full$warstds,levels=c(0,1),labels=c("peace", "war"))

# distribute workload over multiple cores for faster computation & set seed
registerDoMC(cores=7)
set.seed(666)

# Set trainControl

tc <- trainControl(method="cv", summaryFunction=twoClassSummary, classProb=T,
                   savePredictions = T)

### Modeling

modeling <- function(x) {
  # Create folds
  data.full <- mutate(x, fold = createFolds(data.full$warstds, 5, list = F))
  
  # Allocate folds to test & training (folds fixed, allocation randomized)
  sample1 <- sample(5, 1)
  train1 <- filter(data.full, fold %in% sample1)
  test1 <- filter(data.full, !(fold %in% sample1))
  
  sample2 <- sample(5, 2)
  train2 <- filter(data.full, fold %in% sample2)
  test2 <- filter(data.full, !(fold %in% sample2))
  
  sample3 <- sample(5, 3)
  train3 <- filter(data.full, fold %in% sample3)
  test3 <- filter(data.full, !(fold %in% sample3))
  
  sample4 <- sample(5, 4)
  train4 <- filter(data.full, fold %in% sample4)
  test4 <- filter(data.full, !(fold %in% sample4))
  
  ## Random forests
  # Train models
  rf.fit.2 <- train(warstds~., metric="ROC", method="rf", trControl=tc, data=train1) 
  rf.fit.4 <- train(warstds~., metric="ROC", method="rf", trControl=tc, data=train2) 
  rf.fit.6 <- train(warstds~., metric="ROC", method="rf", trControl=tc, data=train3) 
  rf.fit.8 <- train(warstds~., metric="ROC", method="rf", trControl=tc, data=train4) 
  
  # F1-score calculation
  rf.2_f1 <- F1_Score(test1$warstds, predict(rf.fit.2, test1), positive = "war")
  rf.4_f1 <- F1_Score(test2$warstds, predict(rf.fit.4, test2), positive = "war")
  rf.6_f1 <- F1_Score(test3$warstds, predict(rf.fit.6, test3), positive = "war")
  rf.8_f1 <- F1_Score(test4$warstds, predict(rf.fit.8, test4), positive = "war")
  
  ## Logistic regression
  # Train models
  lr.fit.2 <- train(warstds~., metric="ROC", method="glm", family="binomial", 
                    trControl=tc, data=train1)
  lr.fit.4 <- train(warstds~., metric="ROC", method="glm", family="binomial", 
                    trControl=tc, data=train2)
  lr.fit.6 <- train(warstds~., metric="ROC", method="glm", family="binomial", 
                    trControl=tc, data=train3)
  lr.fit.8 <- train(warstds~., metric="ROC", method="glm", family="binomial", 
                    trControl=tc, data=train4)
  
  # F1-score calculation
  lr.2_f1 <- F1_Score(test1$warstds, predict(lr.fit.2, test1), positive = "war")
  lr.4_f1 <- F1_Score(test2$warstds, predict(lr.fit.4, test2), positive = "war")
  lr.6_f1 <- F1_Score(test3$warstds, predict(lr.fit.6, test3), positive = "war")
  lr.8_f1 <- F1_Score(test4$warstds, predict(lr.fit.8, test4), positive = "war")
  
  ## L1- regularized logistic regression
  # Train models
  l1.fit.2 <- train(warstds~., metric="ROC", method="regLogistic", trControl=tc, data=train1)
  l1.fit.4 <- train(warstds~., metric="ROC", method="regLogistic", trControl=tc, data=train2)
  l1.fit.6 <- train(warstds~., metric="ROC", method="regLogistic", trControl=tc, data=train3)
  l1.fit.8 <- train(warstds~., metric="ROC", method="regLogistic", trControl=tc, data=train4)
  
  # F1-score calculation
  l1.2_f1 <- F1_Score(test1$warstds, predict(l1.fit.2, test1), positive = "war")
  l1.4_f1 <- F1_Score(test2$warstds, predict(l1.fit.4, test2), positive = "war")
  l1.6_f1 <- F1_Score(test3$warstds, predict(l1.fit.6, test3), positive = "war")
  l1.8_f1 <- F1_Score(test4$warstds, predict(l1.fit.8, test4), positive = "war")
  
  ## Combine results
  results <- data.frame(Method = rep(c("RF", "LR", "L1-regularized LR"), each = 4), 
                        Training_set = rep(c(.2, .4, .6, .8), 3), 
                        F1 = c(rf.2_f1, rf.4_f1, rf.6_f1, rf.8_f1, 
                               lr.2_f1, lr.4_f1, lr.6_f1, lr.8_f1, 
                               l1.2_f1, l1.4_f1, l1.6_f1, l1.8_f1))
  return(results)
}

### Run models (10 times)
Output <- replicate(10, modeling(data.full))

## Clean model output
Output.data <- as.data.frame(apply(Output, 1, unlist))

Output.grouped <- group_by(Output.data, Method, Training_set) %>% 
  summarise(mean_F1 = mean(F1, na.rm = T), 
            sd_F1 = sd(F1, na.rm = T)) %>% 
  as.data.frame() %>% 
  mutate(Method = as.factor(Method))
levels(Output.grouped$Method)[levels(Output.grouped$Method)=="1"] <- "L1-regularized LR"
levels(Output.grouped$Method)[levels(Output.grouped$Method)=="2"] <- "LR"
levels(Output.grouped$Method)[levels(Output.grouped$Method)=="3"] <- "RF"

## Generate graph
ggplot(Output.grouped, aes(x=Training_set, y=mean_F1)) + 
  facet_grid(.~Method) +
  geom_errorbar(aes(ymin=mean_F1-sd_F1, ymax=mean_F1+sd_F1), width=.1) + 
  geom_line(aes(linetype=Method)) + 
  geom_point(aes(shape=Method)) + 
  labs(y="F1-score", x="Ratio of the Training Set") + 
  theme(axis.title=element_text(size=14), 
        strip.text.x = element_text(size = 16), 
        legend.position = "none")
