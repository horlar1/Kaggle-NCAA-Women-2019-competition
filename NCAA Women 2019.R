# %% [code]
options(warn = -1)
# load libraries
suppressMessages(require(plyr))
suppressMessages(require(data.table))
suppressMessages(require(tidyverse))
suppressMessages(require(xgboost))
suppressMessages(require(MLmetrics))

### load the data
samp_sub <- fread("../input/WSampleSubmissionStage2.csv")
TourneySeeds <- fread("../input/stage2wdatafiles/WNCAATourneySeeds.csv")
TourneyResults <- fread("../input/stage2wdatafiles/WNCAATourneyDetailedResults.csv")
SeasonResults <- fread("../input/stage2wdatafiles/WRegularSeasonDetailedResults.csv")


# There is nothing special about my solution,i kept the modelling part from @raddar solution, while generating advanced  statistics as well as adding the power ranking. Here is a list of advanced stats i have in my model.
# 
# * TSA - True Shooting Attempts
# * TSP - True Shooting Percentage
# * TOP - Turnover Percentage
# * PPP - Point Per Possession
# * DRB - Defensive Rebound Perentage
# * ORB - Offensive Rebound Percentage
# * OER - Opponent Effciency 

# %% [code]
map.func = function(x){
  map = x %>% 
    sapply(FUN = function(x){strsplit(x, '')[[1]][1]}) %>% as.factor()
  return(map)
}
# Data Preparation
###
TourneySeeds <- TourneySeeds %>% 
  mutate(conference = map.func(Seed),
         SeedNum = gsub("[A-Z+a-z]", "", Seed)) %>% 
  select(Season, TeamID, SeedNum, conference)
TourneySeeds$SeedNum = as.numeric(TourneySeeds$SeedNum)

T1 = TourneySeeds
colnames(T1) = c("Season","T1_ID","SeedNum1","conference1")
T2 = TourneySeeds
colnames(T2) = c("Season","T2_ID","SeedNum2","conference2")

### Collect regular season results - double the data by swapping team positions

r1 = SeasonResults[, c("Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "NumOT","WFGM","WFGA", "WAst", "WBlk","WTO","WDR","WOR","WFGM3","WFGA3","WFTM","WFTA","WStl","WPF","LFGM","LFGA", "LAst", "LBlk","LTO","LDR","LOR","LFGM3","LFGA3","LFTM","LFTA","LStl","LPF")]
r2 = SeasonResults[, c("Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT","LFGM","LFGA", "LAst", "LBlk","LTO","LDR","LOR","LFGM3","LFGA3","LFTM","LFTA","LStl","LPF","WFGM","WFGA", "WAst", "WBlk","WTO","WDR","WOR","WFGM3","WFGA3","WFTM","WFTA","WStl","WPF")]
names(r1) = c("Season", "DayNum", "T1", "T1_Points", "T2", "T2_Points", "NumOT","T1_fgm","T1_fga", "T1_ast", "T1_blk","T1_to","T1_dr","T1_or","T1_fgm3","T1_fga3","T1_ftm","T1_fta","T1_stl","T1_pf","T2_fgm", "T2_fga", "T2_ast", "T2_blk","T2_to","T2_dr","T2_or","T2_fgm3","T2_fga3","T2_ftm","T2_fta","T2_stl","T2_pf")
names(r2) =  c("Season", "DayNum", "T1", "T1_Points", "T2", "T2_Points", "NumOT","T1_fgm","T1_fga", "T1_ast", "T1_blk","T1_to","T1_dr","T1_or","T1_fgm3","T1_fga3","T1_ftm","T1_fta","T1_stl","T1_pf","T2_fgm","T2_fga", "T2_ast", "T2_blk","T2_to","T2_dr","T2_or","T2_fgm3","T2_fga3","T2_ftm","T2_fta","T2_stl","T2_pf")
regular_season = rbind(r1, r2)
rm(r1,r2)

### Collect tourney results - double the data by swapping team positions

t1 = TourneyResults[, c("Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore")] %>% mutate(ResultDiff = WScore - LScore)
t2 = TourneyResults[, c("Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore")] %>% mutate(ResultDiff = LScore - WScore)
names(t1) = c("Season", "DayNum", "T1", "T2", "T1_Points", "T2_Points", "ResultDiff")
names(t2) = c("Season", "DayNum", "T1", "T2", "T1_Points", "T2_Points", "ResultDiff")
tourney = rbind(t1, t2)
rm(t1,t2)

## TEAM QUALITY FROM RADDAR SOLUTION
march_teams = select(TourneySeeds, Season, Team = TeamID)
X =  regular_season %>% 
  inner_join(march_teams, by = c("Season" = "Season", "T1" = "Team")) %>% 
  inner_join(march_teams, by = c("Season" = "Season", "T2" = "Team")) %>% 
  select(Season, T1, T2, T1_Points, T2_Points, NumOT) %>% distinct()
X$T1 = as.factor(X$T1)
X$T2 = as.factor(X$T2)
library(lme4)
quality = list()
for (season in unique(X$Season)) {
  glmm = glmer(I(T1_Points > T2_Points) ~  (1 | T1) + (1 | T2), data = X[X$Season == season & X$NumOT == 0, ], family = binomial) 
  random_effects = ranef(glmm)$T1
  quality[[season]] = data.frame(Season = season, Team_Id = as.numeric(row.names(random_effects)), quality = exp(random_effects[,"(Intercept)"]))
}
quality = do.call(rbind, quality)


##### FEATURE ENGINEEING
season_summary = 
  regular_season %>%
  mutate(win14days = ifelse(DayNum > 118 & T1_Points > T2_Points, 1, 0),
         last14days = ifelse(DayNum > 118, 1, 0),
         T1_TSA = T1_fga + 0.475*T1_fta,
         T2_TSA = T2_fga + 0.475*T2_fta,
         T1_TSP = T1_Points / (2*T1_TSA),
         T2_TSP = T2_Points / (2*T2_TSA),
         T1_TOP = T1_to / (T1_fga + 0.475*T1_fta + T1_to),
         T2_TOP = T2_to / (T2_fga + 0.475*T2_fta + T2_to),
         T1_FTF = T1_ftm / T1_fga,
         T2_FTF = T2_ftm / T2_fga,
         T1_POS = (T1_fga - T1_or) + T1_to + 0.475*T1_fta,
         T2_POS = (T2_fga - T2_or) + T2_to + 0.475*T2_fta,
         T1_PPP = T1_Points / T1_POS,
         T2_PPP = T2_Points / T2_POS,
         T2_DRB = T2_dr / (T1_or+ T2_dr),
         T2_ORB = T2_or / (T2_or + T1_dr)) %>% 
  group_by(Season, T1) %>%
  summarize(
    WinRatio14d = sum(win14days) / sum(last14days),
    PointsDiffMean = mean(T1_Points - T2_Points),
    FgaMedian = median(T1_fga),
    FgaMin = min(T1_fga), 
    FgaMax = max(T1_fga), 
    BlkMean = mean(T1_blk),  
    OppFgaMin = min(T2_fga),
    WStlMean = mean(T1_stl),
    WPFMean = mean(T1_pf),
    OppPFMEan = mean(T2_pf),
    MeanASt_TO = mean(T1_ast) / mean(T1_to),
    MeanASt_TO2 = mean(T2_ast) / mean(T2_to),
    OER = mean(T2_Points) / mean(T2_fga),
    PPP = max(T2_PPP),
    TOP = max(T2_TOP),
    TSP = mean(T2_TSP),
    OppDRB = max(T2_DRB),
    OppORB = max(T2_ORB),
    OppdrMean = mean(T2_dr)
  ) 

season_summary_X1 = season_summary 
season_summary_X2 = season_summary 
names(season_summary_X1) = c("Season", "T1", paste0("X1_",names(season_summary_X1)[-c(1,2)]))
names(season_summary_X2) = c("Season", "T2", paste0("X2_",names(season_summary_X2)[-c(1,2)]))




### Combine all features into a data frame
data_matrix =
  tourney %>% 
  left_join(season_summary_X1, by = c("Season", "T1")) %>% 
  left_join(season_summary_X2, by = c("Season", "T2")) %>%
  left_join(select(T1, Season, T1 = T1_ID, Seed1 = SeedNum1, conference1), by = c("Season", "T1")) %>% 
  left_join(select(T2, Season, T2 = T2_ID, Seed2 = SeedNum2,conference2), by = c("Season", "T2")) %>% 
  mutate(SeedDiff = Seed1 - Seed2) %>%
  left_join(select(quality, Season, T1 = Team_Id, quality_march_T1 = quality), by = c("Season", "T1")) %>%
  left_join(select(quality, Season, T2 = Team_Id, quality_march_T2 = quality), by = c("Season", "T2")) 

### add power ranking
## This makes my model to be too confident, burns me alot in losses but good in wins,
## i have only have one 50ish% prediction in the tourney, which was the final game.
data_matrix = data_matrix %>% 
  group_by(Season, conference1) %>% 
  mutate(pwr_rank1 = frank(quality_march_T1,ties.method = "dense")) %>% 
  ungroup() %>% 
  group_by(Season,conference2) %>% 
  mutate(pwr_rank2 = frank(quality_march_T2,ties.method = "dense")) %>% 
  ungroup() 




######################################
### GENERATE TEST SET FEATURE 
######################################
testset <- samp_sub %>% separate(ID, into=c("Season","T1","T2"), sep="_", extra="merge", fill="right") %>% within(rm('Pred')) %>% as.data.frame()

testset$Season <- as.numeric(testset$Season)
testset$T1 <- as.numeric(testset$T1)
testset$T2 <- as.numeric(testset$T2)

testset = testset %>% 
  left_join(season_summary_X1, by = c("Season", "T1")) %>% 
  left_join(season_summary_X2, by = c("Season", "T2")) %>%
  left_join(select(T1, Season, T1 = T1_ID, Seed1 = SeedNum1, conference1), by = c("Season", "T1")) %>% 
  left_join(select(T2, Season, T2 = T2_ID, Seed2 = SeedNum2,conference2), by = c("Season", "T2")) %>% 
  mutate(SeedDiff = Seed1 - Seed2) %>%
  left_join(select(quality, Season, T1 = Team_Id, quality_march_T1 = quality), by = c("Season", "T1")) %>%
  left_join(select(quality, Season, T2 = Team_Id, quality_march_T2 = quality), by = c("Season", "T2")) 

#### add power ranking
testset = testset %>% 
  group_by(Season, conference1) %>% 
  mutate(pwr_rank1 = frank(quality_march_T1,ties.method = "dense")) %>% 
  ungroup() %>% 
  group_by(Season,conference2) %>% 
  mutate(pwr_rank2 = frank(quality_march_T2,ties.method = "dense")) %>% 
  ungroup()



## MODEL BUILDING
#################
features = setdiff(names(data_matrix), c("Season", "DayNum", "T1", "T2", "T1_Points", "T2_Points", "ResultDiff","conference1","conference2"))
dtrain = xgb.DMatrix(as.matrix(data_matrix[, features]), label = data_matrix$ResultDiff)

cauchyobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  c <- 5000 
  x <-  preds-labels
  grad <- x / (x^2/c^2+1)
  hess <- -c^2*(x^2-c^2)/(x^2+c^2)^2
  return(list(grad = grad, hess = hess))
}

xgb_parameters = 
  list(objective = cauchyobj, 
       eval_metric = "mae",
       booster = "gbtree", 
       eta = 0.02,
       subsample = 0.35,
       colsample_bytree = 0.7,
       num_parallel_tree = 10,
       min_child_weight = 40,
       gamma = 10,
       max_depth = 3)

N = nrow(data_matrix)
fold5list = c(
  rep( 1, floor(N/5) ),
  rep( 2, floor(N/5) ),
  rep( 3, floor(N/5) ),
  rep( 4, floor(N/5) ),
  rep( 5, N - 4*floor(N/5) )
)
### Build cross-validation model, repeated 10-times

iteration_count = c()
smooth_model = list()
CV = c()
cvpreds = matrix(0,nrow(data_matrix),ncol = 1)
for (i in 1:1) {
  
  ### Resample fold split
  set.seed(i)
  folds = list()  
  fold_list = sample(fold5list)
  for (k in 1:5) folds[[k]] = which(fold_list == k)
  
  set.seed(120)
  xgb_cv = 
    xgb.cv(
      params = xgb_parameters,
      data = dtrain,
      nrounds = 3000,
      verbose = 0,
      print_every_n = 50,
      folds = folds,
      early_stopping_rounds = 25,
      maximize = FALSE,
      prediction = TRUE
    )
  iteration_count = c(iteration_count, xgb_cv$best_iteration)
  
  ### Fit a smoothed GAM model on predicted result point differential to get probabilities
  smooth_model[[i]] = smooth.spline(x = xgb_cv$pred, y = ifelse(data_matrix$ResultDiff > 0, 1, 0))
  
  #### cv prediction and compute cv score
  target = ifelse(data_matrix$ResultDiff > 0,1,0)
  cv_pred = predict(smooth_model[[i]], xgb_cv$pred)$y
  cvpreds[,i] = cv_pred
  #### compute cv score 
  cv_score = MLmetrics::LogLoss(cv_pred, target)
  CV <- c(CV, cv_score)
  cat("Model CV Logloss score:",cv_score,"\n")
}
mean(CV)




### Build submission models

submission_model = list()
for (i in 1:1) {
  set.seed(i)
  submission_model[[i]] = 
    xgb.train(
      params = xgb_parameters,
      data = dtrain,
      nrounds = round(iteration_count[i]*1.05),
      verbose = 1,
      #nthread = 12,
      maximize = FALSE,
      prediction = TRUE
    )
}



######## MAKING PREDICTION
testset = testset[,features]
dtest = xgb.DMatrix(as.matrix(testset))

probs = list()
for (i in 1:1) {
  preds = predict(submission_model[[i]], dtest)
  probs[[i]] = predict(smooth_model[[i]], preds)$y
}
testset$Pred = Reduce("+", probs) / 1

### Clipping Prediction
testset$Pred[testset$Pred <= 0.025] = 0.025
testset$Pred[testset$Pred >= 0.975] = 0.975

########
testset$Pred[testset$Seed1 == 16 & testset$Seed2 == 1] = 0
testset$Pred[testset$Seed1 == 15 & testset$Seed2 == 2] = 0
testset$Pred[testset$Seed1 == 14 & testset$Seed2 == 3] = 0
testset$Pred[testset$Seed1 == 13 & testset$Seed2 == 4] = 0
testset$Pred[testset$Seed1 == 12 & testset$Seed2 == 5] = 0
testset$Pred[testset$Seed1 == 1 & testset$Seed2 == 16] = 1
testset$Pred[testset$Seed1 == 2 & testset$Seed2 == 15] = 1
testset$Pred[testset$Seed1 == 3 & testset$Seed2 == 14] = 1
testset$Pred[testset$Seed1 == 4 & testset$Seed2 == 13] = 1
testset$Pred[testset$Seed1 == 5 & testset$Seed2 == 12] = 1


### SUBMISSION
samp_sub$Pred = testset$Pred
write.csv(samp_sub, file = "submission.csv", row.names = F)
