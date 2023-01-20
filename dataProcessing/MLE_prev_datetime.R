#sudo xcode-select -switch /Library/Developer/CommandLineTools
#install.packages('lubridate')
#install.packages('readr')
#install.packages('data.table')
#install.packages('plyr')
library(readr)
library(data.table)
library(lattice)
library(plyr)
library(lubridate)

ddir<-"/Users/jasonli/Projects/evaluatingBettingStrategies/"
#ddir<-"/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/"
################### functions for data processing #################
readFileMultiHeader <- function(fname, nline.header=3, ...){
  header <- read_csv(fname, n_max=nline.header, col_names=F)
  header <- c(nullname(apply(as.data.frame(header), 2, function(x){paste(x[!is.na(x)], collapse=":")})))
  content <- read_csv(fname, skip=nline.header, col_names=F, ...)
  names(content) <- header
  content
}
nullname <- function(x) {
  ## make the names of a vector null
  if(!is.null(names(x))) names(x) <- NULL
  x
}
processGameState <- function(game){
  d <- game[,c("teamData:game_id", "gameState:datetime", "gameState:winner", "gameState:teamHome", "gameState:teamAway")]
  names(d) <- c("gameid", "datetime", "winner", "teamHome", "teamAway")
  
  d1 = d2 = d
  names(d1) <- c("gameid", "datetime", "winner", "team", "opp.team")
  d1$teamHome <- d$teamHome
  names(d2) <- c("gameid", "datetime", "winner", "opp.team", "team")
  d2$teamHome <- d$teamHome
  dall <- rbind(d1, d2[,names(d1)])
  dall$y.status = as.integer(dall$team==dall$winner)
  dall$home.or.away <- c("away", "home")[as.integer(dall$team==dall$teamHome)+1]
  dall[order(dall$gameid, dall$home.or.away),]
}

processBooker <- function(booker){
  home.colnmes <- grep("homeProb", names(booker))
  home.cols <- sapply(strsplit(names(booker)[home.colnmes], split=":"), function(x){x[2]})
  home.cols <- gsub(" (%)", "", home.cols, fixed=T)
  
  away.colnmes <- grep("awayProb", names(booker))
  away.cols <- sapply(strsplit(names(booker)[away.colnmes], split=":"), function(x){x[2]})
  away.cols <- gsub(" (%)", "", away.cols, fixed=T)
  
  booker1 <- booker[, c(1,home.colnmes)]
  names(booker1) <- c("gameid", home.cols)
  booker2 <- booker[, c(1,away.colnmes)]
  names(booker2) <- c("gameid", away.cols)
  booker1$home.or.away <- rep("home", nrow(booker1))
  booker2$home.or.away <- rep("away", nrow(booker1))
  booker.all <- rbind(booker1, booker2)
  out <- booker.all[order(booker.all$gameid, booker.all$home.or.away),]
  ##browser()
  names(out)[-c(1,length(out))] <- paste("booker_odds", names(out)[-c(1,length(out))], sep=".")
  out
}


processElo <- function(elo){
  ## team1 columns
  team1.cols <-     c("gameid", "season", "team1", "elo1_pre", "raptor1_pre", "elo_prob1", "raptor_prob1",
                      "team2", "elo2_pre", "raptor2_pre")
  team2.cols <-     c("gameid", "season", "team2", "elo2_pre", "raptor2_pre", "elo_prob2", "raptor_prob2",
                      "team1", "elo1_pre", "raptor1_pre")
  new.team.cols <- c("gameid", "season", "team", "team.elo", "team.elo.raptor", "elo.prob", "raptor.prob",
                      "opp.team", "opp.elo", "opp.elo.raptor")
  elo1 <- elo[,team1.cols]; names(elo1) <- new.team.cols
  elo2 <- elo[,team2.cols]; names(elo2) <- new.team.cols
  
  elo.all <- rbind(elo1, elo2)
  elo.all[order(elo.all$gameid, elo.all$team), c("gameid", "season", "team", "opp.team", "team.elo", "opp.elo", 
                                                 "elo.prob", "team.elo.raptor", "opp.elo.raptor", "raptor.prob")]
}


################### functions for computing the extra columns for machine learning ############
win_probs <- function(home_elo, away_elo, home_court_advantage){
  ## reimplement the Elo_Calculator in R
  ## compute the Elo probabilities
  if(F){
    win_probs(1000, 1000, 40)
  }
  h = 10^(home_elo / 400)
  r = 10^(away_elo / 400)
  a = 10^(home_court_advantage / 400)
  
  denom = r + a * h
  home_prob = a * h / denom
  away_prob = r / denom
  out <- cbind(home_prob, away_prob)
  if(nrow(out)==1) out <- c(out)
  out
}


getFeatureMatrix <- function(dat, per.team.home=F){
  ## obtain feature matrix for regression model
  if(F){
    dat <- data.frame(team=c("A","B"), opp.team=c("B","C"), home.or.away=c("home", "away"))
    getFeatureMatrix(dat, per.team.home=F)
    getFeatureMatrix(dat, per.team.home=T)
  }
  team <- unique(c(dat$team, dat$opp.team))
  nteam <- length(team)
  dat$team.id <- match(dat$team, team)
  dat$opp.id <- match(dat$opp.team, team)
  
  ## game pair assignment
  X <- matrix(0, nrow(dat), nteam)
  X[cbind(1:nrow(X), dat$team.id)] <- 1  ## team assignment
  X[cbind(1:nrow(X), dat$opp.id)] <- -1  ## opp.team assignment
  
  ## per team home court advantage
  if(!per.team.home){
    home.vec <- rep(1, nrow(dat))
    home.vec[dat$home.or.away=="away"] <- -1
    
    Xnew <- cbind(home.vec, X) * log(10)/400
    colnames(Xnew) <- c("home.or.away", team)
  } else {
    ##browser()
    home.mat <- matrix(0, nrow(dat), nteam)
    id <- dat$home.or.away=="home"
    home.mat[rbind(cbind(1:nrow(X), dat$team.id)[id,])] <- 1
    home.mat[rbind(cbind(1:nrow(X), dat$opp.id)[!id,])] <- -1
    
    Xnew <- cbind(home.mat, X) * log(10)/400
    colnames(Xnew) <- c(paste("home", team, sep="."), team)
  }
  
  Xnew
}

adjustEloScore <- function(elo, avg=1500){
  ## Adjust the Elo Scores to have the pre-specified mean
  elo-mean(elo)+avg
}

fitEloScore.lm.old <- function(dat, prob.colnme="booker_odds.Pinnacle", home.court.advantage=NA){
  if(F){
    dat <- new.team.dat[new.team.dat$season=="2020",]
    dat <- dat[(nrow(dat)-70):nrow(dat),]
    
    fit1 <- fitEloScore.lm(dat, prob.colnme="booker_odds.Pinnacle")
    fit2 <- fitEloScore.lm(dat, prob.colnme="raptor.prob")
    c(sqrt(mean(fit1$residuals^2)), sqrt(mean(fit2$residuals^2)))
    plot(adjustEloScore(fit1$coef[-1]), adjustEloScore(fit2$coef[-1]), xlab="Booker", ylab="Raptor")
    abline(0,1,col=2)
  }
  
  dat <- as.data.frame(dat)
  dat$prob <- dat[,prob.colnme]
  if(is.na(home.court.advantage)){  ## if home.court.advantage needs to be estimated
    Xnew <- getFeatureMatrix(dat)
    ##if(is.null(dat$weights)) dat$weights=rep(1, nrow(dat))
    fit <- lm.fit(x=Xnew, y=log((dat$prob)/(1-dat$prob)))
    fit$coef[length(fit$coef)] <- 0
    ##home.advantage <- fit$coef[1]
  } else {
    ##browser()
    Xnew <- getFeatureMatrix(dat)
    Xnew.1 <- Xnew[,-1]
    y <- log((dat$prob)/(1-dat$prob)) - Xnew[,1]*home.court.advantage
    fit <- lm.fit(x=Xnew.1, y=y)
    fit$coef <- c(home.or.away=home.court.advantage, fit$coef)
    fit$coef[length(fit$coef)] <- 0
  }
  
  
  fit
}


fitEloScore.glm <- function(dat, offset.para=NULL, per.team.home=F, ...){
  if(F){
    dat <- new.team.dat[new.team.dat$season>=2021,]
    ##out <- fitEloScore.glm(dat[1:200,])
    dat.fit <- fitEloScore.glm(dat)
    
    dat.fit0 <- fitEloScore.glm(dat, per.team.home=T)
    
    offset.para <- dat.fit$coef[c(1,3, 4)]
    dat.fit2 <- fitEloScore.glm(dat, offset.para=offset.para)
    
    dat.fit3 <- fitEloScore.glm(dat[dat$team=="TOR",], offset.para=dat.fit$coef[-c(1)])
    home.adv <- dat.fit$coef[-1]
    for(tt in names(dat.fit$coef[-1])) {
      print(tt); 
      home.adv[tt] <- fitEloScore.glm(dat[dat$team==tt,], offset.para=dat.fit$coef[-c(1)])$coef[1]
      print(home.adv[tt])
    }
    
    elo.glm <- adjustEloScore(dat.fit$coef[-1])
    est <- ddply(dat, .(team), function(x){data.frame(elo=median(x$elo1_pre))})
    est$glm <- elo.glm[est$team]
    plot(est$elo, est$glm, xlab="median Elo for each team in the season",
         ylab="Estimated Elo from Maximum Likelihood", main="Season 2020")
    abline(0,1,col=2)
  }
  
  ##browser()
  if(is.null(dat$weights)) dat$weights=rep(1, nrow(dat))
  
  Xnew <- getFeatureMatrix(dat, per.team.home=per.team.home)
  df <- data.frame(Xnew)
  df$y.status <- dat$y.status
  nmes <- colnames(Xnew)
  
  if(is.null(offset.para)){
    form2 <- paste(nmes, collapse="+")
    form <- formula(paste("y.status~", form2, "-1", sep=""))
    fit <- glm(form, family=binomial(link="logit"), data=df, weights=dat$weights, ...)
    if(sum(is.na(fit$coef))>0) {
      cat("WARNING:::NA values in fitted coef:", names(fit$coef[is.na(fit$coef)]), "\n")
      fit$coef[is.na(fit$coef)] <- 0
    }
  } else {
    offset.para <- offset.para[names(offset.para) %in% nmes]  ## sometimes offset.para will supply extra, remove those
    para.nmes <- names(offset.para)
    id <- match(para.nmes, nmes)
    for(i in 1:length(id)) Xnew[,id[i]] <- Xnew[,id[i]] * offset.para[i]
    
    
    form1 <- paste("offset(", nmes[id], ")", sep="", collapse="+")
    form2 <- paste(nmes[-id], collapse="+")
    form <- formula(paste("y.status~", form1, "+", form2, "-1", sep=""))
    fit <- glm(form, family=binomial(link="logit"), data=df, weights=dat$weights, ...)
    fit$coef <- c(fit$coef, offset.para)
    ##browser()
    fit$coef <- fit$coef[nmes]
    if(sum(is.na(fit$coef))>0) {cat("NA values in fitted coef:", names(fit$coef[is.na(fit$coef)]), "\n")}
    fit$coef[is.na(fit$coef)] <- 0
  }
  
  fit
}

predictProbUsingEloScore <- function(game.date, elo.prev.date, home.advantage){
  ##est.prev.date <- coef[coef$date==prev.date,]  ## glm estimated elo values based on previous data
  ##home.advantage <- est.prev.date[est.prev.date$team=="home.or.away","estimate"]  ## home advantage 
  ##elo.prev.date <- est.prev.date[est.prev.date$team!="home.or.away","estimate"]
  ##names(elo.prev.date) <- est.prev.date[est.prev.date$team!="home.or.away","team"]
  
  ##game.date <- dat2021[dat2021$date==date,]
  ##browser()
  ngame <- nrow(game.date)
  if(length(home.advantage)==1) home.advantage <- rep(home.advantage, ngame)
  winprob <- rep(NA, ngame)
  home.id <- game.date$home.or.away=="home"
  winprob[home.id] <- rbind(win_probs(home_elo=elo.prev.date[game.date$team[home.id]], away_elo=elo.prev.date[game.date$opp.team[home.id]], 
                                      home_court_advantage = home.advantage[home.id]))[,1]
  winprob[!home.id] <- rbind(win_probs(home_elo=elo.prev.date[game.date$opp.team[!home.id]], away_elo=elo.prev.date[game.date$team[!home.id]], 
                                       home_court_advantage = home.advantage[!home.id]))[,2]
  winprob
}


predictProbUsingEloScore2 <- function(game.date, team.elo.colnme="team.elo", opp.elo.colnme="opp.elo", home.advantage){
  ## similar to predictProbUsingEloScore, but with elo scores from columns of game.date
  ##browser()
  if(length(home.advantage)==1) home.advantage <- rep(home.advantage, nrow(game.date))
  
  game.date <- as.data.frame(game.date)
  ngame <- nrow(game.date)
  #if(length(home.advantage)==1) home.advantage <- rep(1, ngame)
  winprob <- rep(NA, ngame)
  home.id <- game.date$home.or.away=="home"
  ##browser()
  winprob[home.id] <- rbind(win_probs(home_elo=c(game.date[,team.elo.colnme])[home.id], 
                                      away_elo=c(game.date[,opp.elo.colnme])[home.id], 
                                      home_court_advantage = home.advantage[home.id]))[,1]
  winprob[!home.id] <- rbind(win_probs(home_elo=c(game.date[,opp.elo.colnme])[!home.id], 
                                       away_elo=c(game.date[,team.elo.colnme])[!home.id], 
                                       home_court_advantage = home.advantage[!home.id]))[,2]
  winprob
}

simuGamesFromElo <- function(elo, ngame.pair=2, home.court.advantage=50, seed=1){
  if(F){
    ##elo <- elo-median(elo)+1500
    elo <- adjustEloScore(fit$coef[-1])
    df <- simuGamesFromElo(elo, 6)
    
    new.team.dat.home <- new.team.dat[new.team.dat$home.or.away=="home",]
    pair <- paste(new.team.dat.home[1:100,]$team, new.team.dat.home[1:100,]$opp.team)
    df.samp <- df[match(pair, paste(df$team, df$opp.team)),]
    
    if(F) df.fit <- fitEloScore.glm(df)
    df.fit <- fitEloScore.glm(df.samp)
    elo.fit <- df.fit$coef[-1]
    elo.fit <- elo.fit[names(elo)]
    plot(elo, adjustEloScore(elo.fit))
    abline(0,1,col=2)
  }
  
  df1 <- expand.grid(team=names(elo), opp.team=names(elo), game.idx=1:(ngame.pair/2), home.or.away="home")
  df2 <- expand.grid(team=names(elo), opp.team=names(elo), game.idx=(ngame.pair/2+1):ngame.pair, home.or.away="away")
  df <- rbind(df1, df2)
  df <- df[df$team!=df$opp.team,]
  df$team <- as.character(df$team)
  df$opp.team <- as.character(df$opp.team)
  df$elo <- elo[df$team]
  df$opp.elo <- elo[df$opp.team]
  
  ##browser()
  id <- df$home.or.away=="home"
  winprob <- rep(NA, nrow(df))
  winprob[id] <- win_probs(home_elo=df$elo[id], away_elo=df$opp.elo[id], home_court_advantage=home.court.advantage)[,1]
  winprob[!id] <- win_probs(home_elo=df$opp.elo[!id], away_elo=df$elo[!id], home_court_advantage=home.court.advantage)[,2]
  df$winprob <- winprob ##win_probs(home_elo=df$elo, away_elo=df$opp.elo, home_court_advantage=home.court.advantage)
  if(!is.na(seed)) set.seed(seed)
  df$y.status <- rbinom(nrow(df),1,df$winprob)
  df
}



################### join the data #######################

booker <- readFileMultiHeader(paste(ddir,"data/bettingOddsData/adj_prob_win_ALL.csv",sep=""), nline.header=3, guess_max=2e5)
game <- readFileMultiHeader(paste(ddir,"data/gameStats/game_state_data_ALL.csv",sep=""), nline.header=3, guess_max=2e5)

elo <- read_csv(paste(ddir,"data/eloData/nba_elo_all.csv",sep=""), guess_max=2e5)
names(elo)[1] <- "gameid"

game2 <- processGameState(game)
booker2 <- processBooker(booker)
elo2 <- processElo(elo)

dat <- merge(game2, booker2, all.y=T, by=c("gameid", "home.or.away"))
dat <- merge(dat, elo2, by=c("gameid", "team", "opp.team"))
dat <- dat[order(dat$gameid, dat$home.or.away),]
dat$date  <- as_date(substring(dat$gameid, 1, 8))
#extra <- dat[!(dat$gameid %in% new.team.dat$gameid),]
#table(extra$date)
## why there are extra games??? not sure why

output.names <- c("gameid", "team", "season", "y.status", "date", "team.elo.booker.lm", "opp.elo.booker.lm", "team.elo.booker.combined",
                  "opp.elo.booker.combined", "booker_odds.Pinnacle", "elo.prob", "raptor.prob", "predict.prob.booker", "predict.prob.combined", 
                  "elo.court30.prob", "raptor.court30.prob")
    

######################################################
if(T){
  ## 10/31/2022: with the updated scheme
  ## 9/29/2022
  ## add inferred Elo Scores from fitting an inverse model from the booker prob, and use that to predict ahead
  
  ##10/4/2022: try using weights from y.spread
  
  newdf.all <- NULL
  
  #for(season in 2015:2022){
  for(season in 2023){
    cat("#################################################season=", season, "\n")
    
    if(F) dat2021 <- dat.after2019.clean[dat.after2019.clean$season==season, ]
    if(F) dat2021 <- new.team.dat[new.team.dat$season==season, ]
    if(F) dat2021 <- new.team.dat.clean[new.team.dat.clean$season==season, ]
    dat2021 <- dat[dat$season==season,]
    udatetime <- unique(dat2021$datetime)
    
    newdf <- NULL
    coef <- NULL
    for(i in 30:length(udatetime)){
      print(i)
      datetime <- udatetime[i]   ## current date for prediction
      ##past.dat <- dat2021[dat2021$date<=date,] 
      past.dat <- dat2021[dat2021$datetime<datetime,] ## only up to the datetime of current games
      start.date <- past.dat[nrow(past.dat)-70,"date"]
      ##prev.dat <- past.dat[past.dat$date>=start.date,]
      prev.dat <- past.dat[past.dat$date>=start.date,]
      
      game.date <- dat2021[dat2021$datetime==datetime,]
      date <- game.date$date[1]  ## game.date
      
      offset.home.court.advantage = 30
      ##offset.home.court.advantage = 100
      fit1 <- fitEloScore.lm.old(prev.dat, prob.colnme="booker_odds.Pinnacle", home.court.advantage = offset.home.court.advantage)
      if(F){
        plot(fit1$residuals+fitted(fit1), fitted(fit1))
        abline(0,1,col=2)
        abline(-0.5,1,col=2,lty=2)
        abline(0.5,1,col=2,lty=2)
        id <- abs(fit1$residuals)>0.5
        points((fit1$residuals+fitted(fit1))[id], fitted(fit1)[id], col=2, pch=2, cex=2)
        
        plot(prev.dat$date, fit1$residuals)
        abline(h=c(-0.5,0,0.5), col=2)
        points(prev.dat$date[id], fit1$residuals[id], col=2, pch=2, cex=2)
        cat("std of residuals = ",  sqrt(mean(fit1$residuals^2)), "\n")
      }
      
      if(T){
        if(sum(is.na(fit1$coef))>0) {
          ##browser()
          ##fit1$coef[is.na(fit1$coef)] <- 0
          prev.dat <- past.dat[past.dat$date>=start.date-3,]  ## 3 more days (before data for 70 games)
          fit1 <- fitEloScore.lm.old(prev.dat, prob.colnme="booker_odds.Pinnacle", home.court.advantage = offset.home.court.advantage)
          #plot(fit1$residuals+fitted(fit1), fitted(fit1))
          #abline(0,1,col=2)
        }
      }
      
      ##fit1.status <- fitEloScore.glm(prev.dat, start=fit1$coef, control=glm.control(maxit=5, trace=T))
      game.date$predict.prob.booker <- predictProbUsingEloScore(game.date, fit1$coef[-1], fit1$coef[1])
      fit1$coef[-1] <- adjustEloScore(fit1$coef[-1])
      game.date$team.elo.booker.lm <- fit1$coef[-1][game.date$team]
      game.date$opp.elo.booker.lm <- fit1$coef[-1][game.date$opp.team]
      
      ##coef1 <- fit1$coef
      ##coef1[-1] <- adjustEloScore(coef1[-1])
      
      ##dd <- data.frame(date=rep(date, length(coef1)), team=names(coef1), elo.booker.lm=nullname(coef1))
      ##coef <- rbind(coef, dd)
      
      
      if(all(!is.na(fit1$coef))){
        dat.simu <- simuGamesFromElo(elo=fit1$coef[-1], ngame.pair=20, home.court.advantage=offset.home.court.advantage)
        dat.simu$weights <- 0.03
        prev.dat <- prev.dat[prev.dat$date!=date,]
        if(T) prev.dat$weights <- 1
        if(F) prev.dat$weights <-  weightsFromSpread(prev.dat)
        all.dat <- rbind(dat.simu[,c("team","opp.team","home.or.away", "y.status","weights")], 
                         prev.dat[,c("team","opp.team","home.or.away", "y.status","weights")])
        fit <- fitEloScore.glm(all.dat, offset.para=c(home.or.away=offset.home.court.advantage))
        fit$coef[-1] <- adjustEloScore(fit$coef[-1])
        if(F){
          b <- adjustEloScore(fit$coef[-1]) ## combined
          a <- adjustEloScore(fit1$coef[-1]) ## original
          b <- b[names(a)]
          plot(a,b, xlab="booker", ylab="booker+elo")
          abline(0,1,col=2)
        }
        
        game.date$predict.prob.combined <- predictProbUsingEloScore(game.date, fit$coef[-1], fit$coef[1])
        game.date$team.elo.booker.combined <- fit$coef[-1][game.date$team]
        game.date$opp.elo.booker.combined <- fit$coef[-1][game.date$opp.team]
        game.date$elo.court30.prob <- predictProbUsingEloScore2(game.date, team.elo.colnme="team.elo", opp.elo.colnme="opp.elo", 
                                                                home.advantage=offset.home.court.advantage)
        if(season>=2019){
          game.date$raptor.court30.prob <- predictProbUsingEloScore2(game.date, team.elo.colnme="team.elo.raptor", opp.elo.colnme="opp.elo.raptor", 
                                                                     home.advantage=offset.home.court.advantage)
        } else game.date$raptor.court30.prob <- rep(NA, nrow(game.date))
        
        newdf <- rbind(newdf, game.date)
      }
    }
    newdf.all <- rbind(newdf.all, newdf)
  }
  olddf<-read_csv(paste(ddir,"data/eloData/adj_elo_ml_2015_2022.csv",sep=""), guess_max=2e5)
  alldf = rbind(olddf,newdf.all[,output.names]) 
  write_csv(alldf, file=paste(ddir,"data/eloData/adj_elo_ml.csv",sep=""))
  #write_csv(newdf.all[,output.names], file = paste (ddir, "data/eloData/adj_elo_ml_2015_2022.csv", sep=""))
  
  ## accuracy analysis
  if(F){
    newdf.all.orig <- newdf.all
    newdf.all <- newdf.all[!is.na(newdf.all$y.status),]
    prob.colnmes <- c("booker_odds.Pinnacle", "elo.prob", "elo.court30.prob",
                      "raptor.prob", "raptor.court30.prob", "predict.prob.booker", "predict.prob.combined")
    getAccuracy(newdf.all, prob.colnmes=prob.colnmes, layout="horizontal")
    ##horizontal=T)
    adf <- getAccuracy(newdf.all, prob.colnmes=prob.colnmes, layout="vertical")
    xyplot(accu~season, group=class, data=adf, scales=list(x=list(relation="free")), type="l", auto.key=T)
  }
}





