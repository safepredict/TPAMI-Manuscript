# Libraries ---------------------------------------------------------------
library(keras)
library(densenet)
library(useful)
library(tidyverse)
library(scales)
library(ggthemes)
library(hrbrthemes)

# ----
# basic implementation of the algorithm
# -----
algorithm5_static <- function(loss_seq, w_p1 = 0.5, target_eps = 0.1, alpha = 0.1){
  TT <- length(loss_seq);   t = 1;   k = 1;   w_d <-  1-w_p1
  W_p <-  rep(0, TT)
  while(t <= TT){
    w_p = w_p1
    V_sum = 0
    eta = sqrt(-log((1-w_p1) *(1-alpha)^(TT-1))/(1-target_eps)^2 / 2^k)
    while((V_sum <= 2^k) & (t <= TT)){
      W_p[t] <- w_p
      w_p <- alpha + (1-alpha)  / ( 1 + w_d/w_p*exp(eta * (loss_seq[t] - target_eps)))
      w_d <- 1-w_p
      V_sum <- V_sum + w_p * w_d
      t <- t + 1} 
    k <- k + 1}
  return(W_p)
}



algorithm5_single_step <- function(loss = 1, w_p = 0.5, target_eps = 0.1, alpha = 0.1, eta, k , V_sum){
  w_d = 1-w_p
  wp <- alpha + (1-alpha)  / ( 1 + w_d/w_p*exp(eta * (loss - target_eps)))
  Vsum <- V_sum + wp*(1-wp)
  kk <- ifelse(Vsum <= 2^k, k, k+1)
  return(c(wp, Vsum, k))
}


algorithm5_batch <-  function(loss_seq, w_p = 0.5, target_eps = 0.1, alpha = 0.1, TT0 = 60000, V_sum = 0, k = 1){
  TT <- length(loss_seq);   W_p <-  rep(0, TT)
  wp <- w_p; wd <-  1-wp; kk = k; Vsum = V_sum
  
  eta <-  sqrt(-log((1-w_p) *(1-alpha)^(TT0-1))/(1-target_eps)^2 / 2^kk)
  for(t in 1:TT){
    res <- algorithm5_single_step(loss = loss_seq[t], w_p = wp, target_eps = target_eps, alpha = alpha, eta = eta, k = kk, V_sum = Vsum)
    wp <- res[1]; Vsum <- res[2]; kk <- res[3]
    W_p[t] <- wp
  }
  return(list(W_p, wp, Vsum, kk))
}
# ----
# Parameters --------------------------------------------------------------

batch_size <- 64
epochs <- 20
tar_eps = .08
# Data Preparation --------------------------------------------------------

# see ?dataset_cifar10 for more info
#ddata <- dataset_cifar10()
data <- dataset_mnist()

x_train <- data$train$x
x_test <- data$test$x

y_train <- to_categorical(data$train$y, num_classes = 10)
y_test <- to_categorical(data$test$y, num_classes = 10)

x <- abind::abind(x_train,x_test, along = 1)
y <- abind::abind(y_train,y_test, along = 1)
x <- x / 255
rm(x_train, x_test, y_train, y_test, data)

set.seed(42)
random_order <- sample(dim(x)[1])
x <- x[random_order,,] # x[ramdom_order,,,]
x <- array_reshape(x, c(nrow(x), 784))
y <- y[random_order,]

# introduce change-points - one in every 10k
NC = 35
for(ix in 1:(NC-1)){
  pp <- sample(10)
  y[ix*70000/NC + (1:70000/NC), ] <-   y[ix*70000/NC + (1:70000/NC), pp]
}

# Model Parameters -----------------------------------------------------------

epoch_size = 200 ; TT0 = 70000; N <- floor(10000/epoch_size)
alpha = NC/70000

pr <- matrix(NA, dim(y)[1], dim(y)[2])
pr_cbr <- matrix(NA, dim(y)[1], dim(y)[2])
pr_amnesic <- matrix(NA, dim(y)[1], dim(y)[2])


ref_cbr <- rep(NA, dim(y)[1])
ref_amnesic <- rep(NA, dim(y)[1])

ths_cbr <- rep(1, N-1)
ths_amnesic <- rep(1, N-1)

k = 1; k_amnesic = 1; k_cbr = 1
Vsum = 0; Vsum_amnesic = 0; Vsum_cbr = 0
wp = 0.5; wp_amnesic = 0.5; wp_cbr = 0.5

WWp_amnesic <- rep(0,TT0)
WWp_cbr <- rep(0,TT0)
WWp <- rep(0,TT0)
resets <- c()
ms <- c()
ls <- c()



# Model Definition -------------------------------------------------------

x_train = x[(1:epoch_size),]
y_train = y[(1:epoch_size),]
x_test = x[(1:epoch_size),]
y_test = y[(1:epoch_size),]

model <- keras_model_sequential()  %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

model_batch <- keras_model_sequential()  %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')


model %>% compile(
  optimizer = optimizer_adam(), #optimizer_rmsprop(lr = 0.01, decay = 0.05),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

model_batch %>% compile(
  optimizer = optimizer_adam(), #optimizer_rmsprop(lr = 0.01, decay = 0.05), 
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)


history <- model %>% fit(  
  x_train, y_train, 
  validation_data = list(x_test, y_test),
  batch_size = batch_size, 
  epochs = epochs
)

history_batch <- model_batch %>% fit(
  x_train, y_train, 
  validation_data = list(x_test, y_test),
  batch_size = batch_size, 
  epochs = epochs
)

IE = epochs
IE0 = epochs

init = 1
init_batch = 1


ixs <- c(); mms <- c(); lls <- c()
for(ix in 2:(N-1)){
#for(ix in 2:(20000/epoch_size)){
  print(ix)
  x_train = x[init:(ix*epoch_size),]
  y_train = y[init:(ix*epoch_size),]
  x_val = x[(ix-1)*epoch_size + (1:epoch_size),]
  y_val = y[(ix-1)*epoch_size + (1:epoch_size),]
  
  x_train_batch = x[init_batch:(ix*epoch_size),]
  y_train_batch = y[init_batch:(ix*epoch_size),]
  x_test = x[(ix)*epoch_size + (1:epoch_size),]
  y_test = y[(ix)*epoch_size + (1:epoch_size),]
  
  # AMNESIC
  pr_temp <-  predict(model_batch, x_val)
  prs <- apply(pr_temp,1 ,max)
  for(pr_th in sort(unique(prs))){
    if(mean(apply(pr_temp,1 ,which.max)[prs >= pr_th] != 
            apply(y_val,1 ,which.max)[prs >= pr_th]) <= tar_eps){
      ths_amnesic[ix] <- ifelse(is.na(ths_amnesic[ix]), 1,pr_th)
      break
    }
  }
  ths_amnesic[ix] <- ifelse(is.na(ths_amnesic[ix]), 1,ths_amnesic[ix])
  
  pr_temp <-  predict(model_batch, x_test)
  pr_amnesic[(ix)*epoch_size + (1:epoch_size),] <- pr_temp
  prs <- apply(pr_temp,1,max)
  ref_amnesic[(ix)*epoch_size + (1:epoch_size)] <- prs  < ths_amnesic[ix]
  
  l <- apply(pr_temp, 1, which.max) != apply(y_test,1, which.max)
  l_r <- ifelse(prs  < ths_amnesic[ix], tar_eps, l)
  
  res <- algorithm5_batch(l_r, w_p = wp_amnesic, target_eps = tar_eps, alpha = alpha, TT0 = TT0, V_sum = Vsum_amnesic, k = k_amnesic)
  wp_amnesic <- res[[2]]; Vsum_amnesic <- res[[3]]; k_amnesic <- res[[4]]
  WWp_amnesic[(ix)*epoch_size + (1:epoch_size)] <- res[[1]]
  
  m <- mean(res[[1]][prs  >= ths_amnesic[ix]])
  ll <- sum(prs  >= ths_amnesic[ix])
  ixs <- c(ixs, ix); mms <- c(mms, m); lls <- c(lls, ll)
  if((ifelse(is.na(m), 0, m) < 0.01) &  #(ifelse(is.na(ll), 0, ll) > 10) & 
     (IE > 0) ){
    ths_amnesic[ix+1] <- NA
    print(paste0("reset ", m, " ", ll))
    resets <- c(resets, ix*epoch_size)
    ms <- c(ms, m)
    ls <- c(ls, ll)
    IE = 0
    init_batch = ix*epoch_size
  } else {
    
    model_batch <- keras_model_sequential()  %>% 
      layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
      layer_dropout(rate = 0.4) %>% 
      layer_dense(units = 128, activation = 'relu') %>%
      layer_dropout(rate = 0.3) %>%
      layer_dense(units = 10, activation = 'softmax')
    
    model_batch %>% compile(
      optimizer = optimizer_adam() ,#optimizer_rmsprop(lr = 0.01, decay = 0.05),
      loss = "categorical_crossentropy",
      metrics = "accuracy"
    )
    
    history_batch <- model_batch %>% fit(
      x_train_batch, y_train_batch, 
      validation_data = list(x_test, y_test),
      batch_size = batch_size, 
      epochs = epochs #, #*ifelse(ths_amnesic[ix] == 1, ix, 1)
      #initial_epoch = IE 
    )
    IE = IE + epochs
    
  }
  
  
  
  # STATIC
  pr_temp <-  predict(model, x_val)
  prs <- apply(pr_temp,1 ,max)
  for(pr_th in sort(unique(prs))){
    if(mean(apply(pr_temp,1 ,which.max)[prs >= pr_th] != 
            apply(y_val,1 ,which.max)[prs >= pr_th]) <= tar_eps){
      ths_cbr[ix] <- pr_th
      break
    }
  } 
  ths_cbr[ix] <- ifelse(is.na(ths_cbr[ix]), 1,ths_cbr[ix])
  pr_temp <-  predict(model_batch, x_test)
  pr_cbr[(ix)*epoch_size + (1:epoch_size),] <- pr_temp
  prs <- apply(pr_temp,1,max)
  ref_cbr[(ix)*epoch_size + (1:epoch_size)] <- prs  < ths_cbr[ix]
  
  l <- apply(pr_temp, 1, which.max) != apply(y_test,1, which.max)
  l_r <- ifelse(prs  < ths_cbr[ix], tar_eps, l)
  
  res <- algorithm5_batch(l_r, w_p = wp_cbr, target_eps = tar_eps, alpha = alpha, TT0 = TT0, V_sum = Vsum_cbr, k = k_cbr)
  wp_cbr <- res[[2]]; Vsum_cbr <- res[[3]]; k_cbr <- res[[4]]
  WWp_cbr[(ix)*epoch_size + (1:epoch_size)] <- res[[1]]
  
  
  model <- keras_model_sequential()  %>% 
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    optimizer = optimizer_adam() ,#optimizer_rmsprop(lr = 0.01, decay = 0.05),
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )
  
  
  
  history <- model %>% fit(
    x_train, y_train, 
    validation_data = list(x_test, y_test),
    batch_size = batch_size, 
    epochs = epochs # + IE0,  
    #initial_epoch = IE0 
  )
  IE0 = IE0 + epochs
  
  
  pr_temp <-  predict(model_batch, x_test)
  pr[(ix)*epoch_size + (1:epoch_size),] <- pr_temp
  l <- apply(pr_temp, 1, which.max) != apply(y_test,1, which.max)
  
  res <- algorithm5_batch(l, w_p = wp, target_eps = tar_eps, alpha = alpha, TT0 = TT0, V_sum = Vsum, k = k)
  wp <- res[[2]]; Vsum <- res[[3]]; k <- res[[4]]
  WWp[(ix)*epoch_size + (1:epoch_size)] <- res[[1]]
  
}


tibble(ix = ixs, ms = mms, ls = lls) %>%
  dplyr::mutate(res = as.integer(ix*epoch_size) %in% resets ) %>% View(  )

# ----

RESULTS <- tibble(ix = 1:dim(y)[1], 
                  y = apply(y[1:dim(pr)[1],],1, which.max),
                  y_hat = apply(pr, 1, function(x){tx <- which.max(x); ifelse(length(tx) == 1, tx, NA)}),
                  y_hat_cbr = apply(pr_cbr, 1, function(x){tx <- which.max(x); ifelse(length(tx) == 1, tx, NA)}),
                  y_hat_amnesic = apply(pr_amnesic, 1, function(x){tx <- which.max(x); ifelse(length(tx) == 1, tx, NA)}),
                  ref_cbr = ref_cbr, 
                  ref_amnesic = ref_amnesic,
                  safePredictWeights_amnesic = WWp_amnesic,
                  safePredictWeights = WWp, 
                  safePredictWeights_CBR = WWp_cbr) %>% 
  dplyr::mutate(loss_sp = (y != y_hat), 
                loss_cbr = (y != y_hat_cbr),
                loss_amnesic = (y != y_hat_amnesic), 
                is_reset  = ix %in% resets) %>% 
  dplyr::filter(!is.na(y_hat)) 




write_csv(RESULTS, "~/Desktop/safe_predict_cfar/mnist_results_08.csv")


p1 <- RESULTS %>%
  ggplot() + 
  geom_line(aes(ix,1, color = "Base"), lwd = 1) + 
  geom_line(aes(ix,cumsum(safePredictWeights)/ix, color = "SP"), lwd = 1) + 
  geom_line(aes(ix,cumsum(1-ref_cbr)/ix, color = "CBR"), lwd = 1) + 
  geom_line(aes(ix,cumsum(safePredictWeights_amnesic*(1-ref_amnesic))/ix, color = "Amnesic SP"), lwd=1) + 
  geom_line(aes(ix,cumsum(safePredictWeights_CBR*(1-ref_cbr))/ix, color = "SP + CBR"), lwd=1) + 
  geom_vline(aes(xintercept = ifelse(is_reset, ix, NA)), lty = 2, lwd = 1, alpha = 0.8) + 
  theme_ipsum(axis_title_size = 15) + theme(legend.position = "bottom") + scale_color_wsj() + 
  labs(x = "t", y = "efficiency", color = "")

p1

p2 <- RESULTS %>%
  ggplot() + 
  geom_line(aes(ix,cumsum(loss_sp)/ix, color = "Base"), lwd = 1) + 
  geom_line(aes(ix,cumsum(loss_sp*safePredictWeights)/cumsum(safePredictWeights), color = "SP"), lwd = 1) + 
  geom_line(aes(ix,cumsum(loss_cbr*(1-ref_cbr))/cumsum(1-ref_cbr), color = "CBR"), lwd = 1) + 
  geom_line(aes(ix,cumsum(safePredictWeights_amnesic*(1-ref_amnesic)*loss_amnesic)/cumsum(safePredictWeights_amnesic*(1-ref_amnesic)),
                color = "Amnesic SP"), lwd=1) + 
  geom_line(aes(ix,cumsum(safePredictWeights_CBR*(1-ref_cbr)*loss_cbr)/cumsum(safePredictWeights_CBR*(1-ref_cbr)), color = "SP + CBR"), lwd=1) + 
  geom_vline(aes(xintercept = ifelse(is_reset, ix, NA)), lty = 2, lwd = 1, alpha = 0.8) + 
  theme_ipsum(axis_title_size = 15) + theme(legend.position = "bottom") + scale_color_wsj() + 
  labs(x = "t", y = "error rate", color = "")

p2
cowplot::plot_grid(p1,p2,nrow =2)
