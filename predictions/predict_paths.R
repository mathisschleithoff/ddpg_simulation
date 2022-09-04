# Deterministic Growth Model: comparison of predicted paths

# optimal path
c = rep(NA, 50)
k = rep(NA, 50)
k[1] <- 10
inv = rep(NA, 50)
alpha = 0.8
beta = 0.95
t = 100
eps_length = 20
A = 1

for (i in (1:(t-1))){
  c[i] = (1 - beta*alpha)*A*k[i]^alpha
  k[i+1] = k[i]^alpha - c[i]
  inv[i] = k[i]^alpha - c[i]
}

# load predictions of DDPG agent
library(reticulate)
np <- import("numpy")
# set working directory: folder "predictions"
setwd()
file <- "predictions_1.npz"
file <- np$load(file)
actions <- file$f[["rl_actions"]]

# comparison of paths
plot(seq(0,99), actions, type = "l", col = "gold", ylim = c(0,5), xlim = c(-0.5,30),
     ylab = "Investment", xlab = "Period", cex.axis = 1.0, cex.lab = 1.0,
     yaxs = "i", xaxs = "i", lwd = 3)
lines(seq(0,98), inv, type = "l", col = "black", lwd = 3, lty = 3)


# Stochastic Growth Model: comparison of predicted paths

# load predictions of DDPG agent and optimal actions
library(reticulate)
np <- import("numpy")
# set working directory: folder predictions
setwd()
file = "predictions_2.npz"
file <- np$load(file)
rl_actions <- file$f[["rl_actions"]]
opt_actions <- file$f[["opt_actions"]]

# comparison of paths
plot(seq(0,49), rl_actions, type = "l", col = "gold", ylim = c(0,5), xlim = c(-0.5,30),
     ylab = "Investment", xlab = "Period", cex.axis = 1.0, cex.lab = 1.0,
     yaxs = "i", xaxs = "i", lwd = 3)
lines(seq(0,49), opt_actions, type = "l", col = "black", lwd = 3, lty = 3)


# Stochastic Growth Model with Divisible Labor

# load predictions of DDPG agent and LQ-approximation
library(reticulate)
np <- import("numpy")
# set working directory: folder predictions
setwd()
file = "predictions_3.npz"
file <- np$load(file)
rl_labor <- file$f[["rl_labor"]]
rl_inv <- file$f[["rl_inv"]]
lq_labor <- file$f[["lq_labor"]]
lq_inv <- file$f[["lq_inv"]]

# predicted paths for labor
plot(seq(0,49), rl_labor[1:50,], type = "l", col = "gold", ylim = c(0,0.6), xlim = c(-0.5,50),
     ylab = "Labor", xlab = "Period", cex.axis = 1.0, cex.lab = 1.0,
     yaxs = "i", xaxs = "i", lwd = 3)
lines(seq(0,49), lq_labor[1:50], type = "l", col = "black", lwd = 3, lty = 3)

# predicted paths for investment
plot(seq(0,49), rl_inv[1:50,], type = "l", col = "gold", ylim = c(0,0.6), xlim = c(-0.5,50),
     ylab = "Investment", xlab = "Period", cex.axis = 1.0, cex.lab = 1.0,
     yaxs = "i", xaxs = "i", lwd = 3)
lines(seq(0,49), lq_inv[1:50], type = "l", col = "black", lwd = 3, lty = 3)

