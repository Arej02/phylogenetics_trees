library(TreeSim)
library(ape)

num_trees <- 100

output_folder <- "simulated_trees"
dir.create(output_folder, showWarnings = FALSE)

trees <- list()
params_df <- data.frame(
  tree_id = integer(),
  lambda = numeric(),
  lambda2 = numeric(),
  mu = numeric(),
  psi = numeric(),
  t_1 = numeric(),
  n_tips = integer()
)

count <- 0

set.seed(42)  

while (count < num_trees) {
  lambda <- runif(1, 0.3, 1.2) # VS(U(1,0.01,1)) -> cannot generalize towards sparse trees
  ratio_mu <- runif(1, 0, 0.8)         
  mu <- ratio_mu * lambda
  psi <- runif(1, 0.2, 0.6)             
  lambda2 <- runif(1, 2.2 * lambda, 6) 
  total_duration <- 200
  min_interval <- 0.05 * total_duration
  t_1 <- runif(1, min_interval + 5, total_duration - min_interval - 5)
  
  n_tips <- sample(100:1000, 1)
  
  timesky <- c(0, t_1)
  lambdasky <- c(lambda, lambda2)
  deathsky <- rep(mu, 2)
  sampprobsky <- rep(psi, 2)
  
  tree_list <- tryCatch({
    sim.bdsky.stt(n = n_tips, # VS time_stop=200 ( Stopping criteria is tips count and not time)
                  lambdasky = lambdasky,
                  deathsky = deathsky,
                  sampprobsky = sampprobsky,
                  timesky = timesky,
                  timestop = 0)
  }, error = function(e) { NULL })
  
  if (is.null(tree_list) || length(tree_list) == 0) {
    next
  }
  
  tree <- tree_list[[1]]
  if (!is.null(tree) && Ntip(tree) == n_tips) {
    count <- count + 1
    
    tree_file <- file.path(output_folder, paste0("tree_", count, ".nwk"))
    write.tree(tree, file = tree_file)
    
    params_df <- rbind(params_df, data.frame(
      tree_id = count,
      lambda = lambda,
      lambda2 = lambda2,
      mu = mu,
      psi = psi,
      t_1 = t_1,
      n_tips = n_tips
    ))
    
    cat("Successfully simulated and saved tree", count, "with", n_tips, "tips\n")
  }
}

write.csv(params_df, file = file.path(output_folder, "parameters.csv"), row.names = FALSE)

cat("All", num_trees, "trees simulated and saved in", output_folder, "\n")