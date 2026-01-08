library(TreeSim)
library(ape)

num_trees <- 1000
output_folder <- "simulated_trees"
dir.create(output_folder, showWarnings = FALSE)
write.csv(data.frame(
  tree_id = integer(),
  lambda1 = numeric(),
  lambda2 = numeric(),
  mu = numeric(),
  psi = numeric(),
  t_1 = numeric(),
  n_tips = integer()
), file.path(output_folder, "parameters.csv"), row.names = FALSE)

count <- 0
attempts <- 0
max_attempts <- 50000 
t_0 <- 200

set.seed(123)

while (count < num_trees && attempts < max_attempts) {
  attempts <- attempts + 1
  
  lambda1 <- runif(1, 0.3, 1.0)        
  mu_ratio <- runif(1, 0.05, 0.4)      
  mu <- mu_ratio * lambda1
  psi <- runif(1, 0.5, 0.9)            
  
  lambda2 <- runif(1, 2.1 * lambda1, 6)  
  
  t_1 <- runif(1, 0.1 * t_0, 0.5 * t_0)  
  n_target <- sample(80:400, 1)          
  
  tree_result <- tryCatch({
    sim.bdsky.stt(
      n = n_target,
      lambdasky = c(lambda1, lambda2),
      deathsky = c(mu, mu),
      sampprobsky = c(psi, psi),
      timesky = c(0, t_1),
      rho = 0,
      timestop = 0,
      model = "BD"
    )
  }, error = function(e) NULL)
  
  if (is.null(tree_result)) next
  if (is.numeric(tree_result) && length(tree_result) == 1) next
  if (!is.list(tree_result) || length(tree_result) == 0) next
  
  tree <- tree_result[[1]]
  if (!inherits(tree, "phylo")) next
  
  n_real <- Ntip(tree)
  if (n_real < 50 || n_real > 1000) next
  
  count <- count + 1
  
  write.tree(tree, file.path(output_folder, paste0("tree_", count, ".nwk")))
  
  write.table(
    data.frame(count, lambda1, lambda2, mu, psi, t_1, n_real),
    file.path(output_folder, "parameters.csv"),
    sep = ",", append = TRUE, col.names = FALSE, row.names = FALSE
  )
  
  if (count %% 100 == 0) {
    cat("Generated", count, "trees\n")
  }
}

cat("Done. Generated", count, "trees.\n")