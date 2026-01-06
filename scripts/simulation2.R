library(TreeSim)
library(ape)

num_trees <- 1500
output_folder <- "simulated_trees"
dir.create(output_folder, showWarnings = FALSE)

params_file <- file.path(output_folder, "parameters.csv")

header <- data.frame(
  tree_id = integer(),
  lambda1 = numeric(),
  lambda2 = numeric(),
  mu = numeric(),
  psi = numeric(),
  t_1 = numeric(),
  n_tips = integer()
)

write.csv(header, file = params_file, row.names = FALSE)

count <- 0
attempts <- 0
max_attempts <- 3000000

set.seed(42)

total_duration <- 200
min_interval <- 0.05 * total_duration

while (count < num_trees && attempts < max_attempts) {
  attempts <- attempts + 1
  
  lambda1 <- runif(1, 0.3, 1.2)
  ratio_mu <- runif(1, 0, 0.8)
  mu <- ratio_mu * lambda1
  psi <- runif(1, 0.2, 0.6)
  lambda2 <- runif(1, 2.2 * lambda1, 6)
  t_1 <- runif(1, min_interval + 5, total_duration - min_interval - 5)
  
  n_tips_target <- sample(100:1000, 1)
  
  timesky <- c(0, t_1)
  lambdasky <- c(lambda1, lambda2)
  deathsky <- rep(mu, 2)
  sampprobsky <- rep(psi, 2)
  
  tree_list <- tryCatch({
    sim.bdsky.stt(
      n = n_tips_target,
      lambdasky = lambdasky,
      deathsky = deathsky,
      sampprobsky = sampprobsky,
      timesky = timesky,
      timestop = 0
    )
  }, error = function(e) NULL)
  
  if (is.null(tree_list) || length(tree_list) == 0)
    next
  
  tree <- tree_list[[1]]
  
  if (is.null(tree))
    next
  
  n_real <- Ntip(tree)
  
  if (n_real < 100 || n_real > 1000)
    next
  
  count <- count + 1
  
  tree_file <- file.path(output_folder, paste0("tree_", count, ".nwk"))
  write.tree(tree, file = tree_file)
  
  new_row <- data.frame(
    tree_id = count,
    lambda1 = lambda1,
    lambda2 = lambda2,
    mu = mu,
    psi = psi,
    t_1 = t_1,
    n_tips = n_real
  )
  
  write.table(
    new_row,
    file = params_file,
    sep = ",",
    append = TRUE,
    col.names = FALSE,
    row.names = FALSE
  )
  
  if (count %% 50 == 0) {
    cat("Saved", count, "/", num_trees,
        "| attempts:", attempts,
        "| acceptance:", round(count / attempts, 4), "\n")
  }
}

cat("\nDONE\nSaved:", count, "trees\nAttempts:", attempts, "\n")

if (count < num_trees) {
  cat("WARNING: Did not reach", num_trees, "trees before max_attempts.\n")
  cat("Most minimal improvement is to increase psi upper bound (e.g., 0.8 or 0.9).\n")
}
