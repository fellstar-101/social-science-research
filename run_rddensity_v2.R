# ============================================================
# R Script: rddensity tests for statewide ballot measures
# Version 2: includes fiscal/non-fiscal density split
#
# Run this on your local machine with R and the rddensity package
# install.packages("rddensity") if needed
# ============================================================

library(rddensity)

# Set working directory to wherever the margin CSV files are
setwd("C:/Users/class/Downloads/UChicago/13300/results/attempt1")

run_density_test <- function(filepath, label) {
  cat("\n===", label, "===\n")
  data <- read.csv(filepath)
  margins <- data$margin
  cat("N =", length(margins), "\n")
  
  result <- rddensity(X = margins, c = 0)
  summary(result)
  
  cat("\nT-statistic:", result$test$t_jk, "\n")
  cat("P-value:", result$test$p_jk, "\n")
  cat("Bandwidth left:", result$h$left, "\n")
  cat("Bandwidth right:", result$h$right, "\n")
  cat("N left:", result$N$left, "\n")
  cat("N right:", result$N$right, "\n")
  
  # Estimated densities at cutoff from each side
  f_left <- result$hat$left
  f_right <- result$hat$right
  cat("Density est. (left of cutoff):", f_left, "\n")
  cat("Density est. (right of cutoff):", f_right, "\n")
  cat("Ratio (right/left):", f_right / f_left, "\n")
  
  return(result)
}

# ============================================================
# PART A: MAIN DENSITY TESTS (same as v1)
# ============================================================
cat("\n\n########################################\n")
cat("PART A: MAIN DENSITY TESTS\n")
cat("########################################\n")

result_all <- run_density_test("margins_all.csv", "All Measures")
result_init <- run_density_test("margins_initiative.csv", "Initiatives")
result_legref <- run_density_test("margins_legislative_referendum.csv", 
                                  "Legislative Referenda")
result_popref <- run_density_test("margins_popular_referendum.csv", 
                                  "Popular Referenda")

# Summary table
cat("\n\n=== PART A SUMMARY ===\n")
cat(sprintf("%-25s  T-stat   P-value  f_left   f_right  Ratio\n", "Sample"))
cat(rep("-", 70), "\n", sep="")
for (name_label in list(
  list("All Measures", result_all),
  list("Initiatives", result_init),
  list("Legislative Referenda", result_legref),
  list("Popular Referenda", result_popref)
)) {
  r <- name_label[[2]]
  cat(sprintf("%-25s  %6.3f   %6.4f   %6.4f   %6.4f   %5.3f\n", 
      name_label[[1]], 
      r$test$t_jk, 
      r$test$p_jk,
      r$hat$left,
      r$hat$right,
      r$hat$right / r$hat$left))
}

# ============================================================
# PART B: FISCAL vs NON-FISCAL DENSITY SPLIT
# ============================================================
cat("\n\n########################################\n")
cat("PART B: FISCAL vs NON-FISCAL DENSITY SPLIT\n")
cat("(Fiscal = tax_rev OR bond_meas OR budgets)\n")
cat("########################################\n")

result_legref_fiscal <- run_density_test("margins_legref_fiscal.csv", 
                                          "Leg. Ref. - FISCAL")
result_legref_nonfiscal <- run_density_test("margins_legref_nonfiscal.csv", 
                                             "Leg. Ref. - NON-FISCAL")
result_init_fiscal <- run_density_test("margins_init_fiscal.csv", 
                                        "Initiative - FISCAL")
result_init_nonfiscal <- run_density_test("margins_init_nonfiscal.csv", 
                                           "Initiative - NON-FISCAL")

# Summary table
cat("\n\n=== PART B SUMMARY ===\n")
cat(sprintf("%-28s  T-stat   P-value  f_left   f_right  Ratio\n", "Sample"))
cat(rep("-", 75), "\n", sep="")
for (name_label in list(
  list("Leg.Ref. - FISCAL", result_legref_fiscal),
  list("Leg.Ref. - NON-FISCAL", result_legref_nonfiscal),
  list("Initiative - FISCAL", result_init_fiscal),
  list("Initiative - NON-FISCAL", result_init_nonfiscal)
)) {
  r <- name_label[[2]]
  cat(sprintf("%-28s  %6.3f   %6.4f   %6.4f   %6.4f   %5.3f\n", 
      name_label[[1]], 
      r$test$t_jk, 
      r$test$p_jk,
      r$hat$left,
      r$hat$right,
      r$hat$right / r$hat$left))
}

# ============================================================
# OVERALL SUMMARY
# ============================================================
cat("\n\n########################################\n")
cat("OVERALL SUMMARY\n")
cat("########################################\n\n")

cat(sprintf("%-28s  T-stat   P-value  Ratio\n", "Sample"))
cat(rep("-", 55), "\n", sep="")
for (name_label in list(
  list("All Measures", result_all),
  list("Initiatives", result_init),
  list("  Init. - Fiscal", result_init_fiscal),
  list("  Init. - Non-fiscal", result_init_nonfiscal),
  list("Legislative Referenda", result_legref),
  list("  Leg.Ref. - Fiscal", result_legref_fiscal),
  list("  Leg.Ref. - Non-fiscal", result_legref_nonfiscal),
  list("Popular Referenda", result_popref)
)) {
  r <- name_label[[2]]
  cat(sprintf("%-28s  %6.3f   %6.4f   %5.3f\n", 
      name_label[[1]], 
      r$test$t_jk, 
      r$test$p_jk,
      r$hat$right / r$hat$left))
}
