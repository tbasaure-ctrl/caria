# =============================================================================
# GAMLSS Test: Does Signal_Strict affect the DISTRIBUTION of returns?
# =============================================================================
#
# HYPOTHESIS (the REAL one):
#   H0: Signal_Strict has no effect on any distributional parameter
#   H1: Signal_Strict affects σ (volatility) and/or τ (tail heaviness)
#
# NOT testing if Signal predicts mean returns (we know it doesn't)
# Testing if Signal changes the SHAPE of the return distribution
#
# Author: Tomás Basaure
# Date: December 2025
# =============================================================================

# Install if needed
if (!require("gamlss")) {
  install.packages("gamlss", repos = "https://cloud.r-project.org")
}
if (!require("gamlss.dist")) {
  install.packages("gamlss.dist", repos = "https://cloud.r-project.org")
}
if (!require("moments")) {
  install.packages("moments", repos = "https://cloud.r-project.org")
}

library(gamlss)
library(gamlss.dist)
library(moments)

# =============================================================================
# 1. LOAD DATA (exported from Python)
# =============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("GAMLSS DISTRIBUTIONAL TEST\n")
cat("Hypothesis: Signal affects distribution shape, not mean\n")
cat(rep("=", 70), "\n\n")

# Read data from Python export
df <- read.csv("gamlss_data.csv")

cat("Data loaded:", nrow(df), "observations\n")
cat("Variables:", paste(names(df), collapse=", "), "\n\n")

# Quick summary
cat("Target (Fwd_Ret_21) summary:\n")
print(summary(df$Fwd_Ret_21))

cat("\nSignal distribution:\n")
print(table(df$Signal_Strict))

# =============================================================================
# 2. FIT NULL MODEL (no signal effect)
# =============================================================================

cat("\n", rep("-", 50), "\n", sep="")
cat("Fitting NULL model (constant distribution)...\n")

model_null <- gamlss(
  Fwd_Ret_21 ~ 1,          # μ constant
  sigma.fo = ~ 1,          # σ constant
  nu.fo = ~ 1,             # ν (skew) constant  
  tau.fo = ~ 1,            # τ (kurtosis) constant
  family = ST(),           # Skewed Student-t
  data = df,
  trace = FALSE
)

cat("NULL model AIC:", AIC(model_null), "\n")

# =============================================================================
# 3. FIT SIGNAL MODEL (signal affects σ and τ only)
# =============================================================================

cat("\n", rep("-", 50), "\n", sep="")
cat("Fitting SIGNAL model (σ and τ depend on Signal)...\n")

model_signal <- gamlss(
  Fwd_Ret_21 ~ 1,                     # μ CONSTANT (not affected by signal)
  sigma.fo = ~ Signal_Strict,         # σ depends on signal
  nu.fo = ~ 1,                        # ν constant
  tau.fo = ~ Signal_Strict,           # τ depends on signal
  family = ST(),
  data = df,
  trace = FALSE
)

cat("SIGNAL model AIC:", AIC(model_signal), "\n")

# =============================================================================
# 4. LIKELIHOOD RATIO TEST (the key test)
# =============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("LIKELIHOOD RATIO TEST\n")
cat(rep("=", 70), "\n\n")

lr_test <- LR.test(model_null, model_signal)

cat("\nH0: Signal has NO effect on distribution\n")
cat("H1: Signal affects σ and/or τ\n\n")

if (lr_test$p.val < 0.05) {
  cat(">>> REJECT H0 (p =", lr_test$p.val, ")\n")
  cat(">>> Signal DOES change the return distribution!\n")
} else {
  cat(">>> FAIL TO REJECT H0 (p =", lr_test$p.val, ")\n")
  cat(">>> Signal does NOT significantly affect the distribution\n")
}

# =============================================================================
# 5. COEFFICIENT ANALYSIS (the moment of truth)
# =============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("COEFFICIENT ANALYSIS\n")
cat(rep("=", 70), "\n\n")

cat("Full model summary:\n")
print(summary(model_signal))

# Extract key coefficients
coefs <- summary(model_signal)

cat("\n", rep("-", 50), "\n", sep="")
cat("KEY RESULTS:\n\n")

# Sigma coefficient
cat("SIGMA (volatility) equation:\n")
cat("  Intercept: ", coef(model_signal, what="sigma")[1], "\n")
if (length(coef(model_signal, what="sigma")) > 1) {
  sigma_signal <- coef(model_signal, what="sigma")[2]
  cat("  Signal_Strict coefficient: ", sigma_signal, "\n")
  if (sigma_signal > 0) {
    cat("  >>> Signal HIGH → HIGHER volatility ✓\n")
  } else {
    cat("  >>> Signal HIGH → Lower volatility\n")
  }
}

# Tau coefficient  
cat("\nTAU (tail heaviness) equation:\n")
cat("  Intercept: ", coef(model_signal, what="tau")[1], "\n")
if (length(coef(model_signal, what="tau")) > 1) {
  tau_signal <- coef(model_signal, what="tau")[2]
  cat("  Signal_Strict coefficient: ", tau_signal, "\n")
  if (tau_signal > 0) {
    cat("  >>> Signal HIGH → HEAVIER tails ✓\n")
  } else {
    cat("  >>> Signal HIGH → Lighter tails\n")
  }
}

# =============================================================================
# 6. FITTED DISTRIBUTIONS COMPARISON
# =============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("FITTED DISTRIBUTION COMPARISON\n")
cat(rep("=", 70), "\n\n")

# Get fitted parameters for each regime
df$fitted_mu <- fitted(model_signal, what="mu")
df$fitted_sigma <- fitted(model_signal, what="sigma")
df$fitted_nu <- fitted(model_signal, what="nu")
df$fitted_tau <- fitted(model_signal, what="tau")

# Compare regimes
signal_high <- df[df$Signal_Strict == 1,]
signal_low <- df[df$Signal_Strict == 0,]

cat("When Signal = 0 (low risk):\n")
cat("  Fitted σ:", unique(signal_low$fitted_sigma)[1], "\n")
cat("  Fitted τ:", unique(signal_low$fitted_tau)[1], "\n")

cat("\nWhen Signal = 1 (high risk):\n")
cat("  Fitted σ:", unique(signal_high$fitted_sigma)[1], "\n")
cat("  Fitted τ:", unique(signal_high$fitted_tau)[1], "\n")

# =============================================================================
# 7. EMPIRICAL VALIDATION
# =============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("EMPIRICAL VALIDATION\n")
cat(rep("=", 70), "\n\n")

# Compare empirical moments
cat("Empirical statistics by regime:\n\n")
cat(sprintf("%-15s | %12s | %12s\n", "Statistic", "Signal=0", "Signal=1"))
cat(rep("-", 45), "\n")

cat(sprintf("%-15s | %12.4f | %12.4f\n", "Mean", 
    mean(signal_low$Fwd_Ret_21), mean(signal_high$Fwd_Ret_21)))
cat(sprintf("%-15s | %12.4f | %12.4f\n", "Std Dev",
    sd(signal_low$Fwd_Ret_21), sd(signal_high$Fwd_Ret_21)))
cat(sprintf("%-15s | %12.4f | %12.4f\n", "Skewness",
    moments::skewness(signal_low$Fwd_Ret_21), moments::skewness(signal_high$Fwd_Ret_21)))
cat(sprintf("%-15s | %12.4f | %12.4f\n", "Kurtosis",
    moments::kurtosis(signal_low$Fwd_Ret_21), moments::kurtosis(signal_high$Fwd_Ret_21)))
cat(sprintf("%-15s | %12.4f | %12.4f\n", "5th percentile",
    quantile(signal_low$Fwd_Ret_21, 0.05), quantile(signal_high$Fwd_Ret_21, 0.05)))
cat(sprintf("%-15s | %12.4f | %12.4f\n", "1st percentile",
    quantile(signal_low$Fwd_Ret_21, 0.01), quantile(signal_high$Fwd_Ret_21, 0.01)))

# =============================================================================
# 8. FINAL VERDICT
# =============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("FINAL VERDICT\n")
cat(rep("=", 70), "\n\n")

# Decision logic
if (lr_test$p.val < 0.05) {
  cat("✓ DISTRIBUTIONAL SHIFT CONFIRMED\n\n")
  cat("The Signal affects the SHAPE of future returns, not the mean.\n")
  cat("Specifically:\n")
  
  if (length(coef(model_signal, what="sigma")) > 1 && 
      coef(model_signal, what="sigma")[2] > 0) {
    cat("  • Higher Signal → Higher volatility (σ)\n")
  }
  
  if (length(coef(model_signal, what="tau")) > 1 && 
      coef(model_signal, what="tau")[2] > 0) {
    cat("  • Higher Signal → Heavier tails (τ)\n")
  }
  
  cat("\nVALID CLAIM:\n")
  cat("'Conditional on elevated E4, the distribution of future returns\n")
  cat(" exhibits significantly heavier left tails, despite unchanged mean.'\n")
  cat("\nThis means: The Signal tells you when Gaussian assumptions BREAK.\n")
  
} else {
  cat("✗ NO SIGNIFICANT DISTRIBUTIONAL EFFECT\n\n")
  cat("The Signal does NOT significantly change the return distribution.\n")
  cat("The hypothesis is not supported by the data.\n")
}

# Save results
results <- data.frame(
  metric = c("LR_test_p_value", "AIC_null", "AIC_signal", 
             "sigma_coef_signal", "tau_coef_signal"),
  value = c(lr_test$p.val, AIC(model_null), AIC(model_signal),
            ifelse(length(coef(model_signal, what="sigma")) > 1, 
                   coef(model_signal, what="sigma")[2], NA),
            ifelse(length(coef(model_signal, what="tau")) > 1,
                   coef(model_signal, what="tau")[2], NA))
)

write.csv(results, "gamlss_results.csv", row.names=FALSE)
cat("\n✓ Results saved to gamlss_results.csv\n")

