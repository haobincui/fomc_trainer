#!/usr/bin/env Rscript

# Run the exact ordinalForest estimator configuration used by Yoon and Fan
# (2024) on a caller-supplied reduced predictor matrix.  This runner emits
# point classes only: ordinalForest 2.4-3 intentionally returns classprobs=NA
# when perffunction="equal".

options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
  stop(
    paste(
      "usage: run_yoon_fan_ordinal_forest.R",
      "TRAIN_CSV TEST_CSV PREDICTIONS_CSV DIAGNOSTICS_CSV SEED NUM_THREADS"
    )
  )
}

train_path <- args[[1L]]
test_path <- args[[2L]]
predictions_path <- args[[3L]]
diagnostics_path <- args[[4L]]
seed <- suppressWarnings(as.integer(args[[5L]]))
num_threads <- suppressWarnings(as.integer(args[[6L]]))

if (is.na(seed) || seed < 0L) {
  stop("SEED must be a non-negative integer")
}
if (is.na(num_threads) || num_threads != 1L) {
  stop("NUM_THREADS must be exactly 1 for the frozen deterministic contract")
}

suppressPackageStartupMessages(library(ordinalForest))
if (as.character(packageVersion("ordinalForest")) != "2.4.3") {
  stop("ordinalForest package version must be exactly 2.4-3")
}

features <- c(
  "tbill6m_minus_effr_pp",
  "unemployment_rate_pct",
  "real_gdp_growth_annualized_pct"
)
class_order <- c("cut", "hold", "hike")

train <- read.csv(
  train_path,
  check.names = FALSE,
  stringsAsFactors = FALSE,
  colClasses = "character"
)
test <- read.csv(
  test_path,
  check.names = FALSE,
  stringsAsFactors = FALSE,
  colClasses = "character"
)

required_train <- c("row_id", "direction", features)
required_test <- c("row_id", features)
if (!identical(names(train), required_train)) {
  stop("training CSV columns do not match the frozen allowlist")
}
if (!identical(names(test), required_test)) {
  stop("test CSV columns do not match the frozen allowlist")
}
if (nrow(train) != 211L || nrow(test) != 19L) {
  stop("training/test row counts must be exactly 211/19")
}
if (anyDuplicated(train$row_id) || anyDuplicated(test$row_id)) {
  stop("row_id must be unique within each input")
}
if (length(intersect(train$row_id, test$row_id)) != 0L) {
  stop("training and test row_id sets must be disjoint")
}

for (feature in features) {
  train[[feature]] <- suppressWarnings(as.numeric(train[[feature]]))
  test[[feature]] <- suppressWarnings(as.numeric(test[[feature]]))
}
if (!all(is.finite(as.matrix(train[features])))) {
  stop("training features must all be finite")
}
if (!all(is.finite(as.matrix(test[features])))) {
  stop("test features must all be finite")
}

train$direction <- factor(
  train$direction,
  levels = class_order,
  ordered = TRUE
)
if (anyNA(train$direction)) {
  stop("training direction contains a value outside cut/hold/hike")
}
expected_counts <- c(cut = 23L, hold = 156L, hike = 32L)
if (!identical(as.integer(table(train$direction)), as.integer(expected_counts))) {
  stop("training class distribution is not 23/156/32")
}

set.seed(seed)
fit <- ordinalForest::ordfor(
  depvar = "direction",
  data = train[c("direction", features)],
  nsets = 1000L,
  ntreeperdiv = 100L,
  ntreefinal = 5000L,
  importance = "rps",
  perffunction = "equal",
  nbest = 10L,
  naive = FALSE,
  num.threads = num_threads,
  npermtrial = 500L,
  permperdefault = FALSE,
  mtry = 1L,
  min.node.size = 5L,
  replace = TRUE,
  sample.fraction = 1,
  keep.inbag = FALSE
)

prediction <- predict(fit, newdata = test[features])
if (length(prediction$ypred) != nrow(test)) {
  stop("ordinalForest returned an unexpected prediction count")
}
if (!(length(prediction$classprobs) == 1L && is.na(prediction$classprobs))) {
  stop("equal-performance probability-output contract drifted")
}
predicted_direction <- as.character(prediction$ypred)
if (!all(predicted_direction %in% class_order)) {
  stop("ordinalForest returned a class outside cut/hold/hike")
}

predictions <- data.frame(
  row_id = test$row_id,
  predicted_direction = predicted_direction,
  probability_status = rep(
    "unavailable_equal_performance_author_configuration",
    nrow(test)
  ),
  stringsAsFactors = FALSE
)

diagnostic_names <- c(
  "r_version",
  "ordinalForest_version",
  "seed",
  "class_order",
  "nsets",
  "ntreeperdiv",
  "ntreefinal",
  "importance",
  "perffunction",
  "nbest",
  "naive",
  "num_threads",
  "npermtrial",
  "permperdefault",
  "mtry",
  "min_node_size",
  "replace",
  "sample_fraction",
  "keep_inbag",
  "probability_output",
  "optimized_borders",
  "variable_importance"
)
diagnostic_values <- c(
  as.character(getRversion()),
  as.character(packageVersion("ordinalForest")),
  as.character(seed),
  paste(class_order, collapse = "<"),
  as.character(fit$nsets),
  as.character(fit$ntreeperdiv),
  as.character(fit$ntreefinal),
  "rps",
  as.character(fit$perffunction),
  as.character(fit$nbest),
  "false",
  as.character(num_threads),
  "500",
  "false",
  "1",
  "5",
  "true",
  "1",
  "false",
  "NA_under_equal",
  paste(format(fit$bordersbest, digits = 17L), collapse = ";"),
  paste(
    paste(names(fit$varimp), format(fit$varimp, digits = 17L), sep = "="),
    collapse = ";"
  )
)
diagnostics <- data.frame(
  name = diagnostic_names,
  value = diagnostic_values,
  stringsAsFactors = FALSE
)

write.csv(predictions, predictions_path, row.names = FALSE, quote = TRUE)
write.csv(diagnostics, diagnostics_path, row.names = FALSE, quote = TRUE)

cat(
  paste0(
    "ordinalForest run complete: train=", nrow(train),
    " test=", nrow(test),
    " seed=", seed,
    " package=", as.character(packageVersion("ordinalForest")),
    " probability_output=NA_under_equal\n"
  )
)
