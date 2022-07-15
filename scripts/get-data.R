# Imports
library(awardFindR)

# Pull targets
nsf <- awardFindR::get_nsf("BIO", "2019-01-01", "2022-07-01")
write.csv(nsf, "data/keyword-search.csv", row.names = FALSE)