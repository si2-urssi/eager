# Install all deps

# Remotes and AwardFindR
if (!require("remotes")) {
  install.packages("remotes", repos = "http://cran.us.r-project.org")
}
remotes::install_github("ropensci/awardFindR")