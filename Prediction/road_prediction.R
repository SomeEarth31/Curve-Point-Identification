#### File contains your prediction function.
# Keep any .Rdata files you need in this folder too
library(tidyverse)

predict_curve <- function(dat) {
  # Step 1: Define file paths
  input_csv <- "input.csv"
  output_csv <- "output.csv"
  python_script <- "Run_Script.py"
  
  # Step 2: Write the input dataframe to a CSV file
  write.csv(dat, file = input_csv, row.names = FALSE)
  
  # Step 3: Run the Python script
  command <- sprintf("python %s", python_script)
  system(command)
  
  if (file.exists(output_csv)) {
    predict <- read.csv(output_csv)
  } else {
    stop("Output CSV file not found. Check if the Python script ran successfully.")
  }
  
  return(list(predict = predict))
}
