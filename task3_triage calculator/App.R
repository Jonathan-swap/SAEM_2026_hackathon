# SAEM26 Hackathon — Task 3 triage calculator.
# Enter triage vitals once. App predicts the likely drug
# and shows a clean vitals summary to paste into the EMR.
# Run from repo root: R -e "shiny::runApp('task3_triage calculator/App.R')"

library(shiny)
library(here)

# ---- Load model + scaling exported from Task 1 ----
TASK1_OUT <- here("task1_drug_identifier", "out")
coefs   <- read.csv(file.path(TASK1_OUT, "model_coefficients.csv"))
scaling <- read.csv(file.path(TASK1_OUT, "feature_scaling.csv"))

# Cluster-to-drug mapping (from Step 2 cluster profiles)
# ADJUST to whatever cluster labels we ID
DRUG_LABELS <- c("0" = "Kraken Candy",
                 "1" = "Coral Dust",
                 "2" = "Triton Tabs")

# Predict drug probabilities given a named vector of raw vitals
predict_drug <- function(vitals) {
  # Scale: (x - mean) / sd
  z <- (vitals[scaling$feature] - scaling$mean) / scaling$sd
  # Linear scores per class
  feature_cols <- scaling$feature
  scores <- coefs$intercept +
            as.matrix(coefs[, feature_cols]) %*% as.numeric(z)
  # Softmax to probabilities
  exp_scores <- exp(scores - max(scores))
  probs <- exp_scores / sum(exp_scores)
  setNames(as.numeric(probs), DRUG_LABELS[as.character(coefs$cluster)])
}

# ---- UI ----
ui <- fluidPage(
  titlePanel("SAEM Triage Drug Calculator"),
  sidebarLayout(
    sidebarPanel(
      h4("Vitals"),
      numericInput("hr",   "Heart rate (bpm)",         value = 100),
      numericInput("rr",   "Respiratory rate (/min)",  value = 20),
      numericInput("sbp",  "Systolic BP (mmHg)",       value = 120),
      numericInput("dbp",  "Diastolic BP (mmHg)",      value = 75),
      numericInput("spo2", "SpO2 (%)",                 value = 96),
      numericInput("temp", "Temperature (°F)",         value = 98.6),
      numericInput("gcs",  "GCS",                      value = 15),
      numericInput("pain", "Pain (0–10)",              value = 5),
    ),
    mainPanel(
      h4("Predicted drug"),
      tableOutput("probs"),
      h4("Vitals summary (copy to EMR)"),
      tags$div(
        style = "display: inline-block;",
        verbatimTextOutput("summary")
      ),
    )
  )
)

# ---- Server ----
server <- function(input, output) {
  # Build the named vital vector reactively (model was trained on Celsius)
  vitals <- reactive({
    c(triage_heart_rate                = input$hr,
      triage_respiratory_rate          = input$rr,
      triage_snapshot.systolic_bp      = input$sbp,
      triage_snapshot.diastolic_bp     = input$dbp,
      triage_snapshot.oxygen_saturation= input$spo2,
      triage_temperature_c             = (input$temp - 32) * 5 / 9,
      triage_gcs                       = input$gcs,
      triage_pain_scale                = input$pain)
  })

  output$probs <- renderTable({
    p <- predict_drug(vitals())
    data.frame(Drug = names(p),
               Probability = sprintf("%.1f%%", 100 * p))
  })

  output$summary <- renderText({
    sprintf(paste(
      "HR %d  RR %d  BP %d/%d  SpO2 %d%%",
      "Temp %.1fF  GCS %d  Pain %d/10",
      sep = "\n"
    ), input$hr, input$rr, input$sbp, input$dbp, input$spo2,
       input$temp, input$gcs, input$pain)
  })
}

shinyApp(ui, server)