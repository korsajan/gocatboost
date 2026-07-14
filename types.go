package gocatboost

// PredictionType selects how raw model scores are transformed into
// the values returned by Predict and PredictBatch. It mirrors the
// prediction types of the CatBoost C API.
type PredictionType int

const (
	// RawFormulaVal returns the raw model score.
	RawFormulaVal PredictionType = iota
	// Exponent returns the exponent of the raw score.
	Exponent
	// RMSEWithUncertainty returns the prediction with uncertainty for
	// models trained with the RMSEWithUncertainty loss.
	RMSEWithUncertainty
	// Probability transforms the raw score into a class probability.
	Probability
	// Class returns the predicted class label.
	Class
	// MultiProbability returns per-class probabilities for
	// multiclassification models.
	MultiProbability
	// MultiClassSoftmax returns softmax-normalized scores for
	// multiclassification models.
	MultiClassSoftmax
)

var predictionTypeToString = map[PredictionType]string{
	RawFormulaVal:       "RawFormulaVal",
	Exponent:            "Exponent",
	RMSEWithUncertainty: "RMSEWithUncertainty",
	Probability:         "Probability",
	Class:               "Class",
	MultiProbability:    "MultiProbability",
	MultiClassSoftmax:   "MultiClassSoftmax",
}

// String returns the CatBoost C API name of the prediction type.
func (pt PredictionType) String() string {
	return predictionTypeToString[pt]
}
