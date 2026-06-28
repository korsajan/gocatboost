package gocatboost

type PredictionType int

const (
	RawFormulaVal PredictionType = iota
	Exponent
	RMSEWithUncertainty
	Probability
	Class
	MultiProbability
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

func (pt PredictionType) String() string {
	return predictionTypeToString[pt]
}
