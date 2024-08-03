package gocatboost

type PredictionType int

const (
    RawFormulaVal PredictionType = iota
    Exponent
    RMSEWithUncertainty
    Probability
    Class
    MultiProbability
)

var predictionTypeToString = map[PredictionType]string{
    RawFormulaVal:       "RawFormulaVal",
    Exponent:            "Exponent",
    RMSEWithUncertainty: "RMSEWithUuncertainty",
    Probability:         "Probability",
    Class:               "Class",
    MultiProbability:    "MultiProbability",
}

func (pt PredictionType) String() string {
    return predictionTypeToString[pt]
}
