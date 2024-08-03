package gocatboost

import "fmt"

func ExampleFromFile() {
	cb, err := FromFile(testModelPath, WithPredictionType(Probability))
	if err != nil {
		fmt.Printf("loading model from file: %v", err)
		return
	}
	defer cb.Close()

	res, err := cb.Predict([]float64{0.5, 1.5}, []string{"a", "d", "g"})
	if err != nil {
		fmt.Printf("predict: %v", err)
		return
	}
	fmt.Println(res)
	// Output: 0.5116651937415478
}
