// Package gocatboost provides fast CatBoost model inference via CGO
// bindings to the official CatBoost C API (libcatboostmodel).
//
// Every Predict and PredictBatch call crosses the CGO boundary exactly
// once: inputs are packed into flat Go buffers, pinned with
// runtime.Pinner and passed directly to C, with no C heap allocations
// on the hot path.
//
// The CatBoost shared library and its C header must be installed before
// building, see the README for instructions.
//
// A minimal usage example:
//
//	cb, err := gocatboost.FromFile("model.cbm")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer cb.Close()
//
//	result, err := cb.Predict([]float64{0.5, 1.5}, []string{"a", "d", "g"})
package gocatboost
