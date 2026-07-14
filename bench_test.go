package gocatboost

import (
	"fmt"
	"os"
	"testing"
)

var (
	sinkFloat  float64
	sinkFloats []float64
	sinkErr    error
)

func mustModel(b *testing.B) *Catboost {
	b.Helper()
	cb, err := FromFile(testModelPath)
	if err != nil {
		b.Fatal(err)
	}
	return cb
}

func mustModelBuffer(b *testing.B) []byte {
	b.Helper()
	data, err := os.ReadFile(testModelPath)
	if err != nil {
		b.Fatal(err)
	}
	return data
}

func BenchmarkFromFile(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		cb, err := FromFile(testModelPath)
		if err != nil {
			b.Fatal(err)
		}
		cb.Close()
	}
}

func BenchmarkFromBuffer(b *testing.B) {
	data := mustModelBuffer(b)
	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		cb, err := FromBuffer(data)
		if err != nil {
			b.Fatal(err)
		}
		cb.Close()
	}
}

func BenchmarkPredict(b *testing.B) {
	cb := mustModel(b)
	defer cb.Close()

	floats := []float64{0.5, 1.5}
	cats := []string{"a", "d", "g"}

	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		sinkFloat, sinkErr = cb.Predict(floats, cats)
	}
}

func BenchmarkPredictCatStringAlloc(b *testing.B) {
	cb := mustModel(b)
	defer cb.Close()

	floats := []float64{0.5, 1.5}

	cases := []struct {
		name string
		cats []string
	}{
		{"short/3", []string{"a", "d", "g"}},
		{"long/3", []string{"category_value_one", "category_value_two", "category_value_three"}},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				sinkFloat, sinkErr = cb.Predict(floats, tc.cats)
			}
		})
	}
}

func BenchmarkPredictParallel(b *testing.B) {
	cb := mustModel(b)
	defer cb.Close()

	floats := []float64{0.5, 1.5}
	cats := []string{"a", "d", "g"}

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v, err := cb.Predict(floats, cats)
			sinkFloat = v
			sinkErr = err
		}
	})
}

func BenchmarkPredictBatch(b *testing.B) {
	cb := mustModel(b)
	defer cb.Close()

	baseFloats := []float64{0.5, 1.5}
	baseCats := []string{"a", "d", "g"}

	for _, docCount := range []int{1, 10, 100, 1000} {
		floatsBatch := make([][]float64, docCount)
		catsBatch := make([][]string, docCount)
		for i := range docCount {
			floatsBatch[i] = baseFloats
			catsBatch[i] = baseCats
		}

		b.Run(fmt.Sprintf("docs=%d", docCount), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(docCount))
			for b.Loop() {
				sinkFloats, sinkErr = cb.PredictBatch(floatsBatch, catsBatch)
			}
		})
	}
}

func BenchmarkPredictBatchVsPredict(b *testing.B) {
	cb := mustModel(b)
	defer cb.Close()

	baseFloats := []float64{0.5, 1.5}
	baseCats := []string{"a", "d", "g"}

	for _, docCount := range []int{10, 100} {
		floatsBatch := make([][]float64, docCount)
		catsBatch := make([][]string, docCount)
		for i := range docCount {
			floatsBatch[i] = baseFloats
			catsBatch[i] = baseCats
		}

		b.Run(fmt.Sprintf("Batch/docs=%d", docCount), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(docCount))
			for b.Loop() {
				sinkFloats, sinkErr = cb.PredictBatch(floatsBatch, catsBatch)
			}
		})

		b.Run(fmt.Sprintf("Sequential/docs=%d", docCount), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(docCount))
			results := make([]float64, docCount)
			for b.Loop() {
				for i := range docCount {
					results[i], sinkErr = cb.Predict(baseFloats, baseCats)
				}
				sinkFloats = results
			}
		})
	}
}

func BenchmarkPredictBatchParallel(b *testing.B) {
	cb := mustModel(b)
	defer cb.Close()

	baseFloats := []float64{0.5, 1.5}
	baseCats := []string{"a", "d", "g"}

	for _, docCount := range []int{1, 100} {
		floatsBatch := make([][]float64, docCount)
		catsBatch := make([][]string, docCount)
		for i := range docCount {
			floatsBatch[i] = baseFloats
			catsBatch[i] = baseCats
		}

		b.Run(fmt.Sprintf("docs=%d", docCount), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(docCount))
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					v, err := cb.PredictBatch(floatsBatch, catsBatch)
					sinkFloats = v
					sinkErr = err
				}
			})
		})
	}
}

func BenchmarkWithPredictionType(b *testing.B) {
	data := mustModelBuffer(b)
	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		cb, err := FromBuffer(data, WithPredictionType(Probability))
		if err != nil {
			b.Fatal(err)
		}
		cb.Close()
	}
}
