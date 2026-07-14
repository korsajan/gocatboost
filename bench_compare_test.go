//go:build bench_compare

package gocatboost

import (
	"fmt"
	"os"
	"testing"

	mirecl "github.com/mirecl/catboost-cgo/catboost"
)

func mustModelMirecl(b *testing.B) *mirecl.Model {
	b.Helper()
	m, err := mirecl.LoadFullModelFromFile(testModelPath)
	if err != nil {
		b.Fatalf("mirecl load: %v", err)
	}
	return m
}

func mustModelBufferMirecl(b *testing.B) []byte {
	b.Helper()
	data, err := os.ReadFile(testModelPath)
	if err != nil {
		b.Fatal(err)
	}
	return data
}

func BenchmarkCompare_FromFile(b *testing.B) {
	b.Run("impl=ours", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cb, err := FromFile(testModelPath)
			if err != nil {
				b.Fatal(err)
			}
			cb.Close()
		}
	})
	b.Run("impl=mirecl", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			m, err := mirecl.LoadFullModelFromFile(testModelPath)
			if err != nil {
				b.Fatal(err)
			}
			m.Delete()
		}
	})
}

func BenchmarkCompare_FromBuffer(b *testing.B) {
	data := mustModelBuffer(b)
	dataMirecl := mustModelBufferMirecl(b)

	b.Run("impl=ours", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for b.Loop() {
			cb, err := FromBuffer(data)
			if err != nil {
				b.Fatal(err)
			}
			cb.Close()
		}
	})
	b.Run("impl=mirecl", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for b.Loop() {
			m, err := mirecl.LoadFullModelFromBuffer(dataMirecl)
			if err != nil {
				b.Fatal(err)
			}
			m.Delete()
		}
	})
}

func BenchmarkCompare_PredictSingle(b *testing.B) {
	floats64 := []float64{0.5, 1.5}
	floats32 := []float32{0.5, 1.5}
	cats := []string{"a", "d", "g"}

	b.Run("impl=ours", func(b *testing.B) {
		cb := mustModel(b)
		defer cb.Close()
		b.ResetTimer()
		b.ReportAllocs()
		for b.Loop() {
			sinkFloat, sinkErr = cb.Predict(floats64, cats)
		}
	})
	b.Run("impl=mirecl", func(b *testing.B) {
		m := mustModelMirecl(b)
		defer m.Delete()
		b.ResetTimer()
		b.ReportAllocs()
		for b.Loop() {
			sinkFloats, sinkErr = m.PredictSingle(floats32, cats)
		}
	})
}

func BenchmarkCompare_PredictBatch(b *testing.B) {
	baseFloats64 := []float64{0.5, 1.5}
	baseFloats32 := []float32{0.5, 1.5}
	baseCats := []string{"a", "d", "g"}

	for _, docCount := range []int{1, 10, 100, 1000} {
		floatsBatch64 := make([][]float64, docCount)
		floatsBatch32 := make([][]float32, docCount)
		catsBatch := make([][]string, docCount)
		for i := range docCount {
			floatsBatch64[i] = baseFloats64
			floatsBatch32[i] = baseFloats32
			catsBatch[i] = baseCats
		}

		b.Run(fmt.Sprintf("docs=%d/impl=ours", docCount), func(b *testing.B) {
			cb := mustModel(b)
			defer cb.Close()
			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(docCount))
			for b.Loop() {
				sinkFloats, sinkErr = cb.PredictBatch(floatsBatch64, catsBatch)
			}
		})
		b.Run(fmt.Sprintf("docs=%d/impl=mirecl", docCount), func(b *testing.B) {
			m := mustModelMirecl(b)
			defer m.Delete()
			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(docCount))
			for b.Loop() {
				sinkFloats, sinkErr = m.Predict(floatsBatch32, catsBatch)
			}
		})
	}
}

func BenchmarkCompare_PredictParallel(b *testing.B) {
	floats64 := []float64{0.5, 1.5}
	floats32 := []float32{0.5, 1.5}
	cats := []string{"a", "d", "g"}

	b.Run("impl=ours", func(b *testing.B) {
		cb := mustModel(b)
		defer cb.Close()
		b.ResetTimer()
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				v, err := cb.Predict(floats64, cats)
				sinkFloat = v
				sinkErr = err
			}
		})
	})
	b.Run("impl=mirecl", func(b *testing.B) {
		m := mustModelMirecl(b)
		defer m.Delete()
		b.ResetTimer()
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				v, err := m.PredictSingle(floats32, cats)
				sinkFloats = v
				sinkErr = err
			}
		})
	})
}
