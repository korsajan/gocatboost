# GoCatBoost

Go wrapper for CatBoost inference via CGO bindings to the official C API.

[![Go Tests](https://github.com/korsajan/gocatboost/actions/workflows/go_test.yml/badge.svg?branch=main)](https://github.com/korsajan/gocatboost/actions/workflows/go_test.yml)

## Install

```sh
wget https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model_interface/c_api.h \
  -O /usr/local/include/c_api.h

export ARG_VERSION=1.2.7
wget https://github.com/catboost/catboost/releases/download/v${ARG_VERSION}/libcatboostmodel.so \
  -O /usr/local/lib/libcatboostmodel.so

sudo ldconfig
go get github.com/korsajan/gocatboost
```

## Usage

```go
cb, err := gocatboost.FromFile("model.cbm", gocatboost.WithPredictionType(gocatboost.Probability))
if err != nil {
    log.Fatal(err)
}
defer cb.Close()

result, err := cb.Predict([]float64{0.5, 1.5}, []string{"a", "d", "g"})

results, err := cb.PredictBatch(
    [][]float64{{0.5, 1.5}, {0.3, 2.1}},
    [][]string{{"a", "d", "g"}, {"b", "e", "h"}},
)
```

## Benchmarks

CPU: AMD Ryzen 7 4800H (16 threads), Linux 6.8.0, Go 1.26, CatBoost 1.2.7.

```
BenchmarkFromFile-16                                      21249   112985 ns/op      8 B/op    1 allocs/op
BenchmarkFromBuffer-16                                    23775   101318 ns/op      8 B/op    1 allocs/op
BenchmarkPredict-16                                     1759825     1361 ns/op     24 B/op    1 allocs/op
BenchmarkPredictCatStringAlloc/short/3-16               1759994     1359 ns/op     24 B/op    1 allocs/op
BenchmarkPredictCatStringAlloc/long/3-16                1750573     1380 ns/op     24 B/op    1 allocs/op
BenchmarkPredictParallel-16                            11076445      217 ns/op     24 B/op    1 allocs/op
BenchmarkPredictBatch/docs=1-16                         1475469     1652 ns/op     24 B/op    3 allocs/op
BenchmarkPredictBatch/docs=10-16                         363076     6593 ns/op    240 B/op    3 allocs/op
BenchmarkPredictBatch/docs=100-16                         42525    57709 ns/op   2688 B/op    3 allocs/op
BenchmarkPredictBatch/docs=1000-16                         4284   571084 ns/op  24576 B/op    3 allocs/op
BenchmarkPredictBatchVsPredict/Batch/docs=10-16          353916     7014 ns/op    240 B/op    3 allocs/op
BenchmarkPredictBatchVsPredict/Sequential/docs=10-16     172299    14299 ns/op    240 B/op   10 allocs/op
BenchmarkPredictBatchVsPredict/Batch/docs=100-16          41966    57020 ns/op   2688 B/op    3 allocs/op
BenchmarkPredictBatchVsPredict/Sequential/docs=100-16     17320   138242 ns/op   2400 B/op  100 allocs/op
BenchmarkPredictBatchParallel/docs=1-16                 9765358      240 ns/op     24 B/op    3 allocs/op
BenchmarkPredictBatchParallel/docs=100-16                299054     7602 ns/op   2688 B/op    3 allocs/op
BenchmarkWithPredictionType-16                            22905   104170 ns/op      8 B/op    1 allocs/op
```

To reproduce: `make bench`

## Documentation

[CatBoost Documentation](https://catboost.ai/en/docs/concepts/installation)
