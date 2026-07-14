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

// single document
result, err := cb.Predict([]float64{0.5, 1.5}, []string{"a", "d", "g"})

// batch of documents, one CGO call for the whole batch
results, err := cb.PredictBatch(
    [][]float64{{0.5, 1.5}, {0.3, 2.1}},
    [][]string{{"a", "d", "g"}, {"b", "e", "h"}},
)
```

## Benchmarks vs mirecl/catboost-cgo

CPU: AMD Ryzen 7 4800H (16 threads), Linux 6.8.0, Go 1.26, CatBoost 1.2.7.
Flags: -benchtime=3s, -count=5, median of 5 runs.

| Benchmark               | gocatboost | mirecl    | diff            |
|-------------------------|-----------:|----------:|-----------------|
| Predict (single)        | 1349 ns    | 1357 ns   | equal           |
| PredictBatch docs=1     | 1483 ns    | 1780 ns   | ours 1.2x faster |
| PredictBatch docs=10    | 3475 ns    | 6148 ns   | ours 1.8x faster |
| PredictBatch docs=100   | 30.6 us    | 50.3 us   | ours 1.6x faster |
| PredictBatch docs=1000  | 283 us     | 577 us    | ours 2.0x faster |
| Predict (parallel)      | 240 ns     | 230 ns    | equal           |

| Allocations             | gocatboost      | mirecl          |
|-------------------------|-----------------|-----------------|
| PredictBatch (any size) | 1 alloc/op      | 1 alloc/op      |
| Predict (single)        | 2 allocs, 32 B  | 2 allocs, 16 B  |

Why PredictBatch is faster: the whole batch uses 4 flat C allocations regardless
of document count (mirecl allocates per row), all C strings are freed in a single
CGO call instead of one call per string, and predictions are written by CatBoost
directly into the returned Go slice with no intermediate buffer or conversion loop.

Note: mirecl takes float32 input, so float64 data must be converted before calling.
This library takes float64 natively.

To reproduce: `make bench-vs-mirecl`. Full suite: `make bench`.

## Documentation

[CatBoost Documentation](https://catboost.ai/en/docs/concepts/installation)
