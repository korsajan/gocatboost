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

| Benchmark               | gocatboost | mirecl    | diff             |
|-------------------------|-----------:|----------:|------------------|
| Predict (single)        | 962 ns     | 1374 ns   | ours 1.4x faster |
| PredictBatch docs=1     | 1220 ns    | 1794 ns   | ours 1.5x faster |
| PredictBatch docs=10    | 2404 ns    | 6171 ns   | ours 2.6x faster |
| PredictBatch docs=100   | 14.5 us    | 51.1 us   | ours 3.5x faster |
| PredictBatch docs=1000  | 124 us     | 575 us    | ours 4.6x faster |
| Predict (parallel)      | 230 ns     | 227 ns    | equal            |

| Allocations             | gocatboost      | mirecl          |
|-------------------------|-----------------|-----------------|
| Predict (single)        | 1 alloc, 8 B    | 2 allocs, 16 B  |
| PredictBatch docs=100   | 6 allocs        | 1 alloc         |

Why prediction is faster: every Predict and PredictBatch call crosses the CGO
boundary exactly once. Inputs are packed into flat Go buffers that are pinned
with runtime.Pinner and passed directly to C, so there are no C.malloc,
C.CString or C.free calls at all (mirecl pays a CGO transition per row
allocation and per string). Predictions are written by CatBoost directly into
the returned Go slice. The #cgo noescape and #cgo nocallback annotations
(Go 1.24+) further cut the per-call overhead and let small feature arrays stay
on the stack. PredictBatch trades a few cheap Go allocations for the removal
of hundreds of CGO calls.

Note: mirecl takes float32 input, so float64 data must be converted before calling.
This library takes float64 natively.

To reproduce: `make bench-vs-mirecl`. Full suite: `make bench`.

## Documentation

[CatBoost Documentation](https://catboost.ai/en/docs/concepts/installation)
