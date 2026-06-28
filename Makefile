GOCMD        = go
GOTEST       = $(GOCMD) test
GOVET        = $(GOCMD) vet
GOCLEAN      = $(GOCMD) clean
CATBOOST_DIR = /usr/local/lib
CGO_LDFLAGS  = -L$(CATBOOST_DIR) -lcatboostmodel
CGO_CFLAGS   = -I/usr/local/include
LD_ENV       = LD_LIBRARY_PATH=$(CATBOOST_DIR):$$LD_LIBRARY_PATH

BENCH      ?= .
BENCHTIME  ?= 3s
COUNT      ?= 5
BENCH_FLAGS = -run=^$$ -bench=$(BENCH) -benchmem -benchtime=$(BENCHTIME) -count=$(COUNT)

all: test

vet:
	$(LD_ENV) $(GOVET) ./...

test: vet
	$(LD_ENV) $(GOCLEAN) -testcache && $(GOTEST) ./...

clean-cache:
	$(GOCLEAN) -testcache


bench:
	$(LD_ENV) $(GOTEST) $(BENCH_FLAGS) ./...
bench-cpu:
	$(LD_ENV) $(GOTEST) $(BENCH_FLAGS) -cpuprofile=cpu.prof ./...
bench-mem:
	$(LD_ENV) $(GOTEST) $(BENCH_FLAGS) -memprofile=mem.prof ./...
bench-trace:
	$(LD_ENV) $(GOTEST) $(BENCH_FLAGS) -trace=trace.out ./...
bench-save:
	$(LD_ENV) $(GOTEST) $(BENCH_FLAGS) ./... | tee bench-baseline.txt
bench-compare:
	$(LD_ENV) $(GOTEST) $(BENCH_FLAGS) ./... > bench-new.txt
	benchstat bench-baseline.txt bench-new.txt

.PHONY: all test vet clean-cache bench bench-cpu bench-mem bench-trace bench-save bench-compare
