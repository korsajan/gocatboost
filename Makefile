GOCMD=go
GOTEST=$(GOCMD) test
GOVET=$(GOCMD) vet
GOCLEAN=$(GOCMD) clean 
CATBOOST_DIR=/usr/local/lib
CGO_LDFLAGS=-L$(CATBOOST_DIR) -lcatboostmodel
CGO_CFLAGS=-I/usr/local/include


all: test

vet:
	LD_LIBRARY_PATH=$(CATBOOST_DIR):$$LD_LIBRARY_PATH  $(GOVET) ./...

test: vet clean-cache
	LD_LIBRARY_PATH=$(CATBOOST_DIR):$$LD_LIBRARY_PATH $(GOCLEAN) -testcache && $(GOTEST) ./...
	
clean-cache:
	$(GOCLEAN) -testcache

.PHONY: all test vet clean-cache
