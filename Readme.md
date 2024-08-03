# GoCatBoost
GoCatBoost is a wrapper for the CatBoost library, enabling the use of CatBoost in Go.

[![Go Tests](https://github.com/korsajan/gocatboost/actions/workflows/go_test.yml/badge.svg?branch=main)](https://github.com/korsajan/gocatboost/actions/workflows/go_test.yml)


## Install
```sh
# download the header file c_api.h:
wget https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model_interface/c_api.h -O /usr/local/include/c_api.h
# download the compiled library: 
export ARG_VERSION=1.2.5 && wget https://github.com/catboost/catboost/releases/download/v${ARG_VERSION}/libcatboostmodel.so -O /usr/local/lib/libcatboostmodel.so
# updating the dynamic library cache
sudo ldconfig
# install 
go get github.com/korsajan/gocatboost
```

# Documentation
For detailed documentation on CatBoost, please refer to the [CatBoost Documentation](https://catboost.ai/en/docs/concepts/installation).