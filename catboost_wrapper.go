package gocatboost

/*
#cgo LDFLAGS: -L/usr/local/lib -lcatboostmodel
#include "c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type Option func(*Catboost) error

func WithPredictionType(pt PredictionType) Option {
	return func(c *Catboost) error {
		return c.SetPredictionType(pt)
	}
}

type Catboost struct {
	model unsafe.Pointer
}

func makeCatboost() *Catboost {
	return &Catboost{model: C.ModelCalcerCreate()}
}

func FromBuffer(b []byte, opts ...Option) (*Catboost, error) {
	cb := makeCatboost()

	if !C.LoadFullModelFromBuffer(cb.model, unsafe.Pointer(&b[0]), C.size_t(len(b))) {
		cb.Close()
		return nil, fmt.Errorf("load full model from buffer: %w", errorC())
	}

	if err := cb.applyOptions(opts...); err != nil {
		cb.Close()
		return nil, err
	}

	return cb, nil
}

func FromFile(modelPath string, opts ...Option) (*Catboost, error) {
	cb := makeCatboost()
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	if !C.LoadFullModelFromFile(cb.model, cPath) {
		cb.Close()
		return nil, fmt.Errorf("load full model from file: %w", errorC())
	}

	if err := cb.applyOptions(opts...); err != nil {
		cb.Close()
		return nil, err
	}

	return cb, nil
}

func (c *Catboost) SetPredictionType(pt PredictionType) error {
	cStr := C.CString(pt.String())
	defer C.free(unsafe.Pointer(cStr))
	if !C.SetPredictionTypeString(c.model, cStr) {
		return errorC()
	}
	return nil
}

func (c *Catboost) FeaturesCount() (int, int) {
	floats := C.GetFloatFeaturesCount(c.model)
	categorical := C.GetCatFeaturesCount(c.model)
	return int(floats), int(categorical)
}

func (c *Catboost) Predict(floatFeatures []float64, catFeatures []string) (float64, error) {
	return c.predict(floatFeatures, catFeatures)
}

func (c *Catboost) PredictBatch(floatFeatures [][]float64, catFeatures [][]string) ([]float64, error) {
	docCount := len(floatFeatures)
	if docCount == 0 {
		return nil, errors.New("float features must not be empty")
	}
	if len(catFeatures) != docCount {
		return nil, fmt.Errorf("cat features length %d does not match floatFeatures length %d", len(catFeatures), docCount)
	}

	floatFeaturesSize := len(floatFeatures[0])
	catFeaturesSize := len(catFeatures[0])
	for i := 1; i < docCount; i++ {
		if len(floatFeatures[i]) != floatFeaturesSize {
			return nil, fmt.Errorf("float features[%d] has length %d, expected %d", i, len(floatFeatures[i]), floatFeaturesSize)
		}
		if len(catFeatures[i]) != catFeaturesSize {
			return nil, fmt.Errorf("cat features[%d] has length %d, expected %d", i, len(catFeatures[i]), catFeaturesSize)
		}
	}

	floatPtrs := make([]*C.float, docCount)
	defer func() {
		for _, ptr := range floatPtrs {
			C.free(unsafe.Pointer(ptr))
		}
	}()
	for i, ff := range floatFeatures {
		if floatFeaturesSize == 0 {
			break
		}
		floatPtrs[i] = (*C.float)(C.malloc(C.size_t(floatFeaturesSize) * C.sizeof_float))
		if floatPtrs[i] == nil {
			return nil, fmt.Errorf("failed to allocate memory for floatFeatures[%d]", i)
		}
		cgoArray := (*[1 << 30]C.float)(unsafe.Pointer(floatPtrs[i]))[:floatFeaturesSize:floatFeaturesSize]
		for j, v := range ff {
			cgoArray[j] = C.float(v)
		}
	}

	catPtrs := make([]**C.char, docCount)
	defer func() {
		for _, ptr := range catPtrs {
			if ptr == nil {
				continue
			}
			cgoArray := (*[1 << 30]*C.char)(unsafe.Pointer(ptr))[:catFeaturesSize:catFeaturesSize]
			for _, s := range cgoArray {
				C.free(unsafe.Pointer(s))
			}
			C.free(unsafe.Pointer(ptr))
		}
	}()
	for i, cf := range catFeatures {
		if catFeaturesSize == 0 {
			break
		}
		catPtrs[i] = (**C.char)(C.malloc(C.size_t(catFeaturesSize) * C.size_t(unsafe.Sizeof(uintptr(0)))))
		if catPtrs[i] == nil {
			return nil, fmt.Errorf("failed to allocate memory for catFeatures[%d]", i)
		}
		cgoArray := (*[1 << 30]*C.char)(unsafe.Pointer(catPtrs[i]))[:catFeaturesSize:catFeaturesSize]
		for j, s := range cf {
			cgoArray[j] = C.CString(s)
		}
	}

	result := (*C.double)(C.malloc(C.size_t(docCount) * C.sizeof_double))
	if result == nil {
		return nil, errors.New("failed to allocate memory for result")
	}
	defer C.free(unsafe.Pointer(result))

	var floatPtrPtr **C.float
	if floatFeaturesSize > 0 {
		floatPtrPtr = (**C.float)(unsafe.Pointer(&floatPtrs[0]))
	}
	var catPtrPtr ***C.char
	if catFeaturesSize > 0 {
		catPtrPtr = (***C.char)(unsafe.Pointer(&catPtrs[0]))
	}

	success := C.CalcModelPrediction(
		c.model, C.size_t(docCount),
		floatPtrPtr, C.size_t(floatFeaturesSize),
		catPtrPtr, C.size_t(catFeaturesSize),
		result, C.size_t(docCount),
	)
	if !bool(success) {
		return nil, errorC()
	}

	predicts := make([]float64, docCount)
	cgoResult := (*[1 << 30]C.double)(unsafe.Pointer(result))[:docCount:docCount]
	for i := range cgoResult {
		predicts[i] = float64(cgoResult[i])
	}
	return predicts, nil
}

func (c *Catboost) predict(floatFeatures []float64, catFeatures []string) (float64, error) {
	floatCSize := C.size_t(len(floatFeatures))
	floatC := (*C.float)(C.malloc(C.sizeof_float * floatCSize))
	if floatC == nil {
		return 0, fmt.Errorf("failed to allocate memory for floatFeatures")
	}
	defer C.free(unsafe.Pointer(floatC))

	cgoArray := (*[1 << 30]C.float)(unsafe.Pointer(floatC))[:floatCSize:floatCSize]
	for i, v := range floatFeatures {
		cgoArray[i] = C.float(v)
	}

	catFeaturesSize := C.size_t(len(catFeatures))
	cCatFeatures := make([]*C.char, len(catFeatures))
	for i, s := range catFeatures {
		cCatFeatures[i] = C.CString(s)
	}
	defer func() {
		for _, s := range cCatFeatures {
			C.free(unsafe.Pointer(s))
		}
	}()
	var cCatFeaturesPtr **C.char
	if len(cCatFeatures) > 0 {
		cCatFeaturesPtr = (**C.char)(unsafe.Pointer(&cCatFeatures[0]))
	}

	resultC := (*C.double)(C.malloc(C.sizeof_double))
	if resultC == nil {
		return 0, fmt.Errorf("failed to allocate memory for result")
	}
	defer C.free(unsafe.Pointer(resultC))

	success := C.CalcModelPredictionSingle(
		c.model,
		floatC,
		floatCSize,
		cCatFeaturesPtr,
		catFeaturesSize,
		resultC,
		1,
	)
	if !bool(success) {
		return 0, errorC()
	}
	return float64(*resultC), nil
}

func (c *Catboost) Close() {
	C.ModelCalcerDelete(c.model)
	c.model = nil
}

func (c *Catboost) applyOptions(opts ...Option) error {
	for _, opt := range opts {
		if err := opt(c); err != nil {
			return err
		}
	}
	return nil
}

func errorC() error {
	errorString := C.GetErrorString()
	message := C.GoString(errorString)
	return errors.New(message)
}
