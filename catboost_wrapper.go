package gocatboost

/*
#cgo LDFLAGS: -L/usr/local/lib -lcatboostmodel
#include "c_api.h"
#include <stdlib.h>

static void freeCStringArray(char **ptrs, size_t n) {
	for (size_t i = 0; i < n; i++) free(ptrs[i]);
	free(ptrs);
}
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

	var floatPtrPtr **C.float
	if floatFeaturesSize > 0 {
		allFloatVals := (*C.float)(C.malloc(C.size_t(docCount*floatFeaturesSize) * C.sizeof_float))
		if allFloatVals == nil {
			return nil, errors.New("failed to allocate memory for float features")
		}
		defer C.free(unsafe.Pointer(allFloatVals))

		allFloats := unsafe.Slice(allFloatVals, docCount*floatFeaturesSize)
		for i, ff := range floatFeatures {
			base := i * floatFeaturesSize
			for j, v := range ff {
				allFloats[base+j] = C.float(v)
			}
		}

		floatRowPtrs := (**C.float)(C.malloc(C.size_t(docCount) * C.size_t(unsafe.Sizeof((*C.float)(nil)))))
		if floatRowPtrs == nil {
			return nil, errors.New("failed to allocate memory for float row pointers")
		}
		defer C.free(unsafe.Pointer(floatRowPtrs))

		rowPtrSlice := unsafe.Slice(floatRowPtrs, docCount)
		for i := range docCount {
			rowPtrSlice[i] = &allFloats[i*floatFeaturesSize]
		}
		floatPtrPtr = floatRowPtrs
	}

	var catPtrPtr ***C.char
	if catFeaturesSize > 0 {
		totalStrings := docCount * catFeaturesSize
		allCatStrPtrs := (**C.char)(C.malloc(C.size_t(totalStrings) * C.size_t(unsafe.Sizeof((*C.char)(nil)))))
		if allCatStrPtrs == nil {
			return nil, errors.New("failed to allocate memory for cat string pointers")
		}
		defer C.freeCStringArray(allCatStrPtrs, C.size_t(totalStrings))

		catStrSlice := unsafe.Slice(allCatStrPtrs, totalStrings)
		for i, cf := range catFeatures {
			base := i * catFeaturesSize
			for j, s := range cf {
				catStrSlice[base+j] = C.CString(s)
			}
		}

		catRowPtrs := (***C.char)(C.malloc(C.size_t(docCount) * C.size_t(unsafe.Sizeof((**C.char)(nil)))))
		if catRowPtrs == nil {
			return nil, errors.New("failed to allocate memory for cat row pointers")
		}
		defer C.free(unsafe.Pointer(catRowPtrs))

		catRowSlice := unsafe.Slice(catRowPtrs, docCount)
		for i := range docCount {
			catRowSlice[i] = &catStrSlice[i*catFeaturesSize]
		}
		catPtrPtr = catRowPtrs
	}

	predicts := make([]float64, docCount)
	success := C.CalcModelPrediction(
		c.model, C.size_t(docCount),
		floatPtrPtr, C.size_t(floatFeaturesSize),
		catPtrPtr, C.size_t(catFeaturesSize),
		(*C.double)(unsafe.Pointer(&predicts[0])), C.size_t(docCount),
	)
	if !bool(success) {
		return nil, errorC()
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

	cgoArray := unsafe.Slice(floatC, floatCSize)
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

	var result C.double
	success := C.CalcModelPredictionSingle(
		c.model,
		floatC,
		floatCSize,
		cCatFeaturesPtr,
		catFeaturesSize,
		&result,
		1,
	)
	if !bool(success) {
		return 0, errorC()
	}
	return float64(result), nil
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
