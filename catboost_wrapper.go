package gocatboost

/*
#cgo LDFLAGS: -L/usr/local/lib -lcatboostmodel
#cgo noescape CalcModelPrediction
#cgo nocallback CalcModelPrediction
#cgo noescape CalcModelPredictionSingle
#cgo nocallback CalcModelPredictionSingle
#cgo noescape LoadFullModelFromBuffer
#cgo nocallback LoadFullModelFromBuffer
#cgo noescape LoadFullModelFromFile
#cgo nocallback LoadFullModelFromFile
#cgo noescape SetPredictionTypeString
#cgo nocallback SetPredictionTypeString
#cgo nocallback ModelCalcerCreate
#cgo nocallback ModelCalcerDelete
#cgo nocallback GetErrorString
#cgo nocallback GetFloatFeaturesCount
#cgo nocallback GetCatFeaturesCount
#include "c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Option configures a Catboost model during construction.
type Option func(*Catboost) error

// WithPredictionType returns an Option that sets the prediction type
// applied by Predict and PredictBatch.
func WithPredictionType(pt PredictionType) Option {
	return func(c *Catboost) error {
		return c.SetPredictionType(pt)
	}
}

// Catboost is a loaded CatBoost model handle. It is safe for
// concurrent use by multiple goroutines. Callers must release the
// underlying C model with Close.
type Catboost struct {
	model unsafe.Pointer
}

func makeCatboost() *Catboost {
	return &Catboost{model: C.ModelCalcerCreate()}
}

// FromBuffer loads a CatBoost model from an in-memory model file
// representation, such as the contents of a .cbm file.
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

// FromFile loads a CatBoost model from a .cbm file on disk.
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

// SetPredictionType sets the prediction type applied by Predict and
// PredictBatch.
func (c *Catboost) SetPredictionType(pt PredictionType) error {
	cStr := C.CString(pt.String())
	defer C.free(unsafe.Pointer(cStr))
	if !C.SetPredictionTypeString(c.model, cStr) {
		return errorC()
	}
	return nil
}

// FeaturesCount returns the number of float and categorical features
// the model expects.
func (c *Catboost) FeaturesCount() (int, int) {
	floats := C.GetFloatFeaturesCount(c.model)
	categorical := C.GetCatFeaturesCount(c.model)
	return int(floats), int(categorical)
}

// Predict returns the model prediction for a single document described
// by its float and categorical features.
func (c *Catboost) Predict(floatFeatures []float64, catFeatures []string) (float64, error) {
	return c.predict(floatFeatures, catFeatures)
}

// PredictBatch returns model predictions for a batch of documents in a
// single CGO call. Every row of floatFeatures must have the same
// length, and likewise for catFeatures.
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

	var pinner runtime.Pinner
	defer pinner.Unpin()

	var floatRows []*C.float
	if floatFeaturesSize > 0 {
		flat := make([]C.float, docCount*floatFeaturesSize)
		pinner.Pin(&flat[0])
		for i, ff := range floatFeatures {
			base := i * floatFeaturesSize
			for j, v := range ff {
				flat[base+j] = C.float(v)
			}
		}
		floatRows = make([]*C.float, docCount)
		for i := range docCount {
			floatRows[i] = &flat[i*floatFeaturesSize]
		}
	}

	var catRows []**C.char
	if catFeaturesSize > 0 {
		strPtrs := make([]*C.char, docCount*catFeaturesSize)
		pinner.Pin(&strPtrs[0])
		total := 0
		for _, cf := range catFeatures {
			for _, s := range cf {
				total += len(s) + 1
			}
		}
		buf := make([]byte, total)
		pinner.Pin(&buf[0])
		off, k := 0, 0
		for _, cf := range catFeatures {
			for _, s := range cf {
				strPtrs[k] = (*C.char)(unsafe.Pointer(&buf[off]))
				off += copy(buf[off:], s)
				buf[off] = 0
				off++
				k++
			}
		}
		catRows = make([]**C.char, docCount)
		for i := range docCount {
			catRows[i] = &strPtrs[i*catFeaturesSize]
		}
	}

	var floatPtrPtr **C.float
	if floatRows != nil {
		floatPtrPtr = &floatRows[0]
	}
	var catPtrPtr ***C.char
	if catRows != nil {
		catPtrPtr = &catRows[0]
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
	floats := make([]C.float, len(floatFeatures))
	for i, v := range floatFeatures {
		floats[i] = C.float(v)
	}
	var floatsPtr *C.float
	if len(floats) > 0 {
		floatsPtr = &floats[0]
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()

	var catPtrsPtr **C.char
	if len(catFeatures) > 0 {
		catPtrs := make([]*C.char, len(catFeatures))
		pinCStrings(&pinner, catFeatures, catPtrs)
		catPtrsPtr = &catPtrs[0]
	}

	var result C.double
	success := C.CalcModelPredictionSingle(
		c.model,
		floatsPtr,
		C.size_t(len(floatFeatures)),
		catPtrsPtr,
		C.size_t(len(catFeatures)),
		&result,
		1,
	)
	if !bool(success) {
		return 0, errorC()
	}
	return float64(result), nil
}

// Close releases the underlying C model. The Catboost value must not
// be used after Close.
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

func pinCStrings(pinner *runtime.Pinner, strs []string, ptrs []*C.char) {
	total := 0
	for _, s := range strs {
		total += len(s) + 1
	}
	buf := make([]byte, total)
	pinner.Pin(&buf[0])
	off := 0
	for i, s := range strs {
		ptrs[i] = (*C.char)(unsafe.Pointer(&buf[off]))
		off += copy(buf[off:], s)
		buf[off] = 0
		off++
	}
}
