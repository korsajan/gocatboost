package gocatboost

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testModelPath = "./testdata/model.cbm"

func TestFromFile(t *testing.T) {
	cb, err := FromFile(testModelPath)
	require.NoError(t, err)

	f, c := cb.FeaturesCount()
	assert.EqualValues(t, f, 1)
	assert.EqualValues(t, c, 3)
	cb.Close()
}

func TestFromBuffer(t *testing.T) {
	b, err := os.ReadFile(testModelPath)
	require.NoError(t, err)

	cb, err := FromBuffer(b)
	require.NoError(t, err)
	f, c := cb.FeaturesCount()
	assert.EqualValues(t, f, 1)
	assert.EqualValues(t, c, 3)
	cb.Close()
}

func TestSetPredictType(t *testing.T) {
	cb, err := FromFile(testModelPath, WithPredictionType(Probability))
	require.NoError(t, err)
	defer cb.Close()
}

func TestPredictModel(t *testing.T) {
	floats := []float64{0.5, 1.5}
	cats := []string{"a", "d", "g"}

	cb, err := FromFile(testModelPath)
	require.NoError(t, err)
	defer cb.Close()

	predict, err := cb.Predict(floats, cats)
	require.NoError(t, err)
	require.EqualValues(t, 0.04666924366060905, predict)

	predicts, err := cb.PredictBatch([][]float64{floats}, [][]string{cats})
	require.NoError(t, err)
	require.Len(t, predicts, 1)
	require.EqualValues(t, 0.04666924366060905, predicts[0])
}

func TestPredictBatchModel(t *testing.T) {
	floats := []float64{0.5, 1.5}
	cats := []string{"a", "d", "g"}

	cb, err := FromFile(testModelPath)
	require.NoError(t, err)
	defer cb.Close()

	predicts, err := cb.PredictBatch([][]float64{floats}, [][]string{cats})
	require.NoError(t, err)
	require.Len(t, predicts, 1)
	require.EqualValues(t, 0.04666924366060905, predicts[0])
}

func TestFromFileNotFound(t *testing.T) {
	cb, err := FromFile("./testdata/no_such_model.cbm")
	require.Error(t, err)
	require.Nil(t, cb)
}

func TestFromBufferInvalid(t *testing.T) {
	cb, err := FromBuffer([]byte("not a catboost model"))
	require.Error(t, err)
	require.Nil(t, cb)
}

func TestFromFileInvalidOption(t *testing.T) {
	cb, err := FromFile(testModelPath, WithPredictionType(PredictionType(99)))
	require.Error(t, err)
	require.Nil(t, cb)
}

func TestPredictBatchValidation(t *testing.T) {
	cb, err := FromFile(testModelPath)
	require.NoError(t, err)
	defer cb.Close()

	_, err = cb.PredictBatch(nil, nil)
	require.Error(t, err)

	_, err = cb.PredictBatch([][]float64{{0.5, 1.5}}, [][]string{{"a", "d", "g"}, {"b", "e", "h"}})
	require.Error(t, err)

	_, err = cb.PredictBatch([][]float64{{0.5, 1.5}, {0.3}}, [][]string{{"a", "d", "g"}, {"b", "e", "h"}})
	require.Error(t, err)

	_, err = cb.PredictBatch([][]float64{{0.5, 1.5}, {0.3, 2.1}}, [][]string{{"a", "d", "g"}, {"b", "e"}})
	require.Error(t, err)
}
