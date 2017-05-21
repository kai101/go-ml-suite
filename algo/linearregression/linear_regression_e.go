package linearregression

import (
	"math"

	"github.com/montanaflynn/stats"
)

type LinearRegressionE struct {
	X     [][]float64
	Y     []float64
	Theta []float64
	Alpha float64
}

func ComputeLinearRegressionE(input *LinearRegressionE, iter int) *LinearRegressionE {
	normX, _, _ := Normalize(input)
	//fmt.Printf("check normalization %v \n", normX)
	oriX := input.X
	input.X = normX
	i := 0
	for ; i < iter; i++ {
		newTheta := GradienDescentE(input)
		//fmt.Printf("check Theta %v \n", newTheta)
		input.Theta = newTheta
	}

	input.X = oriX
	return input
}

func GradienDescentE(input *LinearRegressionE) []float64 {
	//calculate hypothesis
	var hypothesis []float64
	yLength := len(input.Y)
	hypothesis = make([]float64, yLength, yLength)
	for i, v := range input.X {
		for j, vv := range v {
			hypothesis[i] += vv * input.Theta[j]
		}
	}

	//fmt.Printf("hypothesis operation X %v Theta %v hypo %v\n", input.X, input.Theta, hypothesis)

	//calculate error base on hypothesis and expected output y
	var err []float64
	err = make([]float64, yLength, yLength)

	for i, v := range hypothesis {
		err[i] = v - input.Y[i]
	}

	//fmt.Printf("error %v =  hypo - y %v \n", err, input.Y)

	//calculate theta_change
	var thetaChange []float64
	thetaChange = make([]float64, len(input.Theta))
	rowCount := len(input.X)
	varCount := len(input.X[0])
	for i := 0; i < varCount; i++ {
		for j := 0; j < rowCount; j++ {
			thetaChange[i] += (input.Alpha / float64(yLength)) * input.X[j][i] * err[j]
			//fmt.Printf("check i%v, j%v, thetac%v, x%v, err%v\n", i, j, thetaChange[i], input.X[j][i], err[j])
		}
	}

	//for i := 0; i < len(thetaChange); i++ {
	//	thetaChange[i] = thetaChange[i] * (input.Alpha / float64(yLength))
	//}

	//fmt.Printf("thetachange %v = xtranspose %v * err \n", thetaChange, input.X)

	result := make([]float64, len(input.Theta))

	for i, v := range thetaChange {
		result[i] = input.Theta[i] - v
	}

	//fmt.Printf("result %v = input.Theta %v - thetha change \n", result, input.Theta)

	return result
}

func Normalize(input *LinearRegressionE) ([][]float64, []float64, []float64) {

	var features [][]float64
	//init
	features = make([][]float64, len(input.X[0])-1)

	for _, v := range input.X {
		for j, vv := range v {
			if j == 0 {
				continue
			}
			features[j-1] = append(features[j-1], vv)
		}
	}

	//fmt.Printf("check features %v\n", features)

	copy(features, features)
	var mean []float64
	mean = make([]float64, len(features))
	var stdd []float64
	stdd = make([]float64, len(features))
	for i, v := range features {
		mean[i], _ = stats.Mean(v)
		variance, _ := stats.SampleVariance(v)
		stdd[i] = math.Sqrt(variance)
	}

	//fmt.Printf("check stdd%v \n median%v\n", stdd, mean)

	var normX [][]float64
	normX = make([][]float64, len(input.X))
	copy(normX, input.X)
	for i, v := range normX {
		for j, vv := range v {
			if j == 0 {
				continue
			}
			normX[i][j] = (vv - mean[j-1]) / stdd[j-1]
		}
	}
	//fmt.Printf("check normX%v \n", normX)
	return normX, mean, stdd
}
