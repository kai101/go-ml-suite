package linearregression

type LinearRegressionE struct {
	X     [][]float64
	Y     []float64
	Theta []float64
	Alpha float64
}

func ComputeLinearRegressionE(input *LinearRegressionE, iter int) {
	i := 0
	for ; i < iter; i++ {
		GradienDescentE(input)
	}
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
			thetaChange[i] += input.X[j][i] * err[j]
		}
	}

	for i := 0; i < len(thetaChange); i++ {
		thetaChange[i] = thetaChange[i] * (input.Alpha / float64(yLength))
	}

	//fmt.Printf("thetachange %v = xtranspose %v * err \n", thetaChange, input.X)

	result := make([]float64, len(input.Theta))

	for i, v := range thetaChange {
		result[i] = input.Theta[i] - v
	}

	//fmt.Printf("result %v = input.Theta %v - thetha change \n", result, input.Theta)

	return result
}
