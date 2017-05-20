package linearregression

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

type TestF struct {
	a *mat64.Dense
	b *mat64.Dense
	c *mat64.Dense
	d float64
}

type LinearRegression struct {
	x     *mat64.Dense
	y     *mat64.Dense
	theta *mat64.Dense
	alpha float64
}

func ComputeLinearRegression(input *LinearRegression, iter int) {
	i := 0
	for ; i < iter; i++ {
		GradienDescent(input)
	}
}

func GradienDescent(input *LinearRegression) *mat64.Dense {
	//calculate hypothesis
	hypothesis := mat64.NewDense(0, 0, nil)
	hypothesis.Mul(input.x, input.theta)

	//calculate error base on hypothesis and expected output y
	err := mat64.NewDense(0, 0, nil)
	err.Sub(hypothesis, input.y)

	//calculate theta_change
	xTransposed := mat64.NewDense(0, 0, nil)
	xTransposed.Clone(input.x.T())
	theta_change := mat64.NewDense(0, 0, nil)
	theta_change.Mul(xTransposed, err)
	yRows, _ := input.y.Dims()
	theta_change.Scale((input.alpha / float64(yRows)), theta_change)

	//get final theta
	input.theta.Sub(input.theta, theta_change)
	return input.theta
}

func prd(i *mat64.Dense) {
	ferr := mat64.Formatted(i)
	fmt.Printf("test package %v", ferr)

}
