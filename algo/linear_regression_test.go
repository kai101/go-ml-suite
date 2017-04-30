package linearregression

import (
	"reflect"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestLinearRegression(t *testing.T) {
	c1 := LinearRegression{
		mat64.NewDense(2, 3, []float64{1, 2, 3, 1, 4, 5}),
		mat64.NewDense(2, 1, []float64{2, 3}),
		mat64.NewDense(3, 1, []float64{0, 0, 0}),
		float64(5)}

	d1 := mat64.NewDense(3, 1, []float64{12.5, 40, 52.5}) //1 iteration result.

	cases := []struct {
		in   *LinearRegression
		want *mat64.Dense
	}{{&c1, d1}}

	for _, c := range cases {
		got := GradienDescent(c.in)

		fd1 := mat64.Formatted(got)
		t.Logf("returned value here \n%v \n", fd1)
		//TODO: comparing for the test case.
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("BubbleSort(%v) == %v, want %v", c.in, got, c.want)
		}
	}
}
