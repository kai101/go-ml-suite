package linearregression

import (
	"reflect"
	"testing"
)

func TestLinearRegressionE(t *testing.T) {

	c2 := LinearRegressionE{
		x: [][]float64{
			[]float64{1, 2, 3},
			[]float64{1, 4, 5},
		},
		y:     []float64{2, 3},
		theta: []float64{0, 0, 0},
		alpha: float64(5),
	}
	d1 := []float64{12.5, 40, 52.5} //1 iteration result.

	cases := []struct {
		in   *LinearRegressionE
		want []float64
	}{{&c2, d1}}

	for _, c := range cases {
		got := GradienDescentE(c.in)

		//fd1 := mat64.Formatted(got)
		t.Logf("returned value here \n%v \n", got)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("BubbleSort(%v) == %v, want %v", c.in, got, c.want)
		}
	}
}
