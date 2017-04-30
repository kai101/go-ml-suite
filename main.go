package main

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

func main() {
	a := mat64.NewDense(3, 3, []float64{1, 2, 3, 0, 4, 5, 0, 0, 6})
	//fmt.Printf("with all values:\na = %v\n\n", fa)
	//fmt.Printf("with only non-zero values:\na = % v\n\n", fa)
	//a.Add(float64(1), a)
	print("with all value", a)

	v := mat64.NewDense(0, 0, nil)
	fmt.Printf("type of v is %T\n", v)
	v.Mul(a, a)
	print("test\n", v)
}

func print(msg string, a *mat64.Dense) {

	fa := mat64.Formatted(a, mat64.Prefix("    "), mat64.Squeeze())
	fmt.Printf(msg+"\na = %v\n\n", fa)
}
