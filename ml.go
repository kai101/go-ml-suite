package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"ml/algo/linearregression"
	"os"
	"strconv"

	"github.com/gonum/matrix/mat64"
)

func main() {
	//a := mat64.NewDense(3, 3, []float64{1, 2, 3, 0, 4, 5, 0, 0, 6})
	//fmt.Printf("with all values:\na = %v\n\n", fa)
	//fmt.Printf("with only non-zero values:\na = % v\n\n", fa)
	//a.Add(float64(1), a)
	//print("with all value", a)
	data2 := "./test_data/ex1data2.txt"
	c3 := readData(data2)
	c4 := formLinearE(c3, 2, 2)
	fmt.Printf("x is %v , y is %v \n", c4.X, c4.Y)
	//v := mat64.NewDense(0, 0, nil)
	c2 := linearregression.LinearRegressionE{
		X: [][]float64{
			[]float64{1, 2, 3},
			[]float64{1, 4, 5},
		},
		Y:     []float64{2, 3},
		Theta: []float64{5, 5, 5},
		Alpha: float64(5),
	}

	v := linearregression.GradienDescentE(&c2)

	fmt.Printf("output is  %v\n", v)
	//v.Mul(a, a)
	//print("test\n", v)
}

func print(msg string, a *mat64.Dense) {

	fa := mat64.Formatted(a, mat64.Prefix("    "), mat64.Squeeze())
	fmt.Printf(msg+"\na = %v\n\n", fa)
}

func formLinearE(data [][]float64, xNo, yIndex int) linearregression.LinearRegressionE {
	var cloneData [][]float64
	var y []float64
	cloneData = make([][]float64, len(data))
	copy(cloneData, data)

	y = make([]float64, len(data))
	for i, v := range cloneData {
		y[i] = v[yIndex]
		copy(v[yIndex:], v[yIndex+1:])
		v[len(v)-1] = 0 // or the zero value of T
		v = v[:len(v)-1]
		cloneData[i] = v
	}

	if len(cloneData[0]) != xNo {
		log.Fatal()
	}

	return linearregression.LinearRegressionE{
		X: cloneData,
		Y: y,
	}
}

func readData(path string) [][]float64 {
	f, err := os.Open(path)
	check(err)
	defer f.Close()

	reader := bufio.NewReader(f)
	csvReader := csv.NewReader(reader)

	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	//fmt.Printf("record: %v, type: %T, recordEle: %v,  recordType: %T\n", records, records, records[0][0], records[0][0])
	var result [][]float64
	for _, v := range records {
		row := make([]float64, len(v))
		for j, vv := range v {
			row[j], err = strconv.ParseFloat(vv, 64)
		}
		result = append(result, row)
	}
	return result
	/*
		fmt.Printf("result: %v, type: %T, resultEle: %v,  resultType: %T\n", result, result, result[0][0], result[0][0])

		var line string
		for {
			var buffer bytes.Buffer

			var l []byte
			var isPrefix bool
			for {
				l, isPrefix, err = reader.ReadLine()
				buffer.Write(l)

				// If we've reached the end of the line, stop reading.
				if !isPrefix {
					break
				}

				// If we're just at the EOF, break
				if err != nil {
					break
				}
			}

			if err == io.EOF {
				break
			}

			line = buffer.String()

			fmt.Printf("line we received : \"%v\"\n", line)
		}
	*/
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}
