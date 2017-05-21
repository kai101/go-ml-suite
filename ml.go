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
	c4.Theta = make([]float64, len(c4.X[0]))
	c4.Alpha = 0.01
	iteration := 400
	_, mean, stdd := linearregression.Normalize(&c4)
	fullResult := linearregression.ComputeLinearRegressionE(&c4, iteration)
	fmt.Printf("full result is %v \n", fullResult)

	//prediction
	given := []float64{1650, 3}
	for i, v := range given {
		given[i] = (v - mean[i]) / stdd[i]
	}
	form := make([]float64, len(given)+1)
	form[0] = 1
	copy(form[1:len(form)], given[:])
	prediction := float64(0)

	for i, v := range form {
		prediction += v * c4.Theta[i]
	}

	fmt.Printf("prediction of 1650,3 is %v \n", prediction)

	//fmt.Printf("x is %v , y is %v \n", c4.X, c4.Y)
	//v := mat64.NewDense(0, 0, nil)
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

	//guarding yIndex
	if yIndex > (len(data[0]) - 1) {
		log.Fatal()
	}
	y = make([]float64, len(data))
	for i, v := range cloneData {
		y[i] = v[yIndex]
		//fmt.Printf("before %v \n", v)
		copy(v[1:yIndex+1], v[:yIndex])
		//fmt.Printf("after %v \n", v)
		//v[len(v)-1] = 0 // or the zero value of T
		//v = v[:len(v)-1]
		v[0] = 1
		cloneData[i] = v
	}

	if len(cloneData[0]) != (xNo + 1) {
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
