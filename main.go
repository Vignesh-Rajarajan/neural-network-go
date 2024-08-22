package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"math"
	"math/rand"
)

// Layer represents a single layer in a neural network
type Layer struct {
	Weights mat.Dense
	Biases  mat.Dense
}

func main() {
	//RunModel()
	x, _ := GenerateSpiralData(100, 3)
	PlotScatterData(x, 3)
}

func RunModel() {
	// inputs := [][]float64{{1, 2.0, 3.0, 2.5}, {2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}}
	inputDense := mat.NewDense(3, 4, []float64{1, 2.0, 3.0, 2.5, 2.0, 5.0, -1.0, 2.0, -1.5, 2.7, 3.3, -0.8})
	layer1 := Layer{
		Weights: *mat.NewDense(3, 4, []float64{0.2, 0.8, -0.5, 1.0, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87}),
		Biases:  *mat.NewDense(1, 3, []float64{2.0, 3.0, 0.5}),
	}
	layer1Output := ProcessLayer(inputDense, layer1)
	fmt.Println(mat.Formatted(layer1Output))
	fmt.Println()
	layer2 := Layer{
		Biases:  *mat.NewDense(1, 3, []float64{-1, 2, -0.5}),
		Weights: *mat.NewDense(3, 3, []float64{0.1, -0.14, 0.5, -0.5, 0.12, -0.33, -0.44, 0.73, -0.13}),
	}

	layer2Output := ProcessLayer(layer1Output, layer2)
	fmt.Println(mat.Formatted(layer2Output))
}

func ProcessLayer(input *mat.Dense, layer Layer) *mat.Dense {
	r1, _ := input.Dims()
	_, c2 := layer.Weights.T().Dims()
	result := mat.NewDense(r1, c2, nil)
	result.Mul(input, layer.Weights.T())
	biasReplicated := mat.NewDense(3, 3, nil)
	biasRow := layer.Biases.RawRowView(0)
	for i := 0; i < 3; i++ {
		biasReplicated.SetRow(i, biasRow)
	}
	result.Add(result, biasReplicated)
	return result
}

// GenerateSpiralData This function takes two parameters: the number of samples per class and the number of classes
// samples are actual data and classes are the colours we get if we plot it
// Data point example:
//
// A point might look like this: x[50] = [0.3, 0.7], y[50] = 0
// This means the 51st point (index 50) is at coordinates (0.3, 0.7) and belongs to class 0.
func GenerateSpiralData(samples, classes int) ([][]float64, []uint8) {
	x := make([][]float64, samples*classes)
	y := make([]uint8, samples*classes)

	// x = np.zeros((samples*classes, 2))
	//This loop initializes each inner slice of x to hold samples number of float64 values.
	for i := 0; i < len(x); i++ {
		x[i] = make([]float64, 2)
	}

	for classNumber := 0; classNumber < classes; classNumber++ {
		//This creates a slice r with values ranging from 0 to 1,
		//representing the radius of each point in the spiral.
		//the radius tells you how far each point is from the center
		r := make([]float64, samples)
		for i := range r {
			r[i] = float64(i) / float64(samples-1)
		}

		//This creates a slice t with angle values for each point.
		// if you're drawing a spiral, this determines how much you've turned for each point.
		//The angle increases as you move along the spiral,
		//and a little bit of randomness is added to make the spiral less perfect and more natural-looking
		t := make([]float64, samples)
		for i := range t {
			t[i] = float64(classNumber)*4 + float64(i)*4.0/float64(samples) + rand.NormFloat64()*0.2
		}

		// This loop calculates the x and y coordinates for each point using the radius and angle values,
		//and assigns the class number as the label.
		for ix := classNumber * samples; ix < (classNumber+1)*samples; ix++ {
			idx := ix - classNumber*samples
			// We use some math (sine and cosine) to convert the radius and angle into x and y coordinates.
			//This is like translating the spiral from polar coordinates (radius and angle) to Cartesian coordinates (x and y).
			x[ix][0] = r[idx] * math.Sin(t[idx]*2.5)
			x[ix][1] = r[idx] * math.Cos(t[idx]*2.5)
			y[ix] = uint8(classNumber)
		}
	}
	return x, y
}

func PlotScatterData(values [][]float64, classCount int) {
	x, y := make([]float64, len(values)), make([]float64, len(values))
	for i, v := range values {
		x[i], y[i] = v[0], v[1]
	}

	p := plot.New()
	p.Title.Text = "Spiral Data"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	for class := 0; class < classCount; class++ {
		offset := len(values) / classCount
		points := make(plotter.XYs, offset)
		for i := 0; i < offset; i++ {
			points[i].X = x[i+offset*class]
			points[i].Y = y[i+offset*class]
		}
		s, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		s.GlyphStyle.Color = plotutil.Color(class)
		p.Add(s)
		if err := p.Save(4*vg.Inch, 4*vg.Inch, "plots.png"); err != nil {
			panic(err)
		}
	}
}
